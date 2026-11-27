"""Learn glyphs from ordinary handwriting (no template) plus a transcript of what it says.

Pipeline: ink map -> straighten -> drop ruled lines -> split into text lines -> split each line
into "atoms" (ink pieces separated by blank columns) -> align atoms with the transcript by
dynamic programming (a character = one or more neighbouring atoms) -> cut out each glyph.
"""
import cv2
import numpy as np

from . import charsets
from .extract import PX, darkness
from .template import BASELINE

WORK_WIDTH = 2000          # images are normalised to at most this width before analysis
MAX_ATOMS_PER_CHAR = 8
HANGUL_MM = 9.0            # learned glyphs are scaled so a typical syllable is this tall
WIDTH_COST_LIMIT = 2.0     # characters whose ink width is this far off are not learned

NARROW = set("iljtfrI1!|:;.,'\"`")
WIDE = set("mwMW@%")
BASELINE_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZacemnorsuvwxz0123456789")


def expected_width(ch):
    """Typical ink width of ``ch`` as a fraction of the line's character height."""
    if charsets.is_syllable(ch):
        return 0.85
    if charsets.is_jamo(ch):
        return 0.55
    if ch in ".,'`":
        return 0.12
    if ch in NARROW:
        return 0.22
    if ch in WIDE:
        return 0.8
    if ch.isupper():
        return 0.65
    if ch.isalnum():
        return 0.5
    return 0.4


# --------------------------------------------------------------------------- page analysis

def _ink(image):
    """Opacity map (0..1) of the ink, plus the darkness map it came from."""
    dark = darkness(image)
    values = dark[dark > 0.06]
    if values.size < 100:
        return np.zeros_like(dark), dark
    t, _ = cv2.threshold((values * 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t / 255.0, 0.12)
    strong = np.percentile(values[values > t], 70) if (values > t).any() else t + 0.2
    lo, hi = t * 0.8, max(t * 0.8 + 0.1, (t + strong) / 2)
    return np.clip((dark - lo) / (hi - lo), 0, 1), dark


def _skew_angle(binary):
    small = cv2.resize(binary, (600, max(1, int(binary.shape[0] * 600 / binary.shape[1]))),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = small.shape
    best, best_score = 0.0, -1.0
    for angle in np.arange(-6, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(small, M, (w, h))
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best, best_score = angle, score
    return best


def _remove_rules(alpha, dark):
    """Erase printed ruled/margin lines but keep pen strokes that cross them."""
    binary = (alpha > 0.3).astype(np.uint8)
    h, w = binary.shape
    # Printed lines run across much of the image; a short crop of one line of writing must not
    # have its letters' strokes mistaken for them, hence the floor on the length.
    length = max(w // 8, h // 6, 120)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    rules = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0
    ink_level = np.percentile(dark[binary > 0], 75) if binary.any() else 1.0
    alpha = alpha.copy()
    alpha[rules & (dark < ink_level * 0.85)] = 0
    return alpha


class Line:
    def __init__(self, labels, comps, alpha):
        self.labels, self.comps, self.alpha = labels, comps, alpha
        heights = np.array([c[3] for c in comps], float)
        self.h = float(np.percentile(heights, 85)) if len(heights) else 1.0
        self.atoms = []


def _segment_lines(alpha, single=False):
    """Text lines of an ink map; ``single`` when the image is known to hold one line (a band
    of a copy sheet), so exponents and indices stay with their line."""
    binary = (alpha > 0.35).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    H, W = binary.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 6:
            comps.append((i, x, y, h, w, area, cents[i][1]))
    if not comps:
        return [], labels
    # Page edges, shadows and frames: far bigger than a letter and than a good share of the
    # image (a crop of a single line is mostly letter-sized, so both tests are needed).
    typical = float(np.median([c[3] for c in comps]))
    comps = [c for c in comps if not ((c[3] > 0.25 * H and c[3] > 4 * typical) or
                                      (c[4] > 0.5 * W and c[4] > 6 * typical))]
    if not comps:
        return [], labels
    if single:
        return [Line(labels, comps, alpha)], labels
    size = float(np.median([c[3] for c in comps if c[5] >= np.median([d[5] for d in comps])]))
    # Vertical density of ink-piece centres: one peak per text line, even when tall letters of
    # neighbouring lines touch (a plain row projection would merge those lines).
    profile = np.zeros(H, np.float32)
    for i, x, y, h, w, area, cy in comps:
        if area >= 0.05 * size * size:
            profile[int(cy)] += w
    pitch = _line_pitch(profile, size)
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 0), max(1.0, 0.18 * pitch)).ravel()
    peaks = _peaks(smooth, 0.55 * pitch, 0.05 * smooth.max())  # short lines carry little ink
    if not peaks:
        return [], labels
    cuts = [0] + [a + int(np.argmin(smooth[a:b])) for a, b in zip(peaks, peaks[1:])] + [H]
    groups = [[] for _ in peaks]
    for comp in comps:
        j = int(np.searchsorted(cuts, comp[6], side="right")) - 1
        groups[min(max(j, 0), len(peaks) - 1)].append(comp)
    # Graphs, axes and underlines are far larger than the letters around them: not text.
    for g in groups:
        if len(g) >= 3:
            ref = float(np.percentile([c[3] for c in g], 75))
            g[:] = [c for c in g if c[3] <= 2.2 * ref and c[4] <= 4.5 * ref]
    return [Line(labels, g, alpha) for g in groups if g], labels


def _line_pitch(profile, size):
    """Distance between text lines from the autocorrelation of the ink profile."""
    p = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 0), max(1.0, 0.25 * size)).ravel()
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    if ac[0] <= 0:
        return 2.0 * size
    ac = ac / ac[0]
    lo, hi = int(1.1 * size), min(len(ac) - 1, int(8 * size))  # lines are taller than letters
    if hi <= lo:
        return 2.0 * size
    window = ac[lo:hi]
    best = float(window.max())
    if best <= 0.15:
        return 2.0 * size
    # Multiples of the true pitch correlate as well or better (e.g. long and short lines
    # alternating), so take the first clear local peak rather than the highest one.
    for i in range(1, len(window) - 1):
        if window[i] >= window[i - 1] and window[i] >= window[i + 1] and window[i] >= 0.35 * best:
            return float(lo + i)
    return float(lo + int(np.argmax(window)))


def _peaks(values, min_distance, min_height):
    order = np.argsort(values)[::-1]
    chosen = []
    for i in order:
        if values[i] < min_height:
            break
        if all(abs(i - j) >= min_distance for j in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def _atoms(line):
    """Ink pieces of a line separated by blank columns; over-wide pieces are split."""
    spans = sorted((c[1], c[1] + c[4], c) for c in line.comps)
    merged = []
    for x0, x1, comp in spans:
        if merged and x0 < merged[-1][1] - 0.3 * min(x1 - x0, merged[-1][1] - merged[-1][0]):
            merged[-1][1] = max(merged[-1][1], x1)
            merged[-1][2].append(comp)
        else:
            merged.append([x0, x1, [comp]])
    # Character height: Hangul strokes come in jamo-sized pieces, so measure the stacked
    # pieces (ㄱ over ㅗ ...) rather than single connected components.
    heights = [max(c[2] + c[3] for c in comps) - min(c[2] for c in comps) for _, _, comps in merged]
    if heights:
        line.h = float(np.percentile(heights, 75))
    atoms = []
    for x0, x1, comps in merged:
        for a0, a1 in _cuts(line, x0, x1, comps):
            atoms.append((a0, a1, comps))
    return atoms


def _cuts(line, x0, x1, comps):
    """Split an ink piece wherever little ink crosses: neighbouring letters often touch or
    overlap a little, and the aligner can always glue the pieces back together."""
    h = line.h
    if x1 - x0 <= 0.55 * h:
        return [(x0, x1)]
    ids = [c[0] for c in comps]
    y0 = min(c[2] for c in comps)
    y1 = max(c[2] + c[3] for c in comps)
    column = np.isin(line.labels[y0:y1, x0:x1], ids).sum(axis=0).astype(np.float32)
    column = np.convolve(column, np.ones(3) / 3, mode="same")
    margin = int(0.22 * h)
    limit = 0.35 * float(column.max())
    candidates = [i for i in range(margin, len(column) - margin)
                  if column[i] <= limit and column[i] <= column[i - 1] and column[i] <= column[i + 1]]
    chosen = []
    for i in sorted(candidates, key=lambda i: column[i]):
        if all(abs(i - j) >= 0.25 * h for j in chosen):
            chosen.append(i)
    edges = [x0] + [x0 + i for i in sorted(chosen)] + [x1]
    pieces = list(zip(edges, edges[1:]))
    out = []
    for a, b in pieces:  # anything still very wide is cut at its thinnest column
        out.extend(_split_wide(line, a, b, comps))
    return [(a, b) for a, b, _ in out]


def _split_wide(line, x0, x1, comps):
    if x1 - x0 <= 1.45 * line.h:
        return [(x0, x1, comps)]
    ids = [c[0] for c in comps]
    y0 = min(c[2] for c in comps)
    y1 = max(c[2] + c[3] for c in comps)
    region = np.isin(line.labels[y0:y1, x0:x1], ids)
    column = region.sum(axis=0).astype(np.float32)
    lo, hi = int(0.3 * line.h), len(column) - int(0.3 * line.h)
    if hi <= lo:
        return [(x0, x1, comps)]
    cut = x0 + lo + int(np.argmin(column[lo:hi]))
    return _split_wide(line, x0, cut, comps) + _split_wide(line, cut, x1, comps)


# --------------------------------------------------------------------------- alignment

# Rough pen-path length of each jamo, in units of the syllable height: lets the aligner tell a
# simple syllable (이) from a busy one (볼) even though both are about as wide.
JAMO_INK = {
    "ㄱ": 1.0, "ㄴ": 1.0, "ㄷ": 1.5, "ㄹ": 2.2, "ㅁ": 2.0, "ㅂ": 2.2, "ㅅ": 1.2, "ㅇ": 1.6, "ㅈ": 1.5,
    "ㅊ": 1.8, "ㅋ": 1.4, "ㅌ": 2.0, "ㅍ": 2.2, "ㅎ": 2.0,
    "ㅏ": 1.3, "ㅐ": 2.3, "ㅑ": 1.6, "ㅒ": 2.6, "ㅓ": 1.3, "ㅔ": 2.3, "ㅕ": 1.6, "ㅖ": 2.6,
    "ㅗ": 1.3, "ㅛ": 1.6, "ㅜ": 1.3, "ㅠ": 1.6, "ㅡ": 1.0, "ㅣ": 1.0,
}
SYMBOL_INK = {".": 0.12, ",": 0.25, "'": 0.25, "`": 0.25, '"': 0.5, ":": 0.3, ";": 0.4, "-": 0.5,
              "_": 0.6, "=": 1.0, "+": 1.0, "(": 0.9, ")": 0.9, "[": 1.4, "]": 1.4, "{": 1.6, "}": 1.6,
              "<": 0.9, ">": 0.9, "/": 0.9, "!": 0.8, "?": 1.3, "~": 0.8, "■": 2.5, "·": 0.12}


def _jamo_ink(jamo):
    if jamo in JAMO_INK:
        return JAMO_INK[jamo]
    parts = charsets.JAMO_PARTS.get(jamo) or charsets.MIXED_VOWELS.get(jamo)
    return sum(_jamo_ink(p) for p in parts) if parts else 1.5


def expected_ink(ch):
    """Typical pen-path length of ``ch`` in units of the line's character height."""
    if charsets.is_syllable(ch):
        initial, vowel, final = charsets.decompose(ch)
        return _jamo_ink(initial) + _jamo_ink(vowel) + (0.8 * _jamo_ink(final) if final else 0.0)
    if charsets.is_jamo(ch):
        return _jamo_ink(ch)
    if ch in SYMBOL_INK:
        return SYMBOL_INK[ch]
    if ch in NARROW:
        return 0.9
    if ch.isupper():
        return 2.0
    if ch.isalnum():
        return 1.4
    return 1.5


def align(atoms, chars, spaces, newlines=None, ink_scale=None, width_scale=1.0):
    """Assign transcript characters to runs of atoms by dynamic programming.

    ``atoms``: dicts with x0, x1, line (global index), h, gap (blank before, px; None at a line
    start) and ink (pen-path length, px).  ``chars``/``spaces``/``newlines``: transcript
    characters without whitespace, and whether a space / line break preceded each one.
    ``newlines`` is given only when the transcript keeps the photo's line breaks; ``ink_scale``
    turns on the ink-amount check.  Returns ``(groups, disturbed)``: per char
    ``(first_atom, last_atom, width_cost)`` or ``None``, and the char indices next to which the
    transcript and the image disagreed.
    """
    N, M = len(chars), len(atoms)
    INF = float("inf")
    f = np.full((N + 1, M + 1), INF)
    back = {}
    f[0, 0] = 0.0
    ratio = M / max(N, 1)
    band = max(60, int(0.12 * M))
    exp = [expected_width(c) * (width_scale if charsets.is_syllable(c) else 1.0) for c in chars]
    ink_exp = [expected_ink(c) * ink_scale for c in chars] if ink_scale else None
    line_end = [0] * M  # index of the first atom of the next line
    nxt = M
    for m in range(M - 1, -1, -1):
        line_end[m] = nxt
        if atoms[m]["gap"] is None:
            nxt = m

    def relax(n, m, cost, step):
        if cost < f[n, m]:
            f[n, m], back[(n, m)] = cost, step

    for n in range(N + 1):
        centre = n * ratio
        lo, hi = max(0, int(centre - band)), min(M, int(centre + band))
        for m in range(lo, hi + 1):
            cur = f[n, m]
            if cur == INF:
                continue
            if m < M:
                a = atoms[m]
                relax(n, m + 1, cur + 1.0 + 3.0 * (a["x1"] - a["x0"]) / a["h"], (n, m, "atom"))
                # Skip the rest of this image line (labels of a drawing, a stray note ...).
                end = line_end[m]
                if end > m + 1:
                    whole = a["gap"] is None
                    relax(n, end, cur + (3.0 if whole else 1.5) + 0.15 * (end - m), (n, m, "line"))
            if n < N:
                relax(n + 1, m, cur + 3.0, (n, m, "char"))
            if n == N or m == M:
                continue
            first = atoms[m]
            h = first["h"]
            start_cost = 0.0
            g = first["gap"]
            if g is not None:
                rel = g / h
                if spaces[n]:
                    start_cost = max(0.0, 0.3 - rel) * 4.0
                elif rel > 0.45:
                    start_cost = (rel - 0.45) * 2.5
            if newlines is not None:
                if newlines[n] and g is not None:
                    start_cost += 4.0
                elif not newlines[n] and g is None and n > 0:
                    start_cost += 1.5
            internal, ink = 0.0, 0.0
            for k in range(1, MAX_ATOMS_PER_CHAR + 1):
                last = m + k - 1
                if last >= M or atoms[last]["line"] != first["line"]:
                    break
                if k > 1:
                    internal += 8.0 * (atoms[last]["gap"] / h) ** 2
                ink += atoms[last]["ink"]
                width = atoms[last]["x1"] - first["x0"]
                if width > 2.2 * h:
                    break
                wcost = 4.0 * np.log((width + 0.05 * h) / (exp[n] * h + 0.05 * h)) ** 2
                cost = cur + wcost + internal + start_cost
                if ink_exp is not None:
                    cost += min(3.0, 1.5 * np.log((ink / h + 0.1) / (ink_exp[n] + 0.1)) ** 2)
                relax(n + 1, last + 1, cost, (n, m, (m, last, wcost)))
    result, disturbed = [None] * N, set()
    n, m = N, M
    if f[N, M] == INF:
        return result, disturbed
    while (n, m) != (0, 0):
        pn, pm, step = back[(n, m)]
        if isinstance(step, tuple):
            result[pn] = step
        elif step == "atom":
            if atoms[pm]["x1"] - atoms[pm]["x0"] > 0.15 * atoms[pm]["h"]:
                disturbed.update({n - 1, n})
        elif step == "line":
            # Leftover ink at the end of a line is harmless when the transcript also breaks there.
            if not (newlines is not None and n < N and newlines[n]) and atoms[pm]["gap"] is not None:
                disturbed.update({n - 1, n})
        else:  # char pn has no ink
            disturbed.update({pn - 1, pn, pn + 1})
        n, m = pn, pm
    return result, disturbed


# --------------------------------------------------------------------------- driver

def ink_color(images):
    """Pen colour relative to the paper (0..1 RGB), like the template extractor reports."""
    samples = []
    for image in images:
        small = cv2.resize(image, None, fx=min(1.0, 1200 / image.shape[1]), fy=min(1.0, 1200 / image.shape[1]),
                           interpolation=cv2.INTER_AREA)
        alpha, _ = _ink(small)
        strong = alpha > 0.9
        if strong.sum() > 50:
            paper = np.percentile(small.reshape(-1, 3), 90, axis=0)
            samples.append(np.median(small[strong], axis=0) / np.maximum(paper, 1))
    return np.clip(np.median(samples, axis=0), 0, 1).tolist() if samples else None


def _prepare(image, work_width=WORK_WIDTH):
    scale = min(1.0, work_width / image.shape[1]) if work_width else 1.0
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    alpha, dark = _ink(image)
    angle = _skew_angle((alpha > 0.3).astype(np.uint8))
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = lambda a, v: cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=v)  # noqa: E731
    image_r = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha = _remove_rules(rot(alpha, 0), rot(dark, 0))
    # Map from analysis coordinates back to the original image (for debugging / tests).
    inverse = cv2.invertAffineTransform(M) / scale
    return image_r, alpha, inverse


def _pen_width(labels, lines):
    ids = [c[0] for line in lines for c in line.comps]
    binary = np.isin(labels, ids).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return max(1.0, 2.0 * binary.sum() / perimeter) if perimeter else 3.0


def _parse(transcript):
    text = charsets.normalize_text(transcript)
    chars, spaces, newlines = [], [], []
    prev_space, prev_newline = True, True
    for c in text:
        if c.isspace():
            prev_space = True
            prev_newline = prev_newline or c == "\n"
            continue
        chars.append(c)
        spaces.append(prev_space)
        newlines.append(prev_newline)
        prev_space = prev_newline = False
    lines = sum(1 for line in text.split("\n") if line.strip())
    return chars, spaces, newlines, lines


def extract_freeform(images, transcript, single_line=False, work_width=WORK_WIDTH, skip=()):
    """Learn glyphs from free-form handwriting ``images`` (RGB arrays) whose text is ``transcript``.

    ``single_line``: every image is one line of the transcript (copy-sheet bands).
    ``skip``: indices (into the transcript without whitespace) of characters to find but not
    learn, such as exponents.  Returns ``(glyphs, report)``: glyph dicts in the same format as
    the template extractor and a report with counts and, per transcript character, where it was
    found.
    """
    chars, spaces, newlines, transcript_lines = _parse(transcript)
    pages, atoms = [], []
    line_no = 0
    for image in images:
        image_r, alpha, inverse = _prepare(image, work_width)
        lines, labels = _segment_lines(alpha, single_line)
        pages.append((image_r, alpha, labels, inverse))
        pen = _pen_width(labels, lines)
        for line in lines:
            prev = None
            ys = [c[2] for c in line.comps]
            ye = [c[2] + c[3] for c in line.comps]
            band = labels[min(ys):max(ye)]
            for x0, x1, comps in _atoms(line):
                ink = np.isin(band[:, x0:x1], [c[0] for c in comps]).sum() / pen
                atoms.append({"x0": x0, "x1": x1, "line": line_no, "h": line.h, "comps": comps, "ink": ink,
                              "gap": None if prev is None else max(0, x0 - prev), "page": len(pages) - 1})
                prev = x1
            line_no += 1
    if not atoms or not chars:
        return [], {"chars": len(chars), "learned": 0, "lines": line_no, "positions": {}}

    # Use the transcript's line breaks only if it was typed line by line like the photo.
    keep_lines = abs(transcript_lines - line_no) <= max(2, 0.3 * line_no) and transcript_lines > 1
    breaks = newlines if keep_lines else None
    assignment, disturbed = align(atoms, chars, spaces, breaks)
    # Second pass: calibrate how much ink this writer uses per syllable, then use ink amounts
    # to tell similar-width characters apart.
    ratios, widths = [], []
    for n, g in enumerate(assignment):
        if g is not None and n not in disturbed and charsets.is_syllable(chars[n]):
            h = atoms[g[0]]["h"]
            ratios.append(sum(a["ink"] for a in atoms[g[0]:g[1] + 1]) / h / expected_ink(chars[n]))
            widths.append((atoms[g[1]]["x1"] - atoms[g[0]]["x0"]) / h / expected_width(chars[n]))
    if len(ratios) >= 8:
        width_scale = float(np.clip(np.median(widths), 0.6, 1.4))
        assignment, disturbed = align(atoms, chars, spaces, breaks, ink_scale=float(np.median(ratios)),
                                      width_scale=width_scale)
    trusted = _trusted(assignment, disturbed, spaces)

    # Cut out the glyphs.
    found = []
    for n, group in enumerate(assignment):
        if group is None or n not in trusted:
            continue
        a0, a1 = atoms[group[0]], atoms[group[1]]
        x0, x1 = a0["x0"], a1["x1"]
        image_r, alpha, labels, _ = pages[a0["page"]]
        comps = {c[0]: c for i in range(group[0], group[1] + 1) for c in atoms[i]["comps"]}
        whole, clipped = [], []
        for i, c in comps.items():
            share = (min(x1, c[1] + c[4]) - max(x0, c[1])) / max(1, c[4])
            if share >= 0.7:
                whole.append(i)  # a stroke mostly inside this character belongs to it
            elif share >= 0.3 or c[4] > 1.2 * a0["h"]:
                clipped.append(i)  # shared with a neighbour: keep only our part
        ids = whole + clipped
        if not ids:
            continue
        y0 = min(comps[i][2] for i in ids)
        y1 = max(comps[i][2] + comps[i][3] for i in ids)
        xa = min([comps[i][1] for i in whole] + [x0] * bool(clipped))
        xb = max([comps[i][1] + comps[i][4] for i in whole] + [x1] * bool(clipped))
        region = np.isin(labels[y0:y1, xa:xb], whole)
        if clipped:
            part = np.isin(labels[y0:y1, xa:xb], clipped)
            part[:, :max(0, x0 - xa)] = False
            part[:, max(0, x1 - xa):] = False
            region |= part
        rows = np.nonzero(region.any(axis=1))[0]
        if len(rows) == 0:
            continue
        y0, y1 = y0 + rows[0], y0 + rows[-1] + 1
        region = region[rows[0]:rows[-1] + 1]
        mask = np.where(region, alpha[y0:y1, xa:xb], 0).astype(np.float32)
        if mask.sum() < 5:
            continue
        if not shape_ok(chars[n], mask, a0["h"]):
            continue
        found.append({"n": n, "char": chars[n], "mask": mask, "top": y0, "bottom": y1, "x0": xa,
                      "page": a0["page"], "line": a0["line"]})
    found = _consistent(_even_heights(found))

    skip = set(skip)
    glyphs = _normalise([g for g in found if g["n"] not in skip], pages)
    report = {"chars": len(chars), "learned": len(glyphs), "lines": line_no,
              "positions": {g["n"]: g["center"] for g in found if "center" in g},
              # Ink boxes in analysis coordinates (x0, top, x1, bottom), for measuring style.
              "boxes": [{"n": g["n"], "char": g["char"], "line": g["line"], "page": g["page"],
                         "box": _ink_box(g)} for g in found],
              "spaces": spaces}
    return glyphs, report


def _ink_box(g):
    ys, xs = np.nonzero(g["mask"] > 0.3)
    if len(ys) == 0:
        return (g["x0"], g["top"], g["x0"], g["bottom"])
    return (int(g["x0"] + xs.min()), int(g["top"] + ys.min()), int(g["x0"] + xs.max() + 1),
            int(g["top"] + ys.max() + 1))


RIGHT_BAR_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅣㅘㅙㅚㅝㅞㅟㅢ")
TALL_MARKS = set("()[]{}|")
FLAT_MARKS = set("-_=~")
DOTS = set(".·")
SMALL_MARKS = set(",'`\"")


def _right_bar(mask):
    """Length (fraction of glyph height) of the longest vertical stroke near the right edge --
    where ㅏ ㅓ ㅣ ... stand; the stem of ㅜ or ㅗ is usually further in."""
    ink = mask > 0.3
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return 0.0
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = ink.shape
    best = 0
    for col in ink[:, int(0.62 * W):].T:
        run = longest = 0
        for v in col:
            run = run + 1 if v else 0
            longest = max(longest, run)
        best = max(best, longest)
    return best / H


def _one_stroke_across(mask):
    """True if nearly every row crosses the ink once, as with ( ) | (not 'f(' run together)."""
    ink = mask > 0.3
    rows = [r for r in ink if r.any()]
    if not rows:
        return False
    multi = sum(1 for r in rows if np.count_nonzero(np.diff(r.astype(np.int8)) == 1) + int(r[0]) > 1)
    return multi <= 0.12 * len(rows)


def shape_ok(ch, mask, line_h):
    """Reject a glyph whose shape cannot be that character (catches labels shifted onto a
    neighbour or a half-cut syllable).  Only checks that hold for any handwriting are used."""
    ys, xs = np.nonzero(mask > 0.3)
    if len(ys) == 0:
        return False
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if charsets.is_syllable(ch):
        if h < 0.5 * line_h:
            return False  # e.g. one jamo or a dot
        _, vowel, final = charsets.decompose(ch)
        bar = _right_bar(mask)
        if vowel in RIGHT_BAR_VOWELS:
            # 가, 위 ... end in a long vertical stroke (shorter above a final consonant: 각, 산)
            return bar >= (0.28 if final else 0.4)
        return bar < 0.75  # 고, 우 ... do not
    if ch in TALL_MARKS:
        return h >= 2.0 * w and (ch in "[]{}" or _one_stroke_across(mask))
    if ch in FLAT_MARKS:
        return w >= 1.5 * h and h < 0.5 * line_h
    if ch in DOTS:
        return max(h, w) <= 0.3 * line_h
    if ch in SMALL_MARKS:
        return max(h, w) <= 0.55 * line_h
    return True


def _even_heights(found):
    """Hangul syllables on one line are about equally tall; a much shorter one is most likely
    something else (lowercase letters, one jamo) carrying a shifted label."""
    by_line = {}
    for g in found:
        if charsets.is_syllable(g["char"]):
            by_line.setdefault(g["line"], []).append(g["bottom"] - g["top"])
    typical = {line: float(np.median(hs)) for line, hs in by_line.items() if len(hs) >= 3}
    return [g for g in found if not (charsets.is_syllable(g["char"]) and g["line"] in typical
                                     and g["bottom"] - g["top"] < 0.78 * typical[g["line"]])]


def _signature(mask, size=24):
    m, _, _ = _tight(mask)
    h, w = m.shape
    s = size / max(h, w)
    small = cv2.resize(m, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    out = cv2.GaussianBlur(out, (0, 0), 1.0).ravel()
    out -= out.mean()
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def _tight(mask):
    ys, xs = np.nonzero(mask > 0.05)
    if len(ys) == 0:
        return mask, 0, 0
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1], ys.min(), xs.min()


def _consistent(found):
    """Drop copies of a character that look unlike its other copies (likely mislabelled)."""
    by_char = {}
    for g in found:
        by_char.setdefault(g["char"], []).append(g)
    keep = []
    for ch, items in by_char.items():
        if len(items) < 3:
            keep.extend(items)
            continue
        sig = np.stack([_signature(g["mask"]) for g in items])
        sim = sig @ sig.T
        np.fill_diagonal(sim, np.nan)
        typical = np.nanmedian(sim, axis=1)
        floor = min(0.35, float(np.median(typical)) - 0.25)
        keep.extend(g for g, t in zip(items, typical) if t >= floor)
    keep.sort(key=lambda g: g["n"])
    return keep


def _trusted(assignment, disturbed, spaces, max_width_cost=WIDTH_COST_LIMIT):
    """Characters safe to learn.

    Without reading the letters, a mismatch between transcript and image can only be placed
    to within a word, so every word touching a mismatch is dropped, as is any character whose
    ink width is far from what that character should have.
    """
    word, words = -1, []
    for space in spaces:
        word += 1 if space else 0
        words.append(word)
    bad_words = {words[n] for n in disturbed if 0 <= n < len(words)}
    return {n for n, g in enumerate(assignment)
            if g is not None and words[n] not in bad_words and g[2] <= max_width_cost}


def _normalise(found, pages):
    """Express every glyph like a template glyph: millimetres in a box with a dotted baseline."""
    glyphs = []
    by_line = {}
    for g in found:
        by_line.setdefault(g["line"], []).append(g)
    by_page = {}
    for g in found:
        by_page.setdefault(g["page"], []).append(g)
    scale = {}
    for page, items in by_page.items():
        hangul = [g["bottom"] - g["top"] for g in items if charsets.is_syllable(g["char"])]
        caps = [g["bottom"] - g["top"] for g in items if g["char"] in charsets.UPPER + charsets.DIGITS]
        if hangul:
            scale[page] = HANGUL_MM / float(np.median(hangul))
        elif caps:
            scale[page] = 7.7 / float(np.median(caps))
        else:
            scale[page] = HANGUL_MM / float(np.median([g["bottom"] - g["top"] for g in items]))
    for line, items in by_line.items():
        sitting = [g["bottom"] for g in items if charsets.is_syllable(g["char"]) or g["char"] in BASELINE_SET]
        baseline = float(np.median(sitting or [g["bottom"] for g in items]))
        for g in items:
            s = scale[g["page"]]
            _, _, _, inverse = pages[g["page"]]
            cx, cy = g["x0"] + g["mask"].shape[1] / 2, (g["top"] + g["bottom"]) / 2
            g["center"] = tuple((inverse @ np.array([cx, cy, 1.0])).tolist())
            factor = s * PX
            mask = cv2.resize(g["mask"], (max(1, int(round(g["mask"].shape[1] * factor))),
                                          max(1, int(round(g["mask"].shape[0] * factor)))),
                              interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
            mask = cv2.copyMakeBorder(mask, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            glyphs.append({
                "char": g["char"], "n": g["n"],
                "mask": (np.clip(mask, 0, 1) * 255).astype(np.uint8),
                "left": 0.0,
                "top": float(BASELINE + (g["top"] - baseline) * s - 2 / PX),
                "w": mask.shape[1] / PX, "h": mask.shape[0] / PX,
            })
    return glyphs
