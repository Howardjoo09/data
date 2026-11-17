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
MAX_ATOMS_PER_CHAR = 6
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
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 6)))
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


def _segment_lines(alpha):
    binary = (alpha > 0.35).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    H, W = binary.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6 or h > 0.25 * H or w > 0.5 * W:
            continue
        comps.append((i, x, y, h, w, area, cents[i][1]))
    if not comps:
        return [], labels
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
        atoms.extend(_split_wide(line, x0, x1, comps))
    return atoms


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

def align(atoms, chars, spaces):
    """Assign transcript characters to runs of atoms.

    ``atoms``: list of dicts with x0, x1, line (global index), h, gap (blank before, px or None).
    ``chars``/``spaces``: transcript characters without whitespace and whether a space preceded
    each one.  Returns ``(groups, disturbed)``: per char ``(first_atom, last_atom, width_cost)``
    or ``None``, and the char indices next to which the transcript and the image disagreed
    (an unexplained ink piece or a character that is not in the image).
    """
    N, M = len(chars), len(atoms)
    INF = float("inf")
    f = np.full((N + 1, M + 1), INF)
    back = {}
    f[0, 0] = 0.0
    ratio = M / max(N, 1)
    band = max(40, int(0.07 * M))
    exp = [expected_width(c) for c in chars]

    def skip_atom_cost(a):
        return 1.0 + 3.0 * (a["x1"] - a["x0"]) / a["h"]

    for n in range(N + 1):
        centre = n * ratio
        lo, hi = max(0, int(centre - band)), min(M, int(centre + band))
        for m in range(lo, hi + 1):
            cur = f[n, m]
            if cur == INF:
                continue
            if m < M:  # treat atom m as noise
                cost = cur + skip_atom_cost(atoms[m])
                if cost < f[n, m + 1]:
                    f[n, m + 1], back[(n, m + 1)] = cost, (n, m, None)
            if n < N:  # character missing from the image
                cost = cur + 3.0
                if cost < f[n + 1, m]:
                    f[n + 1, m], back[(n + 1, m)] = cost, (n, m, None)
            if n == N or m == M:
                continue
            first = atoms[m]
            h = first["h"]
            internal = 0.0
            gap_cost = 0.0
            g = first["gap"]
            if g is not None:
                rel = g / h
                if spaces[n]:
                    gap_cost = max(0.0, 0.3 - rel) * 4.0
                elif rel > 0.45:
                    gap_cost = (rel - 0.45) * 2.5
            for k in range(1, MAX_ATOMS_PER_CHAR + 1):
                last = m + k - 1
                if last >= M or atoms[last]["line"] != first["line"]:
                    break
                if k > 1:
                    internal += 8.0 * (atoms[last]["gap"] / h) ** 2
                width = atoms[last]["x1"] - first["x0"]
                if width > 2.2 * h:
                    break
                wcost = 4.0 * np.log((width + 0.05 * h) / (exp[n] * h + 0.05 * h)) ** 2
                cost = cur + wcost + internal + gap_cost
                if cost < f[n + 1, last + 1]:
                    f[n + 1, last + 1], back[(n + 1, last + 1)] = cost, (n, m, (m, last, wcost))
    result, disturbed = [None] * N, set()
    n, m = N, M
    if f[N, M] == INF:
        return result, disturbed
    while (n, m) != (0, 0):
        pn, pm, group = back[(n, m)]
        if group is not None:
            result[pn] = group
        elif pn == n:  # skipped an ink piece between chars n-1 and n
            if atoms[pm]["x1"] - atoms[pm]["x0"] > 0.15 * atoms[pm]["h"]:
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


def _prepare(image):
    scale = min(1.0, WORK_WIDTH / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    alpha, dark = _ink(image)
    angle = _skew_angle((alpha > 0.3).astype(np.uint8))
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = lambda a, v: cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=v)
    image_r = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    alpha = _remove_rules(rot(alpha, 0), rot(dark, 0))
    # Map from analysis coordinates back to the original image (for debugging / tests).
    inverse = cv2.invertAffineTransform(M) / scale
    return image_r, alpha, inverse


def extract_freeform(images, transcript):
    """Learn glyphs from free-form handwriting ``images`` (RGB arrays) whose text is ``transcript``.

    Returns ``(glyphs, report)``: glyph dicts in the same format as the template extractor
    and a report with counts and, per transcript character, where it was found.
    """
    text = charsets.normalize_text(transcript)
    chars, spaces, prev_space = [], [], True
    for c in text:
        if c.isspace():
            prev_space = True
            continue
        chars.append(c)
        spaces.append(prev_space)
        prev_space = False

    pages, atoms = [], []
    line_no = 0
    for image in images:
        image_r, alpha, inverse = _prepare(image)
        lines, labels = _segment_lines(alpha)
        pages.append((image_r, alpha, labels, inverse))
        for line in lines:
            prev = None
            for x0, x1, comps in _atoms(line):
                atoms.append({"x0": x0, "x1": x1, "line": line_no, "h": line.h, "comps": comps,
                              "gap": None if prev is None else max(0, x0 - prev), "page": len(pages) - 1,
                              "line_obj": line})
                prev = x1
            line_no += 1
    assignment, disturbed = align(atoms, chars, spaces) if atoms and chars else ([None] * len(chars), set())
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
        ids = [i for i, c in comps.items() if x0 <= c[1] + c[4] / 2 <= x1 or c[4] > 1.2 * a0["h"]]
        if not ids:
            continue
        y0 = min(comps[i][2] for i in ids)
        y1 = max(comps[i][2] + comps[i][3] for i in ids)
        xa = min(comps[i][1] for i in ids)
        xb = max(comps[i][1] + comps[i][4] for i in ids)
        xa, xb = max(xa, x0 - 2) if len(ids) > 1 else xa, xb
        region = np.isin(labels[y0:y1, xa:xb], ids)
        # Pieces of touching letters that were split apart are clipped to this character.
        clip = np.zeros_like(region)
        clip[:, max(0, x0 - xa):max(0, x1 - xa)] = True
        wide = [i for i in ids if comps[i][4] > 1.2 * a0["h"]]
        if wide:
            region &= clip | ~np.isin(labels[y0:y1, xa:xb], wide)
        mask = np.where(region, alpha[y0:y1, xa:xb], 0).astype(np.float32)
        if mask.sum() < 5:
            continue
        found.append({"n": n, "char": chars[n], "mask": mask, "top": y0, "bottom": y1, "x0": xa,
                      "page": a0["page"], "line": a0["line"]})

    glyphs = _normalise(found, pages)
    report = {"chars": len(chars), "learned": len(glyphs), "lines": line_no,
              "positions": {g["n"]: g["center"] for g in found if "center" in g}}
    return glyphs, report


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
                "char": g["char"],
                "mask": (np.clip(mask, 0, 1) * 255).astype(np.uint8),
                "left": 0.0,
                "top": float(BASELINE + (g["top"] - baseline) * s - 2 / PX),
                "w": mask.shape[1] / PX, "h": mask.shape[0] / PX,
            })
    return glyphs
