"""Assemble Hangul syllables that were never handwritten from handwritten jamo.

Masks are float32 arrays (0..1 ink opacity) at the profile resolution.  Boxes are given in
unit coordinates of the syllable square: (x0, y0, x1, y1).
"""
import cv2
import numpy as np

from .charsets import BOTTOM_VOWELS, JAMO_PARTS, MIXED_VOWELS, RIGHT_VOWELS, decompose

WIDE_RIGHT = set("ㅐㅒㅔㅖ")

# (initial, vowel, final) boxes for vowels written to the right of the initial.
RIGHT = {False: ((0.00, 0.08, 0.58, 0.92), (0.52, 0.00, 1.00, 1.00), None),
         True: ((0.04, 0.02, 0.54, 0.56), (0.50, 0.00, 0.96, 0.62), (0.14, 0.62, 0.86, 1.00))}
RIGHT_WIDE = {False: ((0.02, 0.12, 0.47, 0.88), (0.42, 0.00, 1.00, 1.00), None),
              True: ((0.03, 0.04, 0.46, 0.56), (0.42, 0.00, 0.98, 0.62), (0.14, 0.62, 0.86, 1.00))}
# Vowels written under the initial.
BOTTOM = {False: ((0.18, 0.00, 0.82, 0.52), (0.00, 0.48, 1.00, 0.98), None),
          True: ((0.20, 0.00, 0.80, 0.36), (0.02, 0.33, 0.98, 0.64), (0.16, 0.64, 0.84, 1.00))}
# Mixed vowels: horizontal part under the initial, vertical part on the right.
MIXED = {False: ((0.03, 0.00, 0.60, 0.48), (0.00, 0.44, 0.74, 0.98), (0.62, 0.00, 1.00, 1.00), None),
         True: ((0.04, 0.00, 0.58, 0.34), (0.00, 0.30, 0.72, 0.62), (0.62, 0.00, 1.00, 0.64),
                (0.16, 0.64, 0.84, 1.00))}


def stroke_width(mask):
    """Rough pen width in px: 2·area / perimeter of the binarised strokes."""
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * binary.sum() / perimeter if perimeter else 0.0


def match_stroke(mask, target):
    """Thicken or thin ``mask`` so its strokes are about ``target`` px wide."""
    current = stroke_width(mask)
    if not current or not target:
        return mask
    # Close only part of the gap: full dilation clogs dense jamo such as ㅃ or ㄼ.
    radius = int(round((target - current) / 3))
    radius = max(-1, min(radius, 2))
    if radius == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(radius) + 1,) * 2)
    padded = cv2.copyMakeBorder(mask, abs(radius), abs(radius), abs(radius), abs(radius), cv2.BORDER_CONSTANT, value=0)
    return cv2.dilate(padded, kernel) if radius > 0 else cv2.erode(padded, kernel)


def tight(mask, threshold=0.05):
    ys, xs = np.nonzero(mask > threshold)
    if len(ys) == 0:
        return mask[:1, :1], 0, 0
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1], ys.min(), xs.min()


def _paste_fit(canvas, mask, box, stretch=1.25):
    """Scale ``mask`` into ``box`` (unit coords of ``canvas``) and composite it (max)."""
    mask, _, _ = tight(mask)
    H, W = canvas.shape
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    bw, bh = x1 - x0, y1 - y0
    h, w = mask.shape
    sx, sy = bw / w, bh / h
    s = min(sx, sy)
    # Stretch only along a glyph's long axis: widening a thin stroke like ㅣ just makes it bold.
    sx = min(sx, s * stretch) if w >= 0.6 * h else s
    sy = min(sy, s * stretch) if h >= 0.6 * w else s
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    scaled = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    left = int(round(x0 + (bw - nw) / 2))
    top = int(round(y0 + (bh - nh) / 2))
    left, top = max(0, min(left, W - nw)), max(0, min(top, H - nh))
    region = canvas[top:top + nh, left:left + nw]
    np.maximum(region, scaled[:region.shape[0], :region.shape[1]], out=region)


def side_by_side(masks, gap_ratio):
    """Join jamo horizontally at a common height (used for ㄲ, ㄺ, ㅐ ...)."""
    masks = [tight(m)[0] for m in masks]
    height = max(m.shape[0] for m in masks)
    scaled = [cv2.resize(m, (max(1, int(round(m.shape[1] * height / m.shape[0]))), height),
                         interpolation=cv2.INTER_LINEAR) for m in masks]
    gap = int(round(height * gap_ratio))
    width = sum(m.shape[1] for m in scaled) + gap * (len(scaled) - 1)
    out = np.zeros((height, width), np.float32)
    x = 0
    for m in scaled:
        np.maximum(out[:, x:x + m.shape[1]], m, out=out[:, x:x + m.shape[1]])
        x += m.shape[1] + gap
    return out


class JamoSource:
    """Looks up handwritten jamo, building compound ones from simple ones when needed."""

    def __init__(self, pick):
        self.pick = pick  # pick(char) -> mask or None

    def get(self, jamo):
        mask = self.pick(jamo)
        if mask is not None:
            return mask
        parts = JAMO_PARTS.get(jamo) or MIXED_VOWELS.get(jamo)
        if not parts:
            return None
        masks = [self.get(p) for p in parts]
        if any(m is None for m in masks):
            return None
        vowel = jamo in "ㅐㅒㅔㅖ"
        return side_by_side(masks, -0.05 if vowel else 0.08)


def compose_syllable(ch, source, width_px, height_px):
    """Return a ``height_px`` x ``width_px`` mask for syllable ``ch`` or ``None`` if jamo are missing."""
    initial, vowel, final = decompose(ch)
    has_final = bool(final)
    canvas = np.zeros((height_px, width_px), np.float32)
    first = source.get(initial)
    last = source.get(final) if has_final else None
    if first is None or (has_final and last is None):
        return None
    if vowel in MIXED_VOWELS:
        horizontal, vertical = (source.get(p) for p in MIXED_VOWELS[vowel])
        if horizontal is None or vertical is None:
            return None
        b_init, b_h, b_v, b_final = MIXED[has_final]
        _paste_fit(canvas, first, b_init)
        _paste_fit(canvas, horizontal, b_h, stretch=1.6)
        _paste_fit(canvas, vertical, b_v, stretch=1.6)
    else:
        middle = source.get(vowel)
        if middle is None:
            return None
        if vowel in BOTTOM_VOWELS:
            b_init, b_vowel, b_final = BOTTOM[has_final]
        elif vowel in WIDE_RIGHT:
            b_init, b_vowel, b_final = RIGHT_WIDE[has_final]
        else:
            b_init, b_vowel, b_final = RIGHT[has_final]
        _paste_fit(canvas, first, b_init)
        _paste_fit(canvas, middle, b_vowel, stretch=1.6)
    if has_final:
        _paste_fit(canvas, last, b_final)
    return canvas


assert RIGHT_VOWELS | BOTTOM_VOWELS | set(MIXED_VOWELS) == set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


# --------------------------------------------------------------------------- learning jamo from syllables

def vowel_class(vowel):
    if vowel in MIXED_VOWELS:
        return "mixed"
    if vowel in BOTTOM_VOWELS:
        return "bottom"
    return "right_wide" if vowel in WIDE_RIGHT else "right"


def role_boxes(vclass, has_final):
    """Where each part of a syllable of this shape normally sits: {role: [boxes]}."""
    if vclass == "mixed":
        b_init, b_h, b_v, b_final = MIXED[has_final]
        boxes = {"I": [b_init], "V": [b_h, b_v]}
    else:
        table = {"bottom": BOTTOM, "right_wide": RIGHT_WIDE, "right": RIGHT}[vclass]
        b_init, b_vowel, b_final = table[has_final]
        boxes = {"I": [b_init], "V": [b_vowel]}
    if has_final:
        boxes["F"] = [b_final]
    return boxes


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _iou(a, b):
    inter = _overlap(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _plausible(role, vclass, unit, boxes):
    x0, y0, x1, y1 = unit
    region = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    if _iou(unit, region) < 0.25:
        return False
    if role == "V" and vclass == "bottom" and (x1 - x0) < 0.45:
        return False  # ㅗㅜㅡ... are wide
    if role == "V" and vclass in ("right", "right_wide") and (y1 - y0) < 0.55:
        return False  # ㅏㅓㅣ... are tall
    return True


def harvest(ch, mask):
    """Split a handwritten syllable into its jamo by stroke position.

    Returns ``{role: (jamo, mask, unit_box)}`` with roles I/V/F, or ``None`` when the parts
    cannot be attributed with confidence: a stroke running from one jamo into the next, a
    part far from where that jamo belongs, or parts in the wrong order.  A wrong jamo would
    spoil every syllable built from it, so doubtful syllables are simply not used.
    """
    initial, vowel, final = decompose(ch)
    vclass = vowel_class(vowel)
    mask, _, _ = tight(mask)
    H, W = mask.shape
    if H < 8 or W < 8:
        return None
    binary = cv2.morphologyEx((mask > 0.3).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = role_boxes(vclass, bool(final))
    owner = np.full((H, W), "", dtype=object)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        unit = (x / W, y / H, (x + w) / W, (y + h) / H)
        size = max((unit[2] - unit[0]) * (unit[3] - unit[1]), 1e-6)
        scores = {role: sum(_overlap(unit, b) for b in bs) / size for role, bs in boxes.items()}
        role = max(scores, key=scores.get)
        if size > 0.12 and scores[role] < 0.55:
            return None  # one stroke runs through two jamo: cannot be cut reliably
        owner[labels == i] = role
    parts = {}
    for role, jamo in (("I", initial), ("V", vowel), ("F", final)):
        if role not in boxes:
            continue
        piece = np.where(owner == role, mask, 0).astype(np.float32)
        if piece.sum() < 0.03 * mask.sum():
            return None
        piece, y0, x0 = tight(piece)
        unit = (x0 / W, y0 / H, (x0 + piece.shape[1]) / W, (y0 + piece.shape[0]) / H)
        if not _plausible(role, vclass, unit, boxes[role]):
            return None
        parts[role] = (jamo, piece, unit)
    return parts if _arranged(parts, vclass) else None


def _arranged(parts, vclass):
    """The parts must be laid out like a syllable of this shape (initial left of / above the
    vowel, final underneath), which catches strokes credited to the wrong jamo."""
    i, v = parts["I"][2], parts["V"][2]
    f = parts["F"][2] if "F" in parts else None
    if vclass in ("right", "right_wide"):
        ok = i[2] <= v[0] + 0.25 and i[0] <= 0.3 and v[2] >= 0.75
        if f:
            ok = ok and f[1] >= i[3] - 0.12 and f[1] >= v[3] - 0.3 and f[3] >= 0.9
    elif vclass == "bottom":
        ok = i[3] <= v[1] + 0.2 and i[1] <= 0.1 and (v[3] - v[1]) <= 0.6
        if f:
            ok = ok and f[1] >= v[3] - 0.25 and f[1] >= i[3] and f[3] >= 0.9
    else:
        ok = i[2] <= 0.8 and i[3] <= 0.7 and v[2] >= 0.8
        if f:
            ok = ok and f[1] >= i[3] and f[3] >= 0.9
    return ok


class Composer:
    """Builds unwritten syllables, preferring jamo cut from the user's own syllables.

    Lookup order per part: a jamo cut from a syllable of the same shape (placed exactly where
    it was written), the standalone jamo from a template, a jamo cut from a differently
    shaped syllable, and finally compound jamo assembled from simple ones.
    """

    def __init__(self, standalone, rng):
        self.standalone = standalone  # jamo -> mask or None (template boxes)
        self.rng = rng
        self.bank = {}

    def learn(self, ch, mask):
        parts = harvest(ch, mask)
        if not parts:
            return False
        initial, vowel, final = decompose(ch)
        vclass, has_final = vowel_class(vowel), bool(final)
        for role, (jamo, piece, unit) in parts.items():
            key = (role, jamo, vclass if role != "V" else None, has_final if role != "F" else None)
            self.bank.setdefault(key, []).append((piece, unit))
            self.bank.setdefault((role, jamo, "any", None), []).append((piece, None))
        return True

    def _pick(self, key):
        options = self.bank.get(key)
        if not options:
            return None
        return options[int(self.rng.integers(len(options)))]

    def part(self, role, jamo, vclass, has_final):
        """(mask, unit_box or None) for one part of a syllable, or None."""
        exact = self._pick((role, jamo, vclass if role != "V" else None, has_final if role != "F" else None))
        if exact is not None:
            return exact
        mask = self.standalone(jamo)
        if mask is not None:
            return mask, None
        loose = self._pick((role, jamo, "any", None))
        if loose is not None:
            return loose
        pieces = JAMO_PARTS.get(jamo)
        if pieces:
            found = [self.part(role, p, vclass, has_final) for p in pieces]
            if all(found):
                vowel = jamo in "ㅐㅒㅔㅖ"
                return side_by_side([f[0] for f in found], -0.05 if vowel else 0.08), None
        return None

    def compose(self, ch, width_px, height_px):
        initial, vowel, final = decompose(ch)
        vclass, has_final = vowel_class(vowel), bool(final)
        canvas = np.zeros((height_px, width_px), np.float32)
        wanted = [("I", initial), ("V", vowel)] + ([("F", final)] if has_final else [])
        found = {role: self.part(role, jamo, vclass, has_final) for role, jamo in wanted}
        if found["V"] is None and vclass == "mixed":
            h_part, v_part = MIXED_VOWELS[vowel]
            found_h = self.part("V", h_part, "bottom", has_final)
            found_v = self.part("V", v_part, "right", has_final)
            if found_h and found_v:
                found["V"] = ("split", found_h[0], found_v[0])
        if any(v is None for v in found.values()):
            return None
        boxes = role_boxes(vclass, has_final)
        for role, item in found.items():
            if isinstance(item[0], str):  # mixed vowel built from two parts
                _paste_fit(canvas, item[1], boxes["V"][0], stretch=1.6)
                _paste_fit(canvas, item[2], boxes["V"][1], stretch=1.6)
                continue
            mask, unit = item
            if unit is not None:
                _paste_exact(canvas, mask, unit)
            elif role == "V" and vclass == "mixed":
                union = (min(b[0] for b in boxes["V"]), 0.0, 1.0, max(b[3] for b in boxes["V"]))
                _paste_fit(canvas, mask, union, stretch=1.3)
            else:
                _paste_fit(canvas, mask, boxes[role][0], stretch=1.6 if role == "V" else 1.25)
        return canvas


def _paste_exact(canvas, mask, unit):
    H, W = canvas.shape
    x0, y0 = int(round(unit[0] * W)), int(round(unit[1] * H))
    w = min(W, max(1, int(round((unit[2] - unit[0]) * W))))
    h = min(H, max(1, int(round((unit[3] - unit[1]) * H))))
    scaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA if w < mask.shape[1] else cv2.INTER_LINEAR)
    x0, y0 = min(max(0, x0), W - w), min(max(0, y0), H - h)
    region = canvas[y0:y0 + h, x0:x0 + w]
    np.maximum(region, scaled[:region.shape[0], :region.shape[1]], out=region)
