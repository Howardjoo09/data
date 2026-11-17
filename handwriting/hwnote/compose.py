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
