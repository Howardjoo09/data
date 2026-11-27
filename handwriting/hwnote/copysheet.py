"""Learn handwriting from copy sheets (필사 시트, template layout 2).

Each row of a copy sheet prints one line of text and leaves two squares of note-book grid
below it, where the writer copies the line in their ordinary note-taking hand.  Because the
corner markers fix where every band is, each band is cut out on its own at high resolution
(small note-size letters stay sharp), the grid is erased, and the band is read as exactly one
line of the transcript by the free-form aligner.  Letters written in sentences look like the
writer's notes, which is what the renderer should reproduce.
"""
import json
import os

import cv2
import numpy as np

from . import charsets, freeform
from . import template as T
from .extract import PX, darkness

BAND_PX = 24          # px per mm of the band crops
ABOVE, BELOW = 0.0, 0.4   # mm of the strips around a band included in its crop
GRID_HALF = 0.35      # mm on each side of a grid line that may be erased
MIN_BAND_INK = 0.4    # mm^2 of ink below which a band counts as left empty


def _band_matrix(H, x0, y0):
    """Image -> band crop (BAND_PX px/mm, origin at (x0, y0) mm)."""
    k = BAND_PX / PX
    S = np.array([[k, 0, -x0 * BAND_PX], [0, k, -y0 * BAND_PX], [0, 0, 1]], np.float64)
    return S @ H


def _erase_grid(band, x0, y0):
    """Paint the printed grid lines over with paper colour where no pen stroke covers them."""
    dark = darkness(band)
    h, w = dark.shape
    paper = np.percentile(band.reshape(-1, 3), 90, axis=0)
    lines = np.zeros((h, w), bool)
    half = max(1, int(round(GRID_HALF * BAND_PX)))
    x = T.GRID_STEP * np.ceil(x0 / T.GRID_STEP)
    while (x - x0) * BAND_PX < w:
        c = int(round((x - x0) * BAND_PX))
        lines[:, max(0, c - half):c + half + 1] = True
        x += T.GRID_STEP
    y = T.GRID_STEP * np.ceil(y0 / T.GRID_STEP - 1e-6)
    while (y - y0) * BAND_PX < h:
        r = int(round((y - y0) * BAND_PX))
        lines[max(0, r - half):r + half + 1, :] = True
        y += T.GRID_STEP
    out = band.copy()
    out[lines & (dark < 0.3)] = paper.astype(np.uint8)
    return out


def bands(image, H, n_lines):
    """High-resolution crops of the first ``n_lines`` writing bands of a page photo."""
    out = []
    for row in range(min(n_lines, T.COPY_ROWS)):
        _, band, bottom = T.copy_row(row)
        x0, y0, y1 = T.COPY_X0 + 0.3, band - ABOVE, bottom + BELOW
        size = (int((T.COPY_X1 - 0.3 - x0) * BAND_PX), int((y1 - y0) * BAND_PX))
        crop = cv2.warpPerspective(image, _band_matrix(H, x0, y0), size, flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
        out.append(_erase_grid(crop, x0, y0))
    return out


def source_resolution(H):
    """Pixels per millimetre of the photo or scan itself (at the page centre): the bands are
    cut out at BAND_PX, but cannot hold more detail than this."""
    J = np.linalg.inv(H)
    cx, cy = T.PAGE_W / 2 * PX, T.PAGE_H / 2 * PX
    def at(x, y):
        v = J @ np.array([x, y, 1.0])
        return v[:2] / v[2]
    dx = np.linalg.norm(at(cx + PX, cy) - at(cx, cy))
    dy = np.linalg.norm(at(cx, cy + PX) - at(cx, cy))
    return float(np.sqrt(dx * dy))


def _has_writing(band):
    dark = darkness(band)
    return float((dark > 0.35).sum()) / BAND_PX ** 2 >= MIN_BAND_INK


def transcript(lines):
    """Plain transcript of copy-sheet lines and the indices (non-space characters) of the
    exponents and indices, which are found but not learned as ordinary letters."""
    text, skip, n = [], [], 0
    for line in lines:
        for ch, kind in T.parse_markup(line):
            if ch.isspace():
                continue
            if kind != "normal":
                skip.append(n)
            n += 1
        text.append(T.plain_text(line))
    return "\n".join(text), skip


def read_pages(pages):
    """``pages``: ``[(band_images, lines[, source px/mm])]`` (see :func:`bands`).  Returns ``(glyphs, report,
    band_images, lines)`` for the bands that were written in."""
    images, used, resolution = [], [], []
    for crops, lines, *rest in pages:
        for band, line in zip(crops, lines):
            if _has_writing(band):
                images.append(band)
                used.append(line)
                if rest and rest[0]:
                    resolution.append(rest[0])
    if not images:
        return [], {"chars": 0, "learned": 0, "lines": 0}, [], []
    text, skip = transcript(used)
    glyphs, report = freeform.extract_freeform(images, text, single_line=True, work_width=None, skip=skip)
    report["skip"] = skip
    report["style"] = measure(images, report)
    from .score import slant
    report["style"]["slant"] = slant([g["mask"].astype(np.float32) / 255 for g in glyphs])
    # How much detail the photo held: px a typical syllable had in it (the program's writing is
    # blurred to this when the two are compared; a blurred stroke measures wider).
    heights = [b["box"][3] - b["box"][1] for b in report["boxes"] if charsets.is_syllable(b["char"])]
    if resolution and heights:
        report["style"]["detail_px"] = float(np.median(heights)) / BAND_PX * float(np.median(resolution))
    return glyphs, report, images, used


def measure(images, report):
    """How the writer spaces and sizes their note-taking (see style.from_boxes)."""
    from . import style
    st = style.from_boxes(report["boxes"], report["spaces"])
    st["stroke"] = style.stroke_ratio(images, report["boxes"])
    return st


def save_evaluation(directory, pages):
    """Keep the bands of an evaluation page with their text, for measuring the result."""
    folder = os.path.join(directory, "evaluation")
    os.makedirs(folder, exist_ok=True)
    kept = []
    for crops, lines, *rest in pages:
        for band, line in zip(crops, lines):
            if _has_writing(band):
                name = f"band{len(kept):02d}.png"
                cv2.imwrite(os.path.join(folder, name), cv2.cvtColor(band, cv2.COLOR_RGB2BGR))
                kept.append({"file": name, "text": line, "source_px_per_mm": rest[0] if rest else None})
    with open(os.path.join(folder, "lines.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    return len(kept)


def load_evaluation(directory):
    folder = os.path.join(directory, "evaluation")
    path = os.path.join(folder, "lines.json")
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as f:
        kept = json.load(f)
    images = [cv2.cvtColor(cv2.imread(os.path.join(folder, k["file"])), cv2.COLOR_BGR2RGB) for k in kept]
    return images, [k["text"] for k in kept]


def evaluation_resolution(directory):
    """Pixels per mm of the photo the evaluation page came from, or None if not recorded."""
    path = os.path.join(directory, "evaluation", "lines.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        values = [k.get("source_px_per_mm") for k in json.load(f)]
    values = [v for v in values if v]
    return float(np.median(values)) if values else None
