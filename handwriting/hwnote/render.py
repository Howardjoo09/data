"""Write text onto notebook pages with the glyphs of a handwriting profile."""
import functools
import io
import json
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from . import charsets
from .compose import Composer, JamoSource, match_stroke, rebuild, stroke_width, tight, vowel_class

PX = 10  # profile resolution, px per mm
BASELINE_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZacemnorsuvwxz0123456789")
INK_COLORS = {"black": (0.13, 0.13, 0.16), "blue": (0.10, 0.18, 0.55), "red": (0.72, 0.12, 0.12),
              "pencil": (0.33, 0.33, 0.35)}


@dataclass
class Glyph:
    mask: np.ndarray  # float32 0..1
    top: float        # mm from the top of the template box
    w: float          # mm
    h: float          # mm
    variant: bool = False  # a one-off copy (not cached)


@dataclass
class Options:
    paper: str = "lined"        # lined | grid | blank
    ink: str = "auto"           # auto | black | blue | red | pencil
    line_pitch: float = 8.0     # mm between ruled lines
    size: float = 0.6           # Hangul height as a fraction of the line pitch
    letter_spacing: float = 1.0
    word_spacing: float = 1.0
    messiness: float = 0.35     # 0 = perfectly regular, 1 = loose
    margin_line: bool = True
    dpi: int = 300
    seed: int | None = None
    # Pen width as a fraction of the Hangul height (0.12 in the writer's own notes); 0 keeps
    # the strokes exactly as written in the template.
    stroke: float = 0.075
    # How far each syllable's size is pulled towards the typical size of its shape
    # (0 = as written in its template box, 1 = all the same).
    evenness: float = 1.0

# Grid paper as in the writer's notes: beige sheet, 4.24 mm squares, soft grey-beige lines.
GRID_PAPER = (251, 249, 237)
GRID_LINE = (224, 223, 213)
GRID_STEP = 4.24

MID_LINE = set("-=+~×÷±∓→←⇒⇐⇔·∙:<>≤≥≠≈∞()[]{}|")

# Spacing measured in the writer's notes, in units of the Hangul height.
LETTER_GAP = {"hangul": -0.06, "latin": -0.01}
# Measured on the writer's handwriting of a known passage (see hwnote/style.py), relative to
# the syllable height: syllables are narrower than in template boxes (w/h 0.65), the gap
# between syllables is 0.025 +- 0.08, words are 0.47 apart, and each syllable's centre wanders
# 0.07 up or down -- the writing follows no ruler at the top or the bottom.
VERTICAL_WANDER = 0.0    # tops and bottoms of a line's syllables each in one row
GAP_WANDER = 0.12        # x messiness = 0.04
HANGUL_ASPECT = 0.875
# Each syllable's width stays within this share of the typical width for its shape (a narrow
# 러 next to a wide 한 looked shrunken).
ASPECT_TOLERANCE = 0.15
# Neighbouring letters are spaced by the nearest distance between their ink (not their boxes),
# so the gap looks the same whatever the letters' shapes: this share of the syllable height.
INK_GAP = 0.045
# How alike (0..1) a rebuilt copy of a syllable must be to the one written, to be used.
REBUILD_MIN_SIMILARITY = 0.72
WORD_GAP = 0.42


def _median(values, default):
    return float(np.median(values)) if len(values) else default


@functools.lru_cache(maxsize=1)
def _fallback_font_bytes():
    return bytes(pymupdf.Font("korea").buffer)


@functools.lru_cache(maxsize=64)
def _fallback_font(size):
    return ImageFont.truetype(io.BytesIO(_fallback_font_bytes()), size)


class Profile:
    def __init__(self, directory):
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.id = data["id"]
        self.ink = data.get("ink")
        # Learned from a sample of the writer's sentences (hwnote/style.py).
        self.char_scale = data.get("char_scale", {})
        self.char_width = data.get("char_width", {})
        self.glyphs = {}
        excluded = set(data.get("excluded", []))  # copies the writer chose not to use
        for g in data["glyphs"]:
            if g["file"] in excluded:
                continue
            mask = cv2.imread(os.path.join(directory, g["file"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            self.glyphs.setdefault(g["char"], []).append(
                Glyph(mask.astype(np.float32) / 255.0, g["top"], g["w"], g["h"]))
        baseline = data.get("baseline", 11.5)
        # Letters rebuilt from other letters of the writer, e.g. a round y (see LETTER_STYLES).
        for ch, how in data.get("letter_styles", {}).items():
            built = LETTER_STYLES[how](self.glyphs, baseline) if how in LETTER_STYLES else []
            if built:
                self.glyphs[ch] = built
        self._stats(baseline)

    def _stats(self, baseline):
        syll = [g for c, gs in self.glyphs.items() if charsets.is_syllable(c) for g in gs]
        jamo = [g for c, gs in self.glyphs.items() if c in charsets.CHOSEONG for g in gs]
        latin = [g for c, gs in self.glyphs.items() if c in BASELINE_LETTERS for g in gs]
        caps = [g for c, gs in self.glyphs.items() if c in charsets.UPPER + charsets.DIGITS for g in gs]
        if syll:
            self.hangul_h = _median([g.h for g in syll], 9.0)
            self.hangul_w = _median([g.w for g in syll], self.hangul_h)
            self.hangul_bottom = _median([g.top + g.h for g in syll], 12.0)
        elif jamo:
            self.hangul_h = _median([g.h for g in jamo], 8.0) * 1.15
            self.hangul_w = self.hangul_h * 0.95
            self.hangul_bottom = _median([g.top + g.h for g in jamo], 12.0)
        else:
            self.hangul_h = _median([g.h for g in caps], 8.0)
            self.hangul_w = self.hangul_h
            self.hangul_bottom = baseline
        self.cap_h = _median([g.h for g in caps], self.hangul_h * 0.85)
        self.latin_bottom = _median([g.top + g.h for g in latin], baseline)
        self.stroke = _median([stroke_width(g.mask) for gs in self.glyphs.values() for g in gs[:1]], 4.0)

    @property
    def chars(self):
        return set(self.glyphs)


def compose_size(mask):
    """Size of the tight ink box of a glyph image (parts' unit boxes refer to it)."""
    return tight(mask)[0].shape


def _with_margin(original, inner):
    """Put a rebuilt tight-box image back into the original glyph's frame (same margins)."""
    _, r0, c0 = tight(original)
    out = np.zeros_like(original)
    h, w = inner.shape
    out[r0:r0 + h, c0:c0 + w] = inner[:out.shape[0] - r0, :out.shape[1] - c0]
    return out


# Lowercase letters are written a little larger than in the template boxes, so that they stand
# up to the capitals and to Δ beside them (Δu, Δx); Δ itself is written about their size.
CLASS_SCALE = {"short": 1.15, "desc": 1.15, "tall": 1.0, "cap": 1.0}
DELTA_OVER_SHORT = 1.05


def _descender(glyph, baseline):
    """The part of a glyph below the template baseline (the tail of a g) and where its stem
    enters it.  Returns (tail mask, stem x) or None."""
    row = int(round((baseline - glyph.top) * PX))
    m = glyph.mask
    if row <= 2 or row >= m.shape[0] - 4:
        return None
    tail = m[row - 1:].copy()
    ink = np.nonzero(tail[:4].max(axis=0) > 0.3)[0]
    if len(ink) == 0:
        return None
    return tail, float(ink.mean())


def round_y(glyphs, baseline):
    """A y written as a u whose right stroke runs on below the line into the curled tail of
    the writer's g -- the usual handwritten y.  One copy per (u, g) pair, for variety."""
    out = []
    for u in glyphs.get("u", []):
        um = u.mask
        rows = np.nonzero(um.max(axis=1) > 0.3)[0]
        if len(rows) == 0:
            continue
        lower = um[rows[0] + int(0.6 * (rows[-1] - rows[0])):rows[-1] + 1]
        cols = np.nonzero(lower.max(axis=0) > 0.3)[0]
        if len(cols) == 0:
            continue
        u_bottom = rows[-1]
        # the right stroke of the u: rightmost ink in its lower part, ignoring an exit flick
        right = np.nonzero(um[rows[0]:rows[-1] + 1].max(axis=0) > 0.3)[0]
        x_u = float(np.percentile(right, 85))
        for g in glyphs.get("g", []):
            found = _descender(g, baseline)
            if found is None:
                continue
            tail, x_t = found
            dx = int(round(x_u - x_t))
            left = min(0, dx)
            width = max(um.shape[1], tail.shape[1] + dx) - left
            height = u_bottom + 1 + tail.shape[0]
            canvas = np.zeros((height, width), np.float32)
            canvas[:um.shape[0], -left:-left + um.shape[1]] = np.maximum(
                canvas[:um.shape[0], -left:-left + um.shape[1]], um[:height])
            region = canvas[u_bottom:u_bottom + tail.shape[0], dx - left:dx - left + tail.shape[1]]
            np.maximum(region, tail[:region.shape[0], :region.shape[1]], out=region)
            # join the u's right stroke to the tail's stem
            pen = max(2, int(round(stroke_width(um))))
            stem_rows = np.nonzero(um[:, int(x_u)] > 0.3)[0] if 0 <= int(x_u) < um.shape[1] else []
            y_from = int(stem_rows.max()) if len(stem_rows) else u_bottom
            cv2.line(canvas, (int(round(x_u)) - left, y_from), (int(round(x_t)) + dx - left, u_bottom + 2), 1.0, pen,
                     cv2.LINE_AA)
            cropped, r0, _ = tight(canvas)
            cropped = cv2.copyMakeBorder(cropped, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            top = u.top + (rows[0] - 2) / PX + (r0 - rows[0]) / PX
            out.append(Glyph(cropped, top, cropped.shape[1] / PX, cropped.shape[0] / PX))
    return out


LETTER_STYLES = {"round": round_y}


def latin_class(ch):
    if ch == "Δ":
        return "delta"
    if ch in charsets.UPPER or ch in charsets.DIGITS or ch == "Σ":
        return "cap"
    if ch in "acemnorsuvwxzαεθπσω":
        return "short"
    if ch in "bdhklt":
        return "tall"
    if ch in "gpqyjηβμ":
        return "desc"
    return None


class GlyphFactory:
    """Supplies a glyph for any character: handwritten, composed from jamo, or font fallback."""

    def __init__(self, profile, rng):
        self.p = profile
        self.rng = rng
        self.last = {}
        self.composed, self.fallback = set(), set()
        standalone = lambda j: self._pick(j).mask if j in profile.glyphs else None  # noqa: E731
        self.jamo = JamoSource(standalone)
        natural = {c: (gs[0].w, gs[0].h) for c, gs in profile.glyphs.items() if charsets.is_jamo(c)}
        self.composer = Composer(standalone, rng, natural)
        self.harvested = 0
        by_shape = {}
        self.parts = {}  # id(glyph) -> its jamo, for rebuilding varied copies
        self.vary = True
        for c, variants in profile.glyphs.items():
            if charsets.is_syllable(c):
                for g in variants:
                    parts = self.composer.learn(c, g.mask)
                    if parts:
                        self.parts[id(g)] = parts
                by_shape.setdefault(self.shape(c), []).extend(variants)
        rejected = self.composer.finalize()
        self.parts = {k: v for k, v in self.parts.items() if id(v) not in rejected}
        self.harvested = len(self.parts)
        # How big this writer makes each shape of syllable: 고 is flatter than 를, 이 narrower
        # than 했.  Unwritten syllables are built at the size of their written relatives.
        pad = 4 / PX  # glyph images carry a 2 px margin on each side
        self.shape_size = {key: (float(np.median([g.w for g in gs])) - pad,
                                 float(np.median([g.h for g in gs])) - pad,
                                 float(np.median([g.top + g.h for g in gs])) - pad / 2)
                           for key, gs in by_shape.items() if len(gs) >= 2}
        # Typical ink height of each kind of Latin letter (capital, short, tall, descending).
        by_class = {}
        for c, variants in profile.glyphs.items():
            cls = latin_class(c)
            if cls:
                by_class.setdefault(cls, []).extend(g.h - pad for g in variants)
        self.latin_size = {cls: float(np.median(v)) for cls, v in by_class.items() if len(v) >= 3}
        # How far this writer's descenders (g p y) reach below the line, relative to their height.
        depth = [(g.top + g.h - profile.latin_bottom) / max(g.h - pad, 0.1)
                 for c, gs in profile.glyphs.items() if latin_class(c) == "desc" for g in gs]
        self.desc_depth = float(np.clip(np.median(depth), 0.2, 0.6)) if depth else 0.4
        heights = [g.h - pad for c, gs in profile.glyphs.items() if charsets.is_syllable(c) for g in gs]
        self.syllable_h = float(np.median(heights)) if heights else profile.hangul_h - pad
        self._font_cache = {}
        self._skeletons = {}
        self._signatures = {}

    def _similar(self, glyph, mask):
        from .freeform import _signature
        key = id(glyph)
        if key not in self._signatures:
            self._signatures[key] = (glyph, _signature(glyph.mask))
        return float(self._signatures[key][1] @ _signature(mask))

    def latin_drop(self, ch, glyph, s):
        """How far (in the units of ``s``) the ink bottom of a Latin letter goes below the line:
        letters rest exactly on it, so a line's letters share one bottom, except that the tails
        of g p q y hang below it; punctuation keeps its own place."""
        cls = latin_class(ch)
        if cls == "desc":  # g p q y: the tail goes below the line, as far as the writer's own
            own = (glyph.top + glyph.h - self.p.latin_bottom) / max(glyph.h - 4 / PX, 0.1)
            return float(np.clip(own, 0.25, 0.55)) * (glyph.h - 4 / PX) * s
        if cls is not None:
            return 0.0  # every other letter's lowest point is on the line
        return (glyph.top + glyph.h - 2 / PX - (self.p.latin_bottom - 2 / PX)) * s

    def skeleton(self, glyph):
        if glyph.variant:
            return skeleton(glyph.mask)
        key = id(glyph)
        if key not in self._skeletons:
            self._skeletons[key] = (glyph, skeleton(glyph.mask))  # keep glyph alive: ids stay unique
        return self._skeletons[key][1]

    @staticmethod
    def shape(ch):
        _, vowel, final = charsets.decompose(ch)
        return vowel_class(vowel), bool(final)

    def typical_height(self, ch):
        """Ink height (profile mm) every copy of ``ch`` is drawn at: the writer's typical height
        for this kind of character, rather than whatever size it had in its template box."""
        if charsets.is_syllable(ch):
            return self.syllable_h  # one height for all: tops and bottoms each in one row
        cls = latin_class(ch)
        if cls == "delta":
            short = self.latin_size.get("short")
            return short * CLASS_SCALE["short"] * DELTA_OVER_SHORT if short else None
        size = self.latin_size.get(cls) if cls else None
        return size * CLASS_SCALE.get(cls, 1.0) if size else None

    def hangul_aspect(self, ch, glyph, extra=1.0):
        """Factor for a syllable's width: its own proportions, kept within ASPECT_TOLERANCE of
        the typical proportions of syllables of the same shape."""
        if not charsets.is_syllable(ch) or glyph.h <= 0.5:
            return extra
        own = (glyph.w - 4 / PX) / (glyph.h - 4 / PX)
        size = self._shape_size(ch)
        if not size or own <= 0:
            return extra
        typical = size[0] / size[1]
        want = float(np.clip(own * extra, typical * (1 - ASPECT_TOLERANCE), typical * (1 + ASPECT_TOLERANCE)))
        return want / own

    def _shape_size(self, ch):
        vclass, has_final = self.shape(ch)
        for key in ((vclass, has_final),
                    ({"right": "right_wide", "right_wide": "right"}.get(vclass, vclass), has_final)):
            if key in self.shape_size:
                return self.shape_size[key]
        return None

    def _pick(self, ch):
        variants = self.p.glyphs[ch]
        if len(variants) == 1:
            return variants[0]
        choices = [i for i in range(len(variants)) if i != self.last.get(ch)]
        i = int(self.rng.choice(choices))
        self.last[ch] = i
        return variants[i]

    def get(self, ch):
        """Return ``(glyph, category)``; category ``'hangul'`` or ``'latin'`` decides size/baseline."""
        category = "hangul" if (charsets.is_syllable(ch) or charsets.is_jamo(ch)) else "latin"
        if ch in self.p.glyphs:
            g = self._pick(ch)
            parts = self.parts.get(id(g))
            if parts and self.vary:
                mask = rebuild(ch, parts, compose_size(g.mask), self.composer, self.rng)
                inner, _, _ = tight(mask)
                # A rebuilt copy must still look like the syllable as written; when the cut
                # into jamo went wrong (a stroke shared by ㄹ and ㅗ) the original is used.
                if self._similar(g, inner) >= REBUILD_MIN_SIMILARITY:
                    inner = cv2.copyMakeBorder(inner, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                    return Glyph(inner, g.top, inner.shape[1] / PX, inner.shape[0] / PX, variant=True), category
            return g, category
        if charsets.is_syllable(ch):
            p = self.p
            w_mm, h_mm, bottom = self._shape_size(ch) or (p.hangul_w, p.hangul_h, p.hangul_bottom)
            w_px, h_px = max(8, int(round(w_mm * PX))), max(8, int(round(h_mm * PX)))
            mask = self.composer.compose(ch, w_px, h_px)
            if mask is not None and mask.max() > 0:
                self.composed.add(ch)
                mask = match_stroke(mask, p.stroke)
                pad = (mask.shape[0] - h_px) // 2
                cropped, r0, _ = tight(mask)
                r0 -= pad
                # Jamo rarely fill their boxes; grow the syllable to the writer's typical height
                # for this shape, without letting it get wider than such syllables are.
                grow = min(1.7, max(1.0, 0.92 * h_px / cropped.shape[0]),
                           max(1.0, 1.35 * w_px / cropped.shape[1]))
                if grow > 1.0:
                    cropped = cv2.resize(cropped, None, fx=grow, fy=grow, interpolation=cv2.INTER_LINEAR)
                    r0 = h_px - (h_px - r0) * grow
                return Glyph(cropped, bottom - h_mm + r0 / PX,
                             cropped.shape[1] / PX, cropped.shape[0] / PX), category
        if charsets.is_jamo(ch):
            mask = self.jamo.get(ch)  # e.g. ㄺ written on its own: join ㄹ and ㄱ
            if mask is not None:
                self.composed.add(ch)
                cropped, _, _ = tight(mask)
                return Glyph(cropped, self.p.hangul_bottom - cropped.shape[0] / PX,
                             cropped.shape[1] / PX, cropped.shape[0] / PX), category
        self.fallback.add(ch)
        return self._font_glyph(ch, category), category

    def _font_glyph(self, ch, category):
        if ch in self._font_cache:
            return self._font_cache[ch]
        p = self.p
        em = (p.hangul_h / 0.86 if category == "hangul" else p.cap_h / 0.72) * PX
        font = _fallback_font(max(8, int(round(em))))
        size = int(em * 2.2)
        canvas = Image.new("L", (size, size), 0)
        baseline_y = int(size * 0.7)
        ImageDraw.Draw(canvas).text((size // 4, baseline_y), ch, font=font, fill=255, anchor="ls")
        mask = np.asarray(canvas, np.float32) / 255.0
        before = mask.shape[0]
        mask = match_stroke(mask, p.stroke)  # printed strokes -> the user's pen width
        baseline_y += (mask.shape[0] - before) // 2
        cropped, r0, _ = tight(mask)
        if category == "hangul":
            top = p.hangul_bottom - cropped.shape[0] / PX
        else:
            top = p.latin_bottom + (r0 - baseline_y) / PX
        glyph = Glyph(cropped, top, cropped.shape[1] / PX, cropped.shape[0] / PX)
        self._font_cache[ch] = glyph
        return glyph


# --------------------------------------------------------------------------- page drawing

def _paper(opts, width, height, r, rng):
    page = np.empty((height, width, 3), np.uint8)
    if opts.paper == "grid":
        page[:] = GRID_PAPER
    else:
        page[:] = (250, 249, 244) if opts.paper != "blank" else (252, 252, 250)
    noise = rng.normal(0, 1.6, (height // 4 + 1, width // 4 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)[..., None]
    page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    top, bottom = layout_bounds(opts)
    if opts.paper == "lined":
        y = top
        while y <= bottom + 0.01:
            cv2.line(page, (0, int(y * r)), (width, int(y * r)), (165, 198, 222), max(1, int(r * 0.12)),
                     cv2.LINE_AA)
            y += opts.line_pitch
    elif opts.paper == "grid":
        thick = max(1, int(round(0.15 * r)))
        for x in np.arange(GRID_STEP, 210.0, GRID_STEP):
            cv2.line(page, (int(x * r), 0), (int(x * r), height), GRID_LINE, thick, cv2.LINE_AA)
        for y in np.arange(GRID_STEP, 297.0, GRID_STEP):
            cv2.line(page, (0, int(y * r)), (width, int(y * r)), GRID_LINE, thick, cv2.LINE_AA)
    if opts.margin_line and opts.paper != "blank":
        cv2.line(page, (int(24 * r), 0), (int(24 * r), height), (232, 150, 150), max(1, int(r * 0.15)),
                 cv2.LINE_AA)
    return page.astype(np.float32) / 255.0


def layout_bounds(opts):
    """Y (mm) of the first and last ruled line that carries text."""
    return 30.0, 282.0


def _transform(mask, scale, angle, elastic, rng, aspect=1.0):
    """Resize to output resolution (``aspect`` narrows or widens), then rotate and wobble a
    little. Returns (mask, pad)."""
    h, w = mask.shape
    nw, nh = max(1, int(round(w * scale * aspect))), max(1, int(round(h * scale)))
    m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pad = int(max(nw, nh) * 0.12) + 2
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = m.shape
    rot = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    m = cv2.warpAffine(m, rot, (W, H), flags=cv2.INTER_LINEAR)
    if elastic > 0:
        amp = elastic * max(nw, nh)
        dx = cv2.resize(rng.normal(0, 1, (4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        dy = cv2.resize(rng.normal(0, 1, (4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * amp
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        m = cv2.remap(m, xs + dx, ys + dy, cv2.INTER_LINEAR)
    return np.clip(m, 0, 1), pad


@dataclass
class Placed:
    mask: np.ndarray
    left: int         # px, first ink column in ``mask``
    bottom: int       # px, last ink row in ``mask``
    width: float      # px, width of the ink (advance before the gap)
    drop: float       # px, how far the ink bottom sits below the baseline
    gap: float        # px, space after the glyph


def skeleton(mask):
    """Centre line of the strokes (Zhang-Suen thinning): the path the pen tip travelled.

    Thin, light strokes (a fine iPad pen) are only one or two faint pixels wide, so the mask
    is doubled in size and taken at a low threshold first -- otherwise short strokes such as
    the top bar of ㄷ or ㅍ vanish and 대 turns into 내.  The centre line is then reduced back
    to the glyph's resolution."""
    up = cv2.resize(mask.astype(np.float32), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    binary = cv2.morphologyEx((up > 0.22).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    centre = _thin(binary)
    h, w = mask.shape
    centre = centre[:2 * h, :2 * w].reshape(h, 2, w, 2).max(axis=(1, 3))
    # Dots (the top of ㅎ, a full stop) can vanish entirely; keep one centre pixel of each.
    n, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0.22).astype(np.uint8))
    for i in range(1, n):
        if not centre[labels == i].any():
            centre[int(round(cents[i][1])), int(round(cents[i][0]))] = 1.0
    return centre


def _thin(binary):
    img = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(n.astype(np.int16) for n in ring[:8])
            turns = sum(((a == 0) & (b == 1)).astype(np.int16) for a, b in zip(ring, ring[1:]))
            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (neighbours >= 2) & (neighbours <= 6) & (turns == 1) & side
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(np.float32)


def uniform_pen(mask, target):
    """Redraw a glyph as if written with a monoline pen ``target`` px wide.

    The centre line of every stroke is found (on an enlarged copy, so thin strokes keep their
    shape), little spurs that thinning leaves at corners are pruned, each stroke end is pushed
    back out to the tip of the original stroke (thinning eats into it by half the stroke
    width), and the whole centre line is covered by one round pen.  Thick and thin parts of the
    original writing therefore all come out exactly as wide as the pen."""
    binary = (mask > 0.35).astype(np.uint8)
    if binary.sum() < 3:
        return mask
    width = stroke_width(binary.astype(np.float32))
    f = int(np.clip(np.ceil(7.0 / max(width, 1.0)), 1, 4))
    pad = int(np.ceil(target)) + 3
    big = cv2.copyMakeBorder(mask.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if f > 1:
        big = cv2.resize(big, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 0.6 * f)
    binary = (big > 0.35).astype(np.uint8)
    centre = _thin(binary)
    radius = max(0.5, (target - 1) / 2) * f
    # Thinning leaves spurs about half a stroke width long at corners; real short strokes
    # (the arm of a y, the tick of a ㅏ) are longer and must stay.
    _prune_spurs(centre, max(2, int(0.6 * width * f)))
    _extend_ends(centre, binary, radius)
    pen = _restroke(centre, radius)
    if f > 1:
        h, w = pen.shape
        pen = cv2.resize(pen, (w // f, h // f), interpolation=cv2.INTER_AREA)
        pen = np.clip(pen * 1.1, 0, 1)
    return pen[pad:pad + mask.shape[0], pad:pad + mask.shape[1]]


NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbours(on, y, x):
    return [(y + dy, x + dx) for dy, dx in NEIGHBOURS if (y + dy, x + dx) in on]


def _prune_spurs(centre, length):
    """Remove short side branches (an end that reaches a junction within ``length`` px)."""
    ys, xs = np.nonzero(centre)
    on = set(zip(ys.tolist(), xs.tolist()))
    for y, x in list(on):
        if (y, x) not in on or len(_neighbours(on, y, x)) != 1:
            continue
        path, cur, prev = [(y, x)], (y, x), None
        while len(path) <= length:
            nb = [p for p in _neighbours(on, *cur) if p != prev and p not in path]
            if len(nb) != 1:
                break
            prev, cur = cur, nb[0]
            if len(_neighbours(on, *cur)) >= 3:  # reached a junction: this was a spur
                for p in path:
                    on.discard(p)
                    centre[p] = 0
                break
            path.append(cur)


def _extend_ends(centre, binary, radius):
    """Push every stroke end out along its direction to the tip of the original stroke,
    stopping one pen radius short so the round pen ends where the stroke ended."""
    H, W = centre.shape
    ys, xs = np.nonzero(centre)
    on = set(zip(ys.tolist(), xs.tolist()))
    for y, x in list(on):
        nb = _neighbours(on, y, x)
        if len(nb) != 1:
            continue
        path, cur = [(y, x)], nb[0]
        for _ in range(int(2 * radius) + 4):
            path.append(cur)
            nxt = [p for p in _neighbours(on, *cur) if p not in path]
            if len(nxt) != 1:
                break
            cur = nxt[0]
        by, bx = path[-1]
        vy, vx = y - by, x - bx
        norm = float(np.hypot(vy, vx))
        if norm < 1:
            continue
        vy, vx = vy / norm, vx / norm
        t = 0.0
        while True:
            py, px = int(round(y + vy * (t + 0.5))), int(round(x + vx * (t + 0.5)))
            if not (0 <= py < H and 0 <= px < W) or not binary[py, px]:
                break
            t += 0.5
        reach = t - radius
        step = 0.5
        while step <= reach:
            centre[int(round(y + vy * step)), int(round(x + vx * step))] = 1.0
            step += 0.5


def pen_glyph(glyph_mask, scale, angle, elastic, rng, aspect, target):
    """Scale a glyph to output size and draw it with the monoline pen.  Small letters (formula
    symbols, indices) are drawn at up to 3x and reduced afterwards, so the humps of an m or the
    dot of an i do not melt together while the centre line is traced."""
    height = glyph_mask.shape[0] * scale
    f = int(np.clip(np.ceil(70.0 / max(height, 1.0)), 1, 3))
    mask, pad = _transform(glyph_mask, scale * f, angle, elastic, rng, aspect)
    mask = uniform_pen(mask, target * f)
    if f > 1:
        h, w = mask.shape
        mask = cv2.resize(mask, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
        mask = np.clip(mask * 1.15, 0, 1)
    return mask


def normalize_pen(mask, target):
    """Make the strokes of a rendered glyph ``target`` px wide on average, by growing or
    shrinking them evenly from their edges.  Unlike redrawing the centre line, this keeps
    short strokes (the hook of a ㄷ, the tick of a ㅏ) exactly as long as they were written."""
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() < 4:
        return mask
    current = stroke_width(binary.astype(np.float32))
    radius = (target - current) / 2
    if abs(radius) < 0.2:
        return np.clip((mask - 0.25) / 0.5, 0, 1)
    pad = int(np.ceil(abs(radius))) + 2
    b = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    if radius > 0:  # thicken: anything within radius of the ink
        dist = cv2.distanceTransform((b == 0).astype(np.uint8), cv2.DIST_L2, 5)
        out = np.clip(radius + 0.5 - dist, 0, 1)
        out = np.maximum(out, b.astype(np.float32))
    else:  # thin: keep what lies deeper than radius inside the ink, never below one pixel
        dist = cv2.distanceTransform(b, cv2.DIST_L2, 5)
        out = np.where(b > 0, np.clip(dist + radius + 0.5, 0, 1), 0).astype(np.float32)
        keep = skeleton_fast(b)  # the middle of every stroke survives, so nothing breaks
        out = np.maximum(out, keep)
    return out[pad:-pad, pad:-pad]


def skeleton_fast(binary):
    """Medial pixels of strokes (local maxima of the distance to the background)."""
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    peak = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    return ((dist >= peak - 0.5) & (binary > 0)).astype(np.float32)


def _restroke(centre, radius):
    """Draw a centre line with a round pen of the given radius (px), anti-aliased."""
    pad = int(np.ceil(radius)) + 2
    line = cv2.copyMakeBorder(centre, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform((line == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(radius + 0.5 - dist, 0, 1)[pad:-pad, pad:-pad]


def _crisp(mask):
    """Firm up anti-aliased edges so the ink reads as solid pen strokes."""
    return np.clip((mask - 0.12) / 0.6, 0, 1)


class Renderer:
    def __init__(self, profile, opts):
        self.p, self.o = profile, opts
        self.rng = np.random.default_rng(opts.seed)
        self.r = opts.dpi / 25.4  # output px per mm
        self.factory = GlyphFactory(profile, self.rng)
        self.hangul_h = opts.line_pitch * opts.size
        self.scale = {"hangul": self.hangul_h / profile.hangul_h,
                      "latin": self.hangul_h * 0.86 / profile.cap_h}
        if opts.ink == "auto" and profile.ink:
            # The measured colour is lightened by anti-aliasing and the scan; keep its hue but
            # make it as deep as a pen stroke really is.
            ink = np.clip(np.array(profile.ink), 0, 1)
            self.ink = tuple(float(v) for v in ink * (0.13 / max(float(ink.max()), 0.13)))
        else:
            self.ink = INK_COLORS.get(opts.ink, INK_COLORS["black"])
        self.target_stroke = opts.stroke * self.hangul_h * self.r  # px
        # Ink height of a typical syllable at this size (glyph images carry a 0.4 mm margin).
        self.typical_ink_px = self.scale["hangul"] * (profile.hangul_h - 4 / PX) * self.r

    def place(self, ch):
        o, mess, rng = self.o, self.o.messiness, self.rng
        glyph, category = self.factory.get(ch)
        syllable = charsets.is_syllable(ch)
        s_mm = self.scale[category] * (1 + (0 if syllable else rng.normal(0, 0.02 * mess)))
        if not syllable:
            s_mm *= self.p.char_scale.get(ch, 1.0)
        ink_h = glyph.h - 4 / PX  # the glyph image carries a 2 px margin
        typical = self.factory.typical_height(ch)
        if typical and ink_h > 0 and o.evenness > 0:
            s_mm *= float(np.clip((typical / ink_h) ** o.evenness, 0.6, 1.6))
        scale = s_mm * self.r / PX
        angle, elastic = rng.normal(0, 0.4 * mess), 0.006 * mess
        if o.stroke > 0:
            # Same pen width for every character, so letters copied, assembled from jamo or
            # taken from the fallback font all match -- the strokes themselves are kept.
            aspect = (HANGUL_ASPECT if category == "hangul" else 1.0) * \
                self.factory.hangul_aspect(ch, glyph, self.p.char_width.get(ch, 1.0))
            mask = pen_glyph(glyph.mask, scale, angle, elastic, rng, aspect, self.target_stroke)
        else:
            mask, _ = _transform(glyph.mask, scale, angle, elastic, rng,
                                 HANGUL_ASPECT if category == "hangul" else 1.0)
            mask = _crisp(mask)
        cols = np.nonzero(mask.max(axis=0) > 0.2)[0]
        rows = np.nonzero(mask.max(axis=1) > 0.2)[0]
        if len(cols) == 0:
            cols = rows = np.array([0])
        if category == "hangul":
            drop = 0.0  # every syllable rests on the line: bottoms in one row
        elif ch in MID_LINE:
            # Dashes, operators and brackets are centred on the writing, however they were
            # placed in their template box.
            drop = -(0.42 * self.hangul_h * self.r - (rows[-1] - rows[0] + 1) / 2)
        else:
            drop = self.factory.latin_drop(ch, glyph, s_mm * self.r)
        gap = (LETTER_GAP[category] * o.letter_spacing + rng.normal(0, GAP_WANDER * mess)) * self.hangul_h
        # Neighbours may touch, not overlap much; a gap inside a word never looks like a space.
        gap = float(np.clip(gap, -0.08 * self.hangul_h, 0.0))
        return Placed(mask, int(cols[0]), int(rows[-1]), float(cols[-1] - cols[0] + 1), drop,
                      gap * self.r)

    def space(self):
        return WORD_GAP * self.hangul_h * self.o.word_spacing * self.r * (1 + self.rng.normal(0, 0.08 * self.o.messiness))

    def render(self, text, max_pages=40):
        o, r = self.o, self.r
        text = charsets.normalize_text(text)
        width, height = int(round(210 * r)), int(round(297 * r))
        left, right = 28.0 * r, 192.0 * r
        top, bottom = layout_bounds(o)
        lines_per_page = int((bottom - top) / o.line_pitch) + 1

        # Break the text into lines of placed glyphs.
        lines = []
        for paragraph in text.split("\n"):
            current, x = [], 0.0
            for token in re.findall(r"\S+| +", paragraph):
                if token.isspace():
                    if current:
                        current.append(None)
                        x += self.space() * len(token)
                    continue
                word = [self.place(c) for c in token]
                w = sum(g.width + g.gap for g in word)
                if current and x + w > right - left:
                    while current and current[-1] is None:
                        current.pop()
                    lines.append(current)
                    current, x = [], 0.0
                for g in word:
                    if current and x + g.width > right - left:
                        lines.append(current)
                        current, x = [], 0.0
                    current.append(g)
                    x += g.width + g.gap
            lines.append(current)
            if len(lines) > lines_per_page * max_pages:
                break

        pages = []
        for start in range(0, max(len(lines), 1), lines_per_page):
            page = _paper(o, width, height, r, self.rng)
            for i, line in enumerate(lines[start:start + lines_per_page]):
                self._draw_line(page, line, (top + i * o.line_pitch) * r, left)
            pages.append(Image.fromarray((np.clip(page, 0, 1) * 255).astype(np.uint8)))
            if len(pages) >= max_pages:
                break
        return pages

    def _draw_line(self, page, line, rule_y, left):
        rng, r, mess = self.rng, self.r, self.o.messiness
        baseline = rule_y - 0.7 * r  # writing sits just above the ruled line
        x = left
        ink = np.array(self.ink, np.float32)
        jitter = VERTICAL_WANDER * mess * self.typical_ink_px
        spacer = InkSpacer(0.12 * self.typical_ink_px)
        for g in line:
            if g is None:
                x += self.space()
                spacer.reset()
                continue
            y0 = int(round(baseline + g.drop + (rng.normal(0, jitter) if jitter else 0.0))) - g.bottom
            target = (INK_GAP * self.o.letter_spacing + rng.normal(0, 0.012 * mess)) * self.typical_ink_px
            x0 = spacer.place(g.mask, y0, int(round(x)) - g.left, target)
            _composite(page, g.mask, x0, y0, ink)
            x = x0 + g.left + g.width


class InkSpacer:
    """Places letters one after another so that the nearest ink of neighbours is a set
    distance apart.  Spacing by bounding boxes makes the visible gap depend on the letters'
    shapes (이 then 러 looked jammed together, 고 then 다 far apart)."""

    def __init__(self, band):
        self.band = max(1, int(band))  # rows above/below also count (slanted strokes)
        self.reset()

    def reset(self):
        self.top = None
        self.right = None
        self.end = None

    def place(self, mask, y0, x_box, target):
        """Left edge (page px) for ``mask`` with top row ``y0``.  ``x_box`` is where bounding-box
        spacing would put it; it is used when the letters share no rows."""
        ink = mask > 0.3
        rows = np.nonzero(ink.any(axis=1))[0]
        if len(rows) == 0:
            return x_box
        left = ink[rows].argmax(axis=1)
        right = ink.shape[1] - 1 - ink[rows][:, ::-1].argmax(axis=1)
        x0 = x_box
        if self.right is not None:
            idx = rows + y0 - self.top
            ok = (idx >= 0) & (idx < len(self.right))
            if ok.any():
                need = self.right[idx[ok]] - left[ok]
                need = need[np.isfinite(need)]
                if len(need):
                    x0 = int(round(need.max() + target))
            # never pull a letter back over the previous one's middle
            x0 = max(x0, int(round(self.end - 0.25 * (self.end - self.start))) - int(left.min()))
        # Remember this letter's right edge per row, widened by ``band`` rows.
        top = y0 + rows.min()
        prof = np.full(rows.max() - rows.min() + 1 + 2 * self.band, -np.inf)
        for r, rx in zip(rows - rows.min(), right + x0):
            lo, hi = r, r + 2 * self.band + 1
            prof[lo:hi] = np.maximum(prof[lo:hi], rx)
        self.top, self.right = top - self.band, prof
        self.start, self.end = x0 + int(left.min()), x0 + int(right.max())
        return x0


def _composite(page, alpha, x0, y0, ink):
    H, W = page.shape[:2]
    h, w = alpha.shape
    ax0, ay0 = max(0, -x0), max(0, -y0)
    px0, py0 = max(0, x0), max(0, y0)
    px1, py1 = min(W, x0 + w), min(H, y0 + h)
    if px1 <= px0 or py1 <= py0:
        return
    a = alpha[ay0:ay0 + py1 - py0, ax0:ax0 + px1 - px0][..., None]
    region = page[py0:py1, px0:px1]
    region *= 1.0 - a * (1.0 - ink)  # ink multiplies the paper (lines show through)


# --------------------------------------------------------------------------- outputs

def pages_to_pdf(pages):
    doc = pymupdf.open()
    for image in pages:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92)
        page = doc.new_page(width=595.28, height=841.89)
        page.insert_image(page.rect, stream=buf.getvalue())
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def contact_sheet(profile, columns=14, tile=72):
    """Overview image of every extracted glyph, for checking the extraction."""
    items = [(c, g) for c, gs in profile.glyphs.items() for g in gs]
    rows = max(1, (len(items) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile, rows * (tile + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _fallback_font(12)
    for i, (c, g) in enumerate(items):
        row, col = divmod(i, columns)
        x, y = col * tile, row * (tile + 18)
        h, w = g.mask.shape
        s = min((tile - 10) / w, (tile - 10) / h, 1.0)
        m = cv2.resize(g.mask, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        img = Image.fromarray((255 - m * 255).astype(np.uint8))
        sheet.paste(img, (x + (tile - img.width) // 2, y + (tile - img.height) // 2))
        draw.rectangle((x, y, x + tile - 1, y + tile + 17), outline=(225, 225, 225))
        draw.text((x + tile // 2, y + tile + 8), c, font=font, fill=(200, 40, 40), anchor="mm")
    return sheet
