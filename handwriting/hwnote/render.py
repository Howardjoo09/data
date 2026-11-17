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
from .compose import JamoSource, compose_syllable, match_stroke, stroke_width, tight

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


@dataclass
class Options:
    paper: str = "lined"        # lined | grid | blank
    ink: str = "auto"           # auto | black | blue | red | pencil
    line_pitch: float = 8.0     # mm between ruled lines
    size: float = 0.6           # Hangul height as a fraction of the line pitch
    letter_spacing: float = 1.0
    word_spacing: float = 1.0
    messiness: float = 0.5      # 0 = perfectly regular, 1 = very loose
    margin_line: bool = True
    dpi: int = 200
    seed: int | None = None


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
        self.glyphs = {}
        for g in data["glyphs"]:
            mask = cv2.imread(os.path.join(directory, g["file"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            self.glyphs.setdefault(g["char"], []).append(
                Glyph(mask.astype(np.float32) / 255.0, g["top"], g["w"], g["h"]))
        self._stats(data.get("baseline", 11.5))

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


class GlyphFactory:
    """Supplies a glyph for any character: handwritten, composed from jamo, or font fallback."""

    def __init__(self, profile, rng):
        self.p = profile
        self.rng = rng
        self.last = {}
        self.composed, self.fallback = set(), set()
        self.jamo = JamoSource(lambda j: self._pick(j).mask if j in profile.glyphs else None)
        self._font_cache = {}

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
            return self._pick(ch), category
        if charsets.is_syllable(ch):
            p = self.p
            w_px, h_px = int(round(p.hangul_w * PX)), int(round(p.hangul_h * PX))
            mask = compose_syllable(ch, self.jamo, w_px, h_px)
            if mask is not None and mask.max() > 0:
                self.composed.add(ch)
                mask = match_stroke(mask, p.stroke)
                pad = (mask.shape[0] - h_px) // 2
                cropped, r0, _ = tight(mask)
                r0 -= pad
                # Jamo rarely fill their boxes; grow the syllable towards the typical written size.
                grow = min(1.25, max(1.0, 0.95 * h_px / cropped.shape[0]))
                if grow > 1.0:
                    cropped = cv2.resize(cropped, None, fx=grow, fy=grow, interpolation=cv2.INTER_LINEAR)
                    r0 = h_px - (h_px - r0) * grow
                return Glyph(cropped, p.hangul_bottom - p.hangul_h + r0 / PX,
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
        step = 5.0
        for x in np.arange(5.0, 210.0, step):
            cv2.line(page, (int(x * r), 0), (int(x * r), height), (222, 222, 218), 1, cv2.LINE_AA)
        for y in np.arange(5.0, 297.0, step):
            cv2.line(page, (0, int(y * r)), (width, int(y * r)), (222, 222, 218), 1, cv2.LINE_AA)
    if opts.margin_line and opts.paper != "blank":
        cv2.line(page, (int(24 * r), 0), (int(24 * r), height), (232, 150, 150), max(1, int(r * 0.15)),
                 cv2.LINE_AA)
    return page.astype(np.float32) / 255.0


def layout_bounds(opts):
    """Y (mm) of the first and last ruled line that carries text."""
    return 30.0, 282.0


def _transform(mask, scale, angle, elastic, rng):
    """Resize to output resolution, then rotate and wobble a little. Returns (mask, pad)."""
    h, w = mask.shape
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
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
    pad: int
    width: float      # px, advance width of the ink
    drop: float       # px, how far the glyph bottom sits below the baseline
    gap: float        # px, space after the glyph
    opacity: float


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
            ink = np.clip(np.array(profile.ink), 0, 1) ** 1.5  # photos wash colours out; deepen them
            if ink.max() > 0.5:
                ink *= 0.5 / ink.max()
            self.ink = tuple(float(v) for v in ink)
        else:
            self.ink = INK_COLORS.get(opts.ink, INK_COLORS["black"])

    def place(self, ch):
        o, mess, rng = self.o, self.o.messiness, self.rng
        glyph, category = self.factory.get(ch)
        bottom_ref = self.p.hangul_bottom if category == "hangul" else self.p.latin_bottom
        s_mm = self.scale[category] * (1 + rng.normal(0, 0.035 * mess))
        mask, pad = _transform(glyph.mask, s_mm * self.r / PX, rng.normal(0, 1.8 * mess),
                               0.018 * mess, rng)
        gap_mm = (0.11 if category == "hangul" else 0.05) * self.hangul_h * o.letter_spacing
        gap_mm += rng.normal(0, 0.12 * mess)
        return Placed(mask, pad, glyph.w * s_mm * self.r, (glyph.top + glyph.h - bottom_ref) * s_mm * self.r,
                      max(gap_mm, -0.2) * self.r, float(rng.uniform(0.86, 1.0)))

    def space(self):
        return self.hangul_h * 0.42 * self.o.word_spacing * self.r * (1 + self.rng.normal(0, 0.12 * self.o.messiness))

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
        o, rng, r, mess = self.o, self.rng, self.r, self.o.messiness
        slope = np.tan(np.radians(rng.normal(0, 0.35 * mess)))
        wave_amp, wave_len, phase = rng.uniform(0, 0.35 * mess) * r, rng.uniform(60, 140) * r, rng.uniform(0, 6.3)
        baseline = rule_y - 0.7 * r  # writing sits just above the ruled line
        x = left + rng.normal(0, 0.6 * mess) * r
        ink = np.array(self.ink, np.float32)
        for g in line:
            if g is None:
                x += self.space()
                continue
            dx = x - left
            y_bottom = (baseline + g.drop + dx * slope + wave_amp * np.sin(dx / wave_len * 6.283 + phase)
                        + rng.normal(0, 0.25 * mess) * r)
            mh, mw = g.mask.shape
            x0 = int(round(x + rng.normal(0, 0.15 * mess) * r)) - g.pad
            y0 = int(round(y_bottom)) - (mh - 2 * g.pad) - g.pad
            _composite(page, g.mask * g.opacity, x0, y0, ink)
            x += g.width + g.gap


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
        image.save(buf, "JPEG", quality=80)
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
