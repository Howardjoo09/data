"""Rewrite a typeset PDF page by hand, keeping its layout.

Every character of the PDF is replaced by the writer's glyph at the same place and size, so
fractions, sub- and superscripts and aligned equations stay where the typesetter put them.
Fraction bars, boxes and underlines are redrawn as pen lines; tall brackets built from pieces
become one stretched handwritten bracket; pictures are copied.  Everything is written with
one pen whose width follows the body text, as a person writing the page would.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf
from PIL import Image

from . import charsets, mathfont
from .render import (GAP_WANDER, HANGUL_ASPECT, LETTER_GAP, MID_LINE, VERTICAL_WANDER, WORD_GAP, PX, GlyphFactory, Options, _crisp, _paper, pen_glyph,
                     _transform)

DELIMS = set("()[]{}|")
WIDE_OPS = set("=+-×÷±∓→←⇒⇐⇔<>≤≥≠≈~∈∉⊂∪∩·⋯")
HANGUL_HEIGHT = 0.84  # handwritten syllable height as a fraction of the printed font size
MATH_GAP = 0.05       # space between packed formula characters, as a fraction of the font size
BIG_OPS = {"Σ": 0.95, "∫": 1.15}
PIECE_TO_DELIM = {"⎧": "{", "⎨": "{", "⎩": "{", "⎪": "{", "⎫": "}", "⎬": "}", "⎭": "}",
                  "⎡": "[", "⎣": "[", "⎢": "[", "⎤": "]", "⎦": "]", "⎥": "]"}


@dataclass
class Char:
    ch: str
    bbox: tuple        # x0, y0, x1, y1 in points
    origin: tuple      # baseline point
    size: float        # font size in points
    color: tuple       # 0..1 RGB


def read_page(page):
    """Characters (decoded), drawn lines and picture areas of a PDF page."""
    chars = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                c = span.get("color", 0)
                color = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
                for item in span["chars"]:
                    ch = mathfont.readable(item["c"])
                    if not ch or ch.isspace():
                        continue
                    chars.append(Char(charsets.normalize_text(ch) or ch, tuple(item["bbox"]),
                                      tuple(item["origin"]), span["size"], tuple(v / 255 for v in color)))
    lines = []
    for drawing in page.get_drawings():
        if not drawing.get("color"):
            continue  # fills only (shading): not written by hand
        width = drawing.get("width") or 0.5
        for item in drawing["items"]:
            if item[0] == "l":
                lines.append((item[1].x, item[1].y, item[2].x, item[2].y, width))
            elif item[0] == "re":
                r = item[1]
                for a, b in (((r.x0, r.y0), (r.x1, r.y0)), ((r.x1, r.y0), (r.x1, r.y1)),
                             ((r.x1, r.y1), (r.x0, r.y1)), ((r.x0, r.y1), (r.x0, r.y0))):
                    lines.append((*a, *b, width))
    pictures = []
    for info in page.get_images(full=True):
        for rect in page.get_image_rects(info[0]):
            if rect.width > 5 and rect.height > 5:
                pictures.append(tuple(rect))
    return chars, lines, pictures


def _group_pieces(chars):
    """Join stacked bracket pieces (⎧ ⎪ ⎨ ⎪ ⎩) into single tall delimiters."""
    pieces = [c for c in chars if c.ch in PIECE_TO_DELIM]
    rest = [c for c in chars if c.ch not in PIECE_TO_DELIM]
    groups = []
    for c in sorted(pieces, key=lambda c: (c.bbox[0], c.bbox[1])):
        kind = PIECE_TO_DELIM[c.ch]
        for g in groups:
            last = g[-1]
            if (PIECE_TO_DELIM[last.ch] == kind and abs(last.bbox[0] - c.bbox[0]) < 3
                    and -2 < c.bbox[1] - last.bbox[3] < 4):
                g.append(c)
                break
        else:
            groups.append([c])
    for g in groups:
        x0 = min(c.bbox[0] for c in g)
        x1 = max(c.bbox[2] for c in g)
        y0 = min(c.bbox[1] for c in g)
        y1 = max(c.bbox[3] for c in g)
        rest.append(Char(PIECE_TO_DELIM[g[0].ch], (x0, y0, x1, y1), (x0, y1), (y1 - y0), g[0].color))
    return rest


class PageWriter:
    def __init__(self, profile, opts=None, seed=None):
        self.p = profile
        self.o = opts or Options(paper="grid", margin_line=False)
        self.rng = np.random.default_rng(seed if seed is not None else self.o.seed)
        self.factory = GlyphFactory(profile, self.rng)
        self.k = self.o.dpi / 72.0  # px per point

    # ---------------------------------------------------------------- helpers
    def _glyph(self, ch, height_px, category):
        """Pen-drawn mask of ``ch`` whose Hangul/capital height is ``height_px``."""
        o, rng, mess = self.o, self.rng, self.o.messiness
        glyph, cat = self.factory.get(ch)
        ref = self.p.hangul_h if cat == "hangul" else self.p.cap_h
        s = height_px / ref * (1 + rng.normal(0, 0.015 * mess)) * self.p.char_scale.get(ch, 1.0)
        ink_h = glyph.h - 4 / PX
        if cat == "hangul" and charsets.is_syllable(ch) and o.evenness > 0:
            typical = self.factory._shape_size(ch)
            if typical and ink_h > 0:
                s *= float(np.clip((typical[1] / ink_h) ** o.evenness, 0.8, 1.25))
        return glyph, cat, s

    def _plan_symbol(self, c):
        """Glyph and scale (px per profile-mm) of a non-Hangul character: decided once, so
        the line can be packed with the width it will really be drawn at."""
        k = self.k
        em = c.size * k
        x0, y0, x1, y1 = (v * k for v in c.bbox)
        glyph, cat = self.factory.get(c.ch)
        ink_h = max(glyph.h - 4 / PX, 0.3)
        ink_w = max(glyph.w - 4 / PX, 0.3)
        if c.ch in DELIMS:
            tall = (y1 - y0) > 1.6 * em
            s = (0.92 * (y1 - y0) if tall else 0.82 * em) / ink_h
        elif c.ch in BIG_OPS:
            s = BIG_OPS[c.ch] * em / ink_h
        elif c.ch in WIDE_OPS:
            # Operators are sized by the printed symbol's width (written small in a box).
            s = min(0.72 * (x1 - x0) / ink_w, 0.55 * em / ink_h)
        elif c.ch in ("'", "″"):
            s = 0.32 * em / ink_h
        else:
            glyph, cat, s = self._glyph(c.ch, 0.66 * em, "latin")
            ink_w = max(glyph.w - 4 / PX, 0.3)
        return glyph, s, ink_w * s / k  # width in points

    def _render(self, glyph, s, angle, aspect=1.0):
        """Pen-drawn mask of a glyph at ``s`` px per profile-mm, with its ink box."""
        scale = s / PX
        if self.o.stroke > 0:
            mask = pen_glyph(glyph.mask, scale, angle, 0.004 * self.o.messiness, self.rng, aspect,
                             min(self.pen, 0.15 * glyph.h * s))
        else:
            mask, _ = _transform(glyph.mask, scale, angle, 0.0, self.rng, aspect)
            mask = _crisp(mask)
        rows = np.nonzero(mask.max(axis=1) > 0.2)[0]
        cols = np.nonzero(mask.max(axis=0) > 0.2)[0]
        if len(rows) == 0:
            return mask, None
        return mask, (int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1]))

    def _draw(self, layer, glyph, s, angle, ink_box_fn):
        mask, box = self._render(glyph, s, angle)
        if box is None:
            return
        x, y = ink_box_fn(*box)
        _stamp(layer, mask, int(round(x)), int(round(y)))

    def _hand_line(self, layer, x0, y0, x1, y1, width):
        """A ruled line drawn by hand: slightly uneven, never perfectly straight."""
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 1:
            return
        n = max(2, int(length / (4 * self.k)))
        t = np.linspace(0, 1, n)
        bend = np.cumsum(self.rng.normal(0, 1, n))
        bend -= np.linspace(bend[0], bend[-1], n)
        bend *= 0.25 * self.k * self.o.messiness / max(1.0, np.abs(bend).max())
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length
        pts = np.stack([x0 + (x1 - x0) * t + nx * bend, y0 + (y1 - y0) * t + ny * bend], axis=1)
        cv2.polylines(layer, [np.round(pts * 4).astype(np.int32)], False, 1.0,
                      max(1, int(round(width))), cv2.LINE_AA, shift=2)

    def _write_word(self, word, layer_for, plans, shift):
        """Hangul of one word: same height as print, but packed like handwriting from the
        word's (shifted) printed start -- print gives every syllable a full square."""
        k, o = self.k, self.o
        x = (word[0].bbox[0] + shift.get(id(word[0]), 0.0)) * k
        for c in word:
            em = c.size * k
            glyph, cat, s = plans[id(c)]
            mask, box = self._render(glyph, s, self.rng.normal(0, 0.6 * o.messiness),
                                     HANGUL_ASPECT * self.p.char_width.get(c.ch, 1.0))
            if box is None:
                continue
            a, b, t, u = box
            typical = HANGUL_HEIGHT * em * (self.p.hangul_h - 4 / PX) / self.p.hangul_h
            centre = c.origin[1] * k + 0.06 * em - typical / 2
            centre += self.rng.normal(0, VERTICAL_WANDER * o.messiness * typical)
            _stamp(layer_for(c.color), mask, int(round(x - a)), int(round(centre - (t + u) / 2)))
            gap = LETTER_GAP["hangul"] * o.letter_spacing + self.rng.normal(0, GAP_WANDER * o.messiness)
            x += (b - a + 1) + float(np.clip(gap, -0.07, 0.03)) * HANGUL_HEIGHT * em

    def _reflow(self, chars, plans, sym, body):
        """Horizontal shift (points) per character, packing each line like handwriting.

        Print gives every syllable a full square and spreads formulas out; by hand both are
        written closer.  Syllables of a word and the characters of a formula are packed with
        the widths they are drawn at; a printed space becomes the writer's word gap.  Lines with
        fractions or tall brackets, or crowded by other lines, keep their printed positions so
        equations stay aligned.  Limits and dots written above or below a character move with it."""
        shift = {}
        main = sorted((c for c in chars if c.size >= 0.75 * body and c.ch not in (mathfont.FRACTION, mathfont.OVERDOT)),
                      key=lambda c: c.origin[1])
        rows = []
        for c in main:
            if rows and abs(rows[-1][0] - c.origin[1]) < 0.3 * body:
                rows[-1][1].append(c)
            else:
                rows.append([c.origin[1], [c]])
        bands = [(min(c.bbox[1] for c in r[1]), max(c.bbox[3] for c in r[1])) for r in rows]
        stacked = [c.bbox for c in chars if c.ch == mathfont.FRACTION or (c.ch in DELIMS and c.size > 1.6 * body)]
        small = [c for c in chars if c.size < 0.75 * body and c.ch not in (mathfont.FRACTION,)]
        letter_gap = LETTER_GAP["hangul"] * HANGUL_HEIGHT * self.o.letter_spacing
        k = self.k
        for i, (_, row) in enumerate(rows):
            top, bot = bands[i]
            near = any(j != i and bands[j][0] < bot + 0.3 * body and bands[j][1] > top - 0.3 * body
                       for j in range(len(rows)))
            if near or any(b[1] < bot and b[3] > top for b in stacked):
                continue
            row_top = float(np.median([c.bbox[1] for c in row]))
            row_bot = float(np.median([c.bbox[3] for c in row]))
            riders, attached = [], []
            for c in small:
                mid = (c.bbox[1] + c.bbox[3]) / 2
                if not (top - 0.5 * body <= mid <= bot + 0.5 * body):
                    continue
                if c.bbox[1] > row_bot - 0.15 * body or c.bbox[3] < row_top + 0.15 * body or id(c) not in sym:
                    attached.append(c)   # under/over a character: a limit, a dot
                else:
                    riders.append(c)     # an exponent or index written in the line
            seq = sorted(row + riders, key=lambda c: c.bbox[0])
            units, n = [], 0
            while n < len(seq):  # a Hangul word is one unit, every other character its own
                w = [seq[n]]
                if _is_hangul(seq[n].ch):
                    while (n + len(w) < len(seq) and _is_hangul(seq[n + len(w)].ch)
                           and _same_word(w[-1], seq[n + len(w)])):
                        w.append(seq[n + len(w)])
                units.append(w)
                n += len(w)
            cursor = None
            for u, w in enumerate(units):
                first, last = w[0], w[-1]
                size = first.size
                if _is_hangul(first.ch):
                    width = sum(max(plans[id(x)][0].w - 4 / PX, 0.3) * plans[id(x)][2] * HANGUL_ASPECT
                                * self.p.char_width.get(x.ch, 1.0) / k for x in w)
                    width += letter_gap * sum(x.size for x in w[1:])
                else:
                    width = sym[id(first)][2] if id(first) in sym else (first.bbox[2] - first.bbox[0])
                if cursor is None:
                    start = first.bbox[0]
                else:
                    printed = first.bbox[0] - units[u - 1][-1].bbox[2]
                    if printed > 0.25 * size:  # a space
                        gap = WORD_GAP * HANGUL_HEIGHT * size * self.o.word_spacing
                    else:
                        gap = 0.3 * max(printed, 0.0) + MATH_GAP * size
                    start = cursor + gap
                    if printed > 1.5 * size:
                        # A tab or a second column: keep it where it was printed, so columns
                        # and items set side by side stay lined up.
                        start = max(start, first.bbox[0])
                if _is_hangul(first.ch):
                    for x in w:
                        shift[id(x)] = start - first.bbox[0]
                else:
                    shift[id(first)] = (start + width / 2) - (first.bbox[0] + first.bbox[2]) / 2
                cursor = start + width
            for c in attached:  # follow the character it is written above or below
                cx = (c.bbox[0] + c.bbox[2]) / 2
                base = min(row, key=lambda x: abs((x.bbox[0] + x.bbox[2]) / 2 - cx))
                shift[id(c)] = shift.get(id(base), 0.0)
        return shift

    # ---------------------------------------------------------------- page
    def write(self, page):
        k, o = self.k, self.o
        W, H = int(round(page.rect.width * k)), int(round(page.rect.height * k))
        paper = _paper(o, W, H, o.dpi / 25.4, self.rng)
        chars, lines, pictures = read_page(page)
        chars = _group_pieces(chars)
        sizes = [c.size for c in chars if charsets.is_syllable(c.ch)] or [c.size for c in chars] or [10.0]
        body = float(np.median(sizes))
        self.pen = o.stroke * 0.78 * body * k if o.stroke > 0 else 2.0
        layers = {}

        def layer_for(color):
            key = tuple(round(v, 1) for v in color)
            if key not in layers:
                layers[key] = np.zeros((H, W), np.float32)
            return layers[key]

        original = None
        if pictures or any(c.ch in (mathfont.FRACTION, mathfont.OVERDOT) for c in chars):
            pix = page.get_pixmap(dpi=o.dpi, alpha=False)
            original = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]

        # Each syllable's glyph is chosen up front so the line can be reflowed with its width.
        plans = {id(c): self._glyph(c.ch, HANGUL_HEIGHT * c.size * k, "hangul")
                 for c in chars if _is_hangul(c.ch)}
        sym = {id(c): self._plan_symbol(c) for c in chars
               if not _is_hangul(c.ch) and c.ch not in (mathfont.FRACTION, mathfont.OVERDOT)}
        shift = self._reflow(chars, plans, sym, body)
        word = []  # consecutive Hangul of one word, written closer than print spaces them

        def flush():
            if word:
                self._write_word(word, layer_for, plans, shift)
                word.clear()

        for c in chars:
            dx = shift.get(id(c), 0.0) * k
            x0, y0, x1, y1 = (v * k for v in c.bbox)
            x0, x1 = x0 + dx, x1 + dx
            ox, oy = c.origin[0] * k + dx, c.origin[1] * k
            em = c.size * k
            if _is_hangul(c.ch):
                if word and not _same_word(word[-1], c):
                    flush()
                word.append(c)
                continue
            flush()
            layer = layer_for(c.color)
            angle = self.rng.normal(0, 0.6 * o.messiness)
            if c.ch == mathfont.FRACTION:
                y = _darkest_row(original, x0, y0, x1, y1)
                self._hand_line(layer, x0 + 0.5, y, x1 - 0.5, y, self.pen * 0.9)
                continue
            if c.ch == mathfont.OVERDOT:
                cy = y0 + 0.2 * (y1 - y0)
                cv2.circle(layer, (int((x0 + x1) / 2), int(cy)), max(1, int(self.pen * 0.6)), 1.0, -1, cv2.LINE_AA)
                continue
            cx = (x0 + x1) / 2
            glyph, s, _ = sym[id(c)]
            if c.ch in DELIMS:
                tall = (y1 - y0) > 1.6 * em
                centre_y = (y0 + y1) / 2 if tall else oy - 0.3 * em
                self._draw(layer, glyph, s, angle, lambda a, b, t, u: (cx - (a + b) / 2, centre_y - (t + u) / 2))
            elif c.ch in BIG_OPS or c.ch in WIDE_OPS:
                self._draw(layer, glyph, s, angle, lambda a, b, t, u: (cx - (a + b) / 2, oy - 0.3 * em - (t + u) / 2))
            elif c.ch in ("'", "″"):
                self._draw(layer, glyph, s, angle, lambda a, b, t, u: (cx - (a + b) / 2, oy - 0.72 * em - t))
            else:
                if c.ch in MID_LINE:
                    place = lambda a, b, t, u, em=em: (cx - (a + b) / 2, oy - 0.28 * em - (t + u) / 2)  # noqa: E731
                else:
                    drop = (glyph.top + glyph.h - 2 / PX - (self.p.latin_bottom - 2 / PX)) * s
                    place = lambda a, b, t, u, drop=drop: (cx - (a + b) / 2, oy + drop - u)  # noqa: E731
                self._draw(layer, glyph, s, angle, place)
        flush()

        black = layer_for((0, 0, 0))
        for x0, y0, x1, y1, width in lines:
            self._hand_line(black, x0 * k, y0 * k, x1 * k, y1 * k, max(self.pen * 0.75, width * k))

        for rect in pictures:  # copied as printed, like a figure pasted into a notebook
            a, b, c2, d = (int(round(v * k)) for v in rect)
            region = original[max(0, b):d, max(0, a):c2].astype(np.float32) / 255.0
            paper[max(0, b):d, max(0, a):c2] *= region[:paper[max(0, b):d, max(0, a):c2].shape[0],
                                                       :paper[max(0, b):d, max(0, a):c2].shape[1]]

        for key, alpha in layers.items():
            ink = _pen_colour(key)
            paper *= 1.0 - np.clip(alpha, 0, 1)[..., None] * (1.0 - ink)
        return Image.fromarray((np.clip(paper, 0, 1) * 255).astype(np.uint8))


def _is_hangul(ch):
    return charsets.is_syllable(ch) or charsets.is_jamo(ch)


def _same_word(prev, c):
    return abs(prev.origin[1] - c.origin[1]) < 0.2 * c.size and c.bbox[0] - prev.bbox[2] < 0.25 * c.size


def _pen_colour(color):
    """Printed colours as a pen would show them: black text in dark ink, colours kept deep."""
    c = np.array(color, np.float32)
    if c.max() < 0.25:
        return np.array((0.11, 0.11, 0.13), np.float32)
    return np.clip(c * 0.75, 0, 1)


def _darkest_row(original, x0, y0, x1, y1):
    if original is None:
        return (y0 + y1) / 2
    region = original[int(y0):int(y1) + 1, int(x0):int(x1) + 1].mean(axis=2)
    if region.size == 0:
        return (y0 + y1) / 2
    return y0 + float(np.argmin(region.mean(axis=1)))


def _stamp(layer, mask, x, y):
    H, W = layer.shape
    h, w = mask.shape
    ax, ay = max(0, -x), max(0, -y)
    bx, by = max(0, x), max(0, y)
    ex, ey = min(W, x + w), min(H, y + h)
    if ex <= bx or ey <= by:
        return
    region = layer[by:ey, bx:ex]
    np.maximum(region, mask[ay:ay + ey - by, ax:ax + ex - bx], out=region)


def convert_pdf(pdf_bytes, profile, opts=None, pages=None, seed=None):
    """Handwritten images of ``pages`` (0-based indices; all if None) of a PDF."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    writer = PageWriter(profile, opts, seed)
    indices = range(doc.page_count) if pages is None else pages
    out = [writer.write(doc[i]) for i in indices]
    doc.close()
    return out
