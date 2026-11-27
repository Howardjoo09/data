"""Printable handwriting template: page geometry, PDF generation and page-ID encoding.

All geometry is in millimetres on an A4 page.  Four ArUco markers in the corners let the
extractor undo the perspective of a phone photo or scan; a strip of black squares between
the top markers carries the template ID and page number.
"""
import json
import os
import secrets
import time

import cv2
import pymupdf

from .charsets import LABEL_DISPLAY, LABEL_HINTS

PAGE_W, PAGE_H = 210.0, 297.0
MARKER_SIZE = 12.0
MARKER_DICT = cv2.aruco.DICT_4X4_50
# Top-left corner of each marker: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
MARKERS = {0: (8.0, 8.0), 1: (190.0, 8.0), 2: (190.0, 277.0), 3: (8.0, 277.0)}

BIT_COUNT = 42           # sync bit + 24-bit template id + 8-bit page + 8-bit checksum + sync bit
BIT_X0, BIT_Y, BIT_PITCH, BIT_SIZE = 30.0, 12.5, 3.4, 2.8

COLS, ROWS = 10, 11
BOX = 16.0               # writing box (square)
COL_PITCH, ROW_PITCH = 17.0, 20.5
GRID_X0, GRID_Y0 = 20.5, 36.0
LABEL_H = 4.5            # printed label strip above every box
BASELINE = 11.5          # dotted baseline, measured from the top of the box
CELLS_PER_PAGE = COLS * ROWS

# Copy sheets (layout 2): a printed line to copy, then two squares of note-book grid to write
# it in, as in the writer's notes.  Rows sit on the same 4.24 mm grid the renderer draws.
GRID_STEP = 4.24
COPY_X0, COPY_X1 = 17.0, 193.0          # grid area, mm
COPY_FIRST = 9                          # first row starts on grid line 9 (38.2 mm)
COPY_ROWS = 18
COPY_ROW = 3                            # grid squares per row: 1 printed + 2 to write in
COPY_TEXT_PT = 8.5
COPY_WRAP = 125.0                       # printed lines are at most this long (mm)
GRID_PAPER = (251 / 255, 249 / 255, 237 / 255)
GRID_LINE = (224 / 255, 223 / 255, 213 / 255)

INK_GRAY = (0.45, 0.45, 0.45)
BORDER = (0.62, 0.72, 0.88)
GUIDE = (0.78, 0.86, 1.0)
PT = 72.0 / 25.4         # points per millimetre


def copy_row(index):
    """(top of the printed strip, top of the writing band, bottom of the band) in mm."""
    top = (COPY_FIRST + index * COPY_ROW) * GRID_STEP
    return top, top + GRID_STEP, top + COPY_ROW * GRID_STEP


def parse_markup(line):
    """Split a copy-sheet line into ``[(char, kind)]``; ``^x`` is a superscript and ``_x`` a
    subscript character, everything else is ``"normal"``."""
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c in "^_" and i + 1 < len(line):
            out.append((line[i + 1], "sup" if c == "^" else "sub"))
            i += 2
        else:
            out.append((c, "normal"))
            i += 1
    return out


def plain_text(line):
    return "".join(c for c, _ in parse_markup(line))


def cell_box(index):
    """Top-left (x, y) in mm of the writing box for the ``index``-th cell on a page."""
    row, col = divmod(index, COLS)
    return GRID_X0 + col * COL_PITCH, GRID_Y0 + row * ROW_PITCH + LABEL_H


def marker_corners_mm(marker_id):
    x, y = MARKERS[marker_id]
    s = MARKER_SIZE
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


def _crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_bits(template_id, page_no):
    payload = (template_id & 0xFFFFFF) << 8 | (page_no & 0xFF)
    raw = payload.to_bytes(4, "big")
    value = payload << 8 | _crc8(raw)
    return [1] + [(value >> (39 - i)) & 1 for i in range(40)] + [1]


def decode_bits(bits):
    """Inverse of :func:`encode_bits`; returns ``(template_id, page_no)`` or ``None``."""
    if len(bits) != BIT_COUNT or bits[0] != 1 or bits[-1] != 1:
        return None
    value = 0
    for b in bits[1:-1]:
        value = value << 1 | b
    payload, crc = value >> 8, value & 0xFF
    if _crc8(payload.to_bytes(4, "big")) != crc:
        return None
    return payload >> 8, payload & 0xFF


def bit_rects():
    return [(BIT_X0 + i * BIT_PITCH, BIT_Y, BIT_SIZE, BIT_SIZE) for i in range(BIT_COUNT)]


# --------------------------------------------------------------------------- storage

def new_template(data_dir, chars):
    os.makedirs(os.path.join(data_dir, "templates"), exist_ok=True)
    while True:
        template_id = secrets.randbelow(0xFFFFFF - 1) + 1
        path = template_path(data_dir, template_id)
        if not os.path.exists(path):
            break
    pages = [chars[i:i + CELLS_PER_PAGE] for i in range(0, len(chars), CELLS_PER_PAGE)] or [[]]
    info = {"id": template_id, "layout": 1, "created": time.time(), "pages": pages}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    return info


def template_path(data_dir, template_id):
    return os.path.join(data_dir, "templates", f"{int(template_id)}.json")


def load_template(data_dir, template_id):
    path = template_path(data_dir, template_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- PDF

def _rect(x, y, w, h):
    return pymupdf.Rect(x * PT, y * PT, (x + w) * PT, (y + h) * PT)


def _markers(shape, template_id, page_no):
    """Corner markers (drawn module by module, 6x6 with the black border) and the page code."""
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICT)
    for marker_id, (mx, my) in MARKERS.items():
        modules = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
        m = MARKER_SIZE / 6
        for r in range(6):
            for c in range(6):
                if modules[r, c] < 128:
                    shape.draw_rect(_rect(mx + c * m, my + r * m, m, m))
    for bit, (x, y, w, h) in zip(encode_bits(template_id, page_no), bit_rects()):
        if bit:
            shape.draw_rect(_rect(x, y, w, h))
    shape.finish(color=None, fill=(0, 0, 0), width=0)


def build_pdf(info):
    """Return the template PDF bytes for a stored template ``info``."""
    if info.get("layout") == 2:
        return _build_copy_pdf(info)
    font = pymupdf.Font("korea")
    doc = pymupdf.open()
    total = len(info["pages"])
    for page_no, chars in enumerate(info["pages"]):
        page = doc.new_page(width=PAGE_W * PT, height=PAGE_H * PT)
        page.insert_font(fontname="kr", fontbuffer=font.buffer)
        shape = page.new_shape()
        _markers(shape, info["id"], page_no)
        # Writing boxes.
        for index in range(len(chars)):
            x, y = cell_box(index)
            shape.draw_rect(_rect(x, y, BOX, BOX))
        shape.finish(color=BORDER, width=0.6, fill=None)
        for index in range(len(chars)):
            x, y = cell_box(index)
            shape.draw_line(pymupdf.Point((x + 1) * PT, (y + BASELINE) * PT),
                            pymupdf.Point((x + BOX - 1) * PT, (y + BASELINE) * PT))
        shape.finish(color=GUIDE, width=0.7, dashes="[2 2] 0")
        shape.commit()

        def text(x, y, s, size, color=INK_GRAY, center_width=None):
            if center_width is not None:
                x += (center_width - font.text_length(s, fontsize=size) / PT) / 2
            page.insert_text(pymupdf.Point(x * PT, y * PT), s, fontsize=size, fontname="kr", color=color)

        text(30, 25.5, f"손글씨 템플릿   ID {info['id']}   {page_no + 1}/{total}쪽", 10, (0.15, 0.15, 0.15))
        text(30, 30.5, "칸마다 위 글자를 평소 필기하듯 쓰세요. 한글은 칸 가운데에 평소 크기로, 영어·숫자는 점선을 "
                       "바닥선으로 삼아 쓰고, 기호는 편하게 쓰면 됩니다.", 6.5)
        text(30, 33.5, "같은 글자가 다시 나오면 매번 새로 쓰세요(복사·붙여넣기 금지). 칸 밖으로 나가지 않게, 모서리의 "
                       "검은 사각형은 가리지 마세요.", 6.5)
        for index, ch in enumerate(chars):
            x, y = cell_box(index)
            hint = LABEL_HINTS.get(ch)
            shown = LABEL_DISPLAY.get(ch, ch)
            if hint:
                text(x + 1, y - 1.2, shown, 9)
                text(x + 1.5 + font.text_length(shown, fontsize=9) / PT + 1, y - 1.4, hint, 5.5)
            else:
                text(x, y - 1.2, shown, 9, center_width=BOX)
    doc.subset_fonts()
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def wrap_copy_text(paragraphs, width=COPY_WRAP, size=COPY_TEXT_PT):
    """Break paragraphs into printed lines at most ``width`` mm long (at word boundaries)."""
    font = pymupdf.Font("korea")
    measure = lambda t: font.text_length(plain_text(t), fontsize=size) / PT  # noqa: E731
    lines = []
    for para in paragraphs:
        current = ""
        for word in para.split(" "):
            trial = f"{current} {word}" if current else word
            if current and measure(trial) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
    return lines


def _build_copy_pdf(info):
    font = pymupdf.Font("korea")
    doc = pymupdf.open()
    total = len(info["pages"])
    evaluation = set(info.get("eval_pages", []))
    for page_no, lines in enumerate(info["pages"]):
        page = doc.new_page(width=PAGE_W * PT, height=PAGE_H * PT)
        page.insert_font(fontname="kr", fontbuffer=font.buffer)
        shape = page.new_shape()
        _markers(shape, info["id"], page_no)
        first_top = copy_row(0)[0]
        last_bottom = copy_row(COPY_ROWS - 1)[2]
        shape.draw_rect(_rect(COPY_X0, first_top, COPY_X1 - COPY_X0, last_bottom - first_top))
        shape.finish(color=None, fill=GRID_PAPER, width=0)
        for row in range(COPY_ROWS):
            top, band, bottom = copy_row(row)
            for k in range(COPY_ROW):  # horizontal grid lines of the band (and its bottom)
                y = band + k * GRID_STEP
                shape.draw_line(pymupdf.Point(COPY_X0 * PT, y * PT), pymupdf.Point(COPY_X1 * PT, y * PT))
            x = GRID_STEP * int(COPY_X0 / GRID_STEP + 1)
            while x < COPY_X1:
                shape.draw_line(pymupdf.Point(x * PT, band * PT), pymupdf.Point(x * PT, bottom * PT))
                x += GRID_STEP
        shape.finish(color=GRID_LINE, width=0.5)
        for row in range(COPY_ROWS):  # printed strips are plain paper
            top, band, _ = copy_row(row)
            shape.draw_rect(_rect(COPY_X0, top + 0.15, COPY_X1 - COPY_X0, GRID_STEP - 0.3))
        shape.finish(color=None, fill=(1, 1, 1), width=0)
        shape.commit()

        def text(x, y, s, size, color=INK_GRAY):
            page.insert_text(pymupdf.Point(x * PT, y * PT), s, fontsize=size, fontname="kr", color=color)

        title = "평가용 필사 시트" if page_no in evaluation else "필사 시트"
        text(30, 25.5, f"{title}   ID {info['id']}   {page_no + 1}/{total}쪽", 10, (0.15, 0.15, 0.15))
        text(30, 30.5, "회색 글을 바로 아래 모눈 두 칸에 평소 노트 필기하듯 옮겨 쓰세요. 크기·속도·글씨체 모두 "
                       "평소 그대로, 한 줄을 한 줄에 쓰면 됩니다.", 6.5)
        text(30, 33.5, "틀린 글자는 지우지 말고 그냥 두세요(그 단어만 학습에서 빠집니다). 모서리의 검은 "
                       "사각형은 가리지 마세요.", 6.5)
        for row, line in enumerate(lines[:COPY_ROWS]):
            top, band, _ = copy_row(row)
            x, base = COPY_X0 + 2.0, band - 1.0
            for ch, kind in parse_markup(line):
                size = COPY_TEXT_PT * (0.62 if kind != "normal" else 1.0)
                dy = -1.3 if kind == "sup" else 0.7 if kind == "sub" else 0.0
                text(x, base + dy, ch, size, (0.35, 0.35, 0.35))
                x += font.text_length(ch, fontsize=size) / PT
    doc.subset_fonts()
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def render_page_image(pdf_bytes, page_no, px_per_mm=10):
    """Rasterise one template page (used by the self-test and demo sample generator)."""
    import numpy as np
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    zoom = px_per_mm / PT
    pix = doc[page_no].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    doc.close()
    return image
