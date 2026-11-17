"""Fill a template with a handwriting-style TTF and fake a phone photo of it.

Used by the self-test and to produce a demo sample when no real handwriting is at hand:

    python tools/make_sample.py --font NanumPen.ttf --out sample_dir
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hwnote import charsets, template as T  # noqa: E402

SCALE = 12  # px per mm of the simulated print


def fill_page(pdf_bytes, page_no, chars, font_path, seed=0, ink=(25, 35, 105), skip=0.02):
    rng = np.random.default_rng(seed + page_no)
    page = Image.fromarray(T.render_page_image(pdf_bytes, page_no, SCALE))
    draw = ImageDraw.Draw(page)
    fonts = {}

    def font(size_mm):
        key = round(size_mm * SCALE)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(font_path, key)
        return fonts[key]

    def em_for(sample, height_mm):
        """Font size (mm) at which ``sample`` is ``height_mm`` tall, so any TTF fits the boxes."""
        box = font(10).getbbox(sample)
        return 10 * height_mm * SCALE / max(1, box[3] - box[1])

    hangul_em, latin_em = em_for("가", 10.0), em_for("H", 8.0)
    for index, ch in enumerate(chars):
        if rng.random() < skip:  # people skip the odd box
            continue
        bx, by = T.cell_box(index)
        dx, dy = rng.normal(0, 0.5, 2)
        cx = (bx + T.BOX / 2 + dx) * SCALE
        if charsets.is_syllable(ch) or charsets.is_jamo(ch):
            size = hangul_em * rng.uniform(0.93, 1.07)
            draw.text((cx, (by + T.BOX / 2 + dy) * SCALE), ch, font=font(size), fill=ink, anchor="mm")
        else:
            size = latin_em * rng.uniform(0.94, 1.06)
            draw.text((cx, (by + T.BASELINE + dy * 0.4) * SCALE), ch, font=font(size), fill=ink, anchor="ms")
    return np.asarray(page)


def fake_photo(page, seed=0):
    """Perspective, uneven light, noise and JPEG artefacts like a quick phone snapshot."""
    rng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    out_w, out_h = int(w * 1.25), int(h * 1.18)
    margin = np.array([out_w - w, out_h - h]) / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.05, 0.05, (4, 2)) * [w, h]
    dst = np.float32(src + margin + jitter)
    H = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(page, H, (out_w, out_h), borderValue=(92, 84, 76)).astype(np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    light = 0.78 + 0.22 * (xx / out_w) * (1 - 0.35 * yy / out_h)
    photo = photo * light[..., None] + rng.normal(0, 4, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (0, 0), 0.8)
    if rng.random() < 0.5:
        photo = cv2.rotate(photo, cv2.ROTATE_180)  # upside-down shots must work too
    photo = cv2.resize(photo, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", photo[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


# --------------------------------------------------------------------------- free-form samples

def freeform_page(text, font_path, seed=0, ink=(30, 30, 40), width_mm=170, pitch_mm=8.0, hangul_mm=5.2,
                  gap=(0.04, 0.16)):
    """Write ``text`` on lined paper like ordinary notes.

    Returns ``(rgb image, truth)`` where ``truth`` lists ``(char, x0, y0, x1, y1)`` for every
    non-space character, so tests can check where the extractor found each one.
    """
    rng = np.random.default_rng(seed)
    margin = 15 * SCALE
    hangul_em = latin_em = None
    fonts = {}

    def font(size_px):
        key = int(round(size_px))
        if key not in fonts:
            fonts[key] = ImageFont.truetype(font_path, key)
        return fonts[key]

    box = font(100).getbbox("가")
    hangul_em = 100 * hangul_mm * SCALE / (box[3] - box[1])
    box = font(100).getbbox("H")
    latin_em = 100 * hangul_mm * 0.85 * SCALE / (box[3] - box[1])
    h_px = hangul_mm * SCALE

    # Lay the text out first so the page can be sized to fit.
    placements, x, line = [], 0.0, 0
    limit = width_mm * SCALE
    for paragraph in charsets.normalize_text(text).split("\n"):
        for word in paragraph.split(" "):
            sizes = [hangul_em if (charsets.is_syllable(c) or charsets.is_jamo(c)) else latin_em for c in word]
            sizes = [s * rng.uniform(0.92, 1.08) for s in sizes]
            widths = [font(s).getlength(c) for c, s in zip(word, sizes)]
            gaps = [h_px * rng.uniform(*gap) for _ in word]
            if x > 0 and x + sum(widths) + sum(gaps) > limit:
                x, line = 0.0, line + 1
            for c, s, w, g in zip(word, sizes, widths, gaps):
                placements.append((c, s, x, line))
                x += w + g
            x += h_px * rng.uniform(0.35, 0.55)
        x, line = 0.0, line + 1
    lines = line
    height = int(2 * margin + (lines + 1) * pitch_mm * SCALE)
    page = Image.new("RGB", (int(limit + 2 * margin), height), (250, 250, 246))
    draw = ImageDraw.Draw(page)
    for i in range(lines + 2):
        y = margin + i * pitch_mm * SCALE
        draw.line((0, y, page.width, y), fill=(170, 200, 228), width=2)
    truth = []
    drift = [rng.normal(0, 0.25 * SCALE) for _ in range(lines + 1)]
    for c, s, x0, ln in placements:
        base = margin + (ln + 1) * pitch_mm * SCALE - 1.0 * SCALE + drift[ln] + rng.normal(0, 0.12 * SCALE)
        pos = (margin + x0, base)
        draw.text(pos, c, font=font(s), fill=ink, anchor="ls")
        bx = draw.textbbox(pos, c, font=font(s), anchor="ls")
        truth.append((c, *bx))
    return np.asarray(page), truth


def freeform_photo(page, truth, seed=0, angle=None):
    """Slight rotation, uneven light, noise and JPEG; the truth boxes are carried along."""
    rng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    angle = rng.uniform(-2.5, 2.5) if angle is None else angle
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    photo = cv2.warpAffine(page, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.8 + 0.2 * (xx / w) * (1 - 0.3 * yy / h)
    photo = photo * light[..., None] + rng.normal(0, 3.5, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (0, 0), 0.7)
    scale = 0.7
    photo = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    moved = []
    for c, x0, y0, x1, y1 in truth:
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], np.float32) @ M.T * scale
        moved.append((c, *corners.min(0), *corners.max(0)))
    ok, jpeg = cv2.imencode(".jpg", photo[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes(), moved


def make(data_dir, font_path, set_keys, custom_text="", seed=0, skip=0.02):
    """Create a template in ``data_dir`` and return ``(info, [jpeg bytes per page])``."""
    info = T.new_template(data_dir, charsets.build_charlist(set_keys, custom_text))
    pdf = T.build_pdf(info)
    photos = [fake_photo(fill_page(pdf, n, chars, font_path, seed, skip=skip), seed + n)
              for n, chars in enumerate(info["pages"])]
    return info, photos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    parser.add_argument("--sets", default="upper,lower,digits,punct,jamo,hangul_basic")
    parser.add_argument("--text", default="")
    parser.add_argument("--skip", type=float, default=0.02, help="fraction of boxes left empty")
    parser.add_argument("--bundle", action="store_true", help="also copy the template JSON next to the photos")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    info, photos = make(args.data, args.font, args.sets.split(","), args.text, skip=args.skip)
    if args.bundle:
        with open(os.path.join(args.out, f"template_{info['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
    for n, jpeg in enumerate(photos):
        Image.open(io.BytesIO(jpeg)).save(os.path.join(args.out, f"demo_page{n + 1}.jpg"), quality=80)
    print(f"template {info['id']}: {len(photos)} page photo(s) written to {args.out}")


if __name__ == "__main__":
    main()
