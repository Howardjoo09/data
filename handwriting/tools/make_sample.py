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
