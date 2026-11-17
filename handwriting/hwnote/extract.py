"""Turn scanned / photographed templates into a handwriting profile (one image per glyph)."""
import io
import json
import os
import secrets
import time

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps

from . import template as T

PX = 10                  # canonical resolution: pixels per millimetre
INSET = 0.7              # mm cut from each side of a writing box (skips the printed border)
ALPHA_LO, ALPHA_HI = 0.24, 0.55   # darkness range mapped to ink opacity 0..1
MIN_INK = 25.0           # summed opacity below which a box counts as empty


class ExtractError(ValueError):
    """Problem the user can fix (bad photo, unknown template ...)."""


# --------------------------------------------------------------------------- input

def load_images(data, filename=""):
    """Decode an uploaded image or PDF into a list of RGB uint8 arrays."""
    if data[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ExtractError(f"PDF를 열 수 없습니다: {filename}") from exc
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=250, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            images.append(arr[:, :, :3].copy())
        doc.close()
        return images
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise ExtractError(f"이미지를 읽을 수 없습니다 (JPG·PNG·PDF만 지원): {filename}") from exc
    return [np.asarray(image)]


# --------------------------------------------------------------------------- page geometry

def _detect_markers(gray):
    dictionary = cv2.aruco.getPredefinedDictionary(T.MARKER_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    best = {}
    for scale in (1.0, 0.5, 0.25):
        if scale < 1.0 and max(gray.shape) * scale < 900:
            break
        img = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        corners, ids, _ = detector.detectMarkers(img)
        found = {}
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if int(i) in T.MARKERS:
                    found[int(i)] = c.reshape(4, 2) / scale
        if len(found) > len(best):
            best = found
        if len(best) == 4:
            break
    return best


def rectify(image):
    """Warp a photo/scan of a template page onto the canonical A4 grid (PX px per mm)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found = _detect_markers(gray)
    if len(found) < 3:
        raise ExtractError("페이지 모서리의 검은 표식(4개)을 찾지 못했습니다. "
                           "종이 전체가 보이도록, 밝고 반듯하게 다시 촬영해 주세요.")
    src = np.concatenate([found[i] for i in sorted(found)]).astype(np.float32)
    dst = np.array([p for i in sorted(found) for p in T.marker_corners_mm(i)], np.float32) * PX
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        raise ExtractError("페이지 기울기를 계산하지 못했습니다. 다시 촬영해 주세요.")
    size = (int(T.PAGE_W * PX), int(T.PAGE_H * PX))
    return cv2.warpPerspective(image, H, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def read_page_id(page):
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY).astype(np.float32)
    strip = gray[int((T.BIT_Y - 2) * PX):int((T.BIT_Y + T.BIT_SIZE + 2) * PX),
                 int((T.BIT_X0 - 2) * PX):int((T.BIT_X0 + T.BIT_COUNT * T.BIT_PITCH + 2) * PX)]
    paper, ink = np.percentile(strip, 95), np.percentile(strip, 3)
    if paper - ink < 40:
        return None
    threshold = (paper + ink) / 2
    bits = []
    for x, y, w, h in T.bit_rects():
        c = 0.25 * w
        cell = gray[int((y + c) * PX):int((y + h - c) * PX), int((x + c) * PX):int((x + w - c) * PX)]
        bits.append(1 if cell.mean() < threshold else 0)
    return T.decode_bits(bits)


def darkness(page):
    """Per-pixel ink darkness 0..1 with uneven lighting divided out."""
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    background = cv2.GaussianBlur(background, (0, 0), 12)
    return np.clip(1.0 - gray / np.maximum(background, 1.0), 0.0, 1.0)


# --------------------------------------------------------------------------- glyphs

def _clean(alpha):
    """Remove dust, printed border remnants and dotted-baseline remnants from a box crop."""
    mask = (alpha > 0.3).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = alpha.shape
    base = int(round((T.BASELINE - INSET) * PX))
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        border_line = touches and ((bh <= 8 and bw >= 25) or (bw <= 8 and bh >= 25))
        guide_dash = y >= base - 4 and y + bh <= base + 5 and area < 80 and bh <= 6
        if area < 10 or border_line or guide_dash:
            alpha[labels == i] = 0
    return alpha


def extract_page(page, chars):
    """Cut every written box out of a rectified page. Returns (glyphs, empty_chars, ink_rgb)."""
    dark = darkness(page)
    alpha_page = np.clip((dark - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0)
    glyphs, empty, ink_samples = [], [], []
    for index, ch in enumerate(chars):
        bx, by = T.cell_box(index)
        x0, y0 = int(round((bx + INSET) * PX)), int(round((by + INSET) * PX))
        x1, y1 = int(round((bx + T.BOX - INSET) * PX)), int(round((by + T.BOX - INSET) * PX))
        alpha = _clean(alpha_page[y0:y1, x0:x1].copy())
        if alpha.sum() < MIN_INK:
            empty.append(ch)
            continue
        ys, xs = np.nonzero(alpha > 0.08)
        pad = 2
        top, bottom = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
        left, right = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
        crop = alpha[top:bottom, left:right]
        strong = alpha > 0.85
        if strong.sum() > 20:
            ink_samples.append(np.median(page[y0:y1, x0:x1][strong], axis=0))
        glyphs.append({
            "char": ch,
            "mask": (crop * 255).astype(np.uint8),
            # Position of the cropped mask inside the writing box, in millimetres.
            "left": float(left / PX + INSET), "top": float(top / PX + INSET),
            "w": crop.shape[1] / PX, "h": crop.shape[0] / PX,
        })
    ink = None
    if ink_samples:
        paper = np.percentile(page.reshape(-1, 3), 90, axis=0)
        ink = np.clip(np.median(ink_samples, axis=0) / np.maximum(paper, 1), 0, 1).tolist()
    return glyphs, empty, ink


# --------------------------------------------------------------------------- profiles

def profile_dir(data_dir, profile_id):
    return os.path.join(data_dir, "profiles", profile_id)


def build_profile(data_dir, uploads, template_hint=None, profile_id=None):
    """Create (or extend) a profile from uploaded files ``[(bytes, filename), ...]``."""
    pages, warnings = [], []
    for data, name in uploads:
        for n, image in enumerate(load_images(data, name)):
            label = f"{name} {n + 1}쪽" if len(uploads) > 1 or n else name
            try:
                page = rectify(image)
            except ExtractError as exc:
                warnings.append(f"{label}: {exc}")
                continue
            page_id = read_page_id(page)
            if page_id is None and template_hint:
                page_id = (int(template_hint), len(pages))
                warnings.append(f"{label}: 페이지 코드를 읽지 못해 입력한 템플릿 ID와 업로드 순서를 사용했습니다.")
            if page_id is None:
                warnings.append(f"{label}: 상단의 페이지 코드를 읽지 못했습니다. 템플릿 ID를 직접 입력해 보세요.")
                continue
            pages.append((label, page, page_id))
    resolved = []
    for label, page, (template_id, page_no) in pages:
        info = T.load_template(data_dir, template_id)
        if info is None or page_no >= len(info["pages"]):
            warnings.append(f"{label}: 템플릿 {template_id} {page_no + 1}쪽 정보를 찾지 못했습니다 "
                            "(이 서버에서 만든 템플릿만 인식할 수 있습니다).")
            continue
        resolved.append((page, info["pages"][page_no]))
    if not resolved:
        raise ExtractError("처리할 수 있는 템플릿 페이지가 없습니다.\n" + "\n".join(warnings))

    if profile_id:
        directory = profile_dir(data_dir, profile_id)
        if not os.path.exists(os.path.join(directory, "profile.json")):
            raise ExtractError("이어서 추가할 글씨체 프로필을 찾지 못했습니다.")
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as f:
            profile = json.load(f)
    else:
        profile_id = secrets.token_hex(5)
        directory = profile_dir(data_dir, profile_id)
        os.makedirs(directory)
        profile = {"id": profile_id, "created": time.time(), "box": T.BOX, "baseline": T.BASELINE,
                   "glyphs": [], "ink": None}

    empty_all, inks = [], []
    for page, chars in resolved:
        glyphs, empty, ink = extract_page(page, chars)
        empty_all.extend(empty)
        if ink:
            inks.append(ink)
        for g in glyphs:
            name = f"g{len(profile['glyphs']):04d}.png"
            cv2.imwrite(os.path.join(directory, name), g.pop("mask"))
            g["file"] = name
            profile["glyphs"].append(g)
    if inks:
        profile["ink"] = np.median(inks, axis=0).tolist()
    with open(os.path.join(directory, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    if not profile["glyphs"]:
        warnings.append("글자를 하나도 찾지 못했습니다. 펜이 너무 연하거나 칸이 비어 있지 않은지 확인해 주세요.")
    return profile, empty_all, warnings
