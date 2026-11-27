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
MIN_INK_SMALL = 5.0      # ... for characters that are only a dot or two
SMALL_CHARS = set(".,:;'`\"·")


class ExtractError(ValueError):
    """Problem the user can fix (bad photo, unknown template ...)."""


# --------------------------------------------------------------------------- input

def iter_images(data, filename="", dpi=250):
    """Decode an uploaded image or PDF into RGB uint8 arrays, one page at a time."""
    if data[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ExtractError(f"PDF를 열 수 없습니다: {filename}") from exc
        try:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                yield arr[:, :, :3].copy()
        finally:
            doc.close()
        return
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise ExtractError(f"이미지를 읽을 수 없습니다 (JPG·PNG·PDF만 지원): {filename}") from exc
    yield np.asarray(image)


def load_images(data, filename=""):
    """Decode an uploaded image or PDF into a list of RGB uint8 arrays."""
    return list(iter_images(data, filename))


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


def rectify(image, with_homography=False):
    """Warp a photo/scan of a template page onto the canonical A4 grid (PX px per mm); with
    ``with_homography`` also return the image-to-page matrix (for sharper crops)."""
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
    page = cv2.warpPerspective(image, H, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return (page, H) if with_homography else page


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
        piece = labels == i
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        border_line = touches and ((bh <= 8 and bw >= 25) or (bw <= 8 and bh >= 25))
        # Remnants of the dotted baseline are flat dashes; a full stop on the baseline is round.
        guide_dash = (y >= base - 4 and y + bh <= base + 5 and area < 80 and bh <= 6
                      and bw >= 2.5 * bh)
        # Dust is tiny or faint; the dot of a fine pen (. : ! i) is small but dark.
        dust = area < 4 or (area < 12 and alpha[piece].mean() < 0.55)
        if dust or border_line or guide_dash:
            alpha[piece] = 0
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
        if alpha.sum() < (MIN_INK_SMALL if ch in SMALL_CHARS else MIN_INK):
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


def build_profile(data_dir, uploads, template_hint=None, profile_id=None, transcript="", new_id=None):
    """Create (or extend) a profile from uploaded files ``[(bytes, filename), ...]``.

    Pages showing the corner markers are read as templates.  Any other page is read as
    free-form handwriting, which needs ``transcript`` (the text written on those pages, in
    order).  Returns ``(profile, empty_template_boxes, warnings)``; ``profile["report"]``
    describes the free-form part.
    """
    from . import copysheet, freeform  # imported here: they build on this module

    pages, warnings, loose, copies, evaluation = [], [], [], [], []
    for data, name in uploads:
        # Copy sheets are written at note size: render PDFs sharply enough for small letters.
        for n, image in enumerate(iter_images(data, name, dpi=400)):
            label = f"{name} {n + 1}쪽" if len(uploads) > 1 or n else name
            try:
                page, H = rectify(image, with_homography=True)
            except ExtractError as exc:
                if transcript.strip():
                    scale = min(1.0, 3000 / image.shape[1])
                    loose.append(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                                 if scale < 1 else image)
                else:
                    warnings.append(f"{label}: 템플릿 표식이 없습니다. 자유롭게 쓴 손글씨라면 "
                                    "사진에 적힌 내용을 함께 입력해 주세요.")
                continue
            page_id = read_page_id(page)
            if page_id is None and template_hint:
                page_id = (int(template_hint), len(pages) + len(copies) + len(evaluation))
                warnings.append(f"{label}: 페이지 코드를 읽지 못해 입력한 템플릿 ID와 업로드 순서를 사용했습니다.")
            if page_id is None:
                warnings.append(f"{label}: 상단의 페이지 코드를 읽지 못했습니다. 템플릿 ID를 직접 입력해 보세요.")
                continue
            template_id, page_no = page_id
            info = T.load_template(data_dir, template_id)
            if info is None or page_no >= len(info["pages"]):
                warnings.append(f"{label}: 템플릿 {template_id} {page_no + 1}쪽 정보를 찾지 못했습니다 "
                                "(이 서버에서 만든 템플릿만 인식할 수 있습니다).")
                continue
            if info.get("layout") == 2:  # copy sheet: keep only the sharp band crops
                lines = info["pages"][page_no]
                target = evaluation if page_no in info.get("eval_pages", []) else copies
                target.append((copysheet.bands(image, H, len(lines)), lines))
            else:
                pages.append((page, info["pages"][page_no]))
    resolved = pages
    copy_glyphs, copy_report = [], None
    if copies:
        copy_glyphs, copy_report, _, copy_lines = copysheet.read_pages(copies)
        copy_lines = [T.plain_text(line) for line in copy_lines]
        if copy_report["chars"] and copy_report["learned"] < 0.4 * copy_report["chars"]:
            warnings.append(f"필사 시트: {copy_report['chars']}자 중 {copy_report['learned']}자만 확실하게 "
                            "찾았습니다. 줄을 건너뛰거나 다른 줄에 쓰지 않았는지 확인해 주세요.")
    report = None
    loose_glyphs = []
    if loose:
        loose_glyphs, report = freeform.extract_freeform(loose, transcript)
        report.pop("positions", None)
        report["ink"] = freeform.ink_color(loose)
        if report["learned"] < 0.5 * report["chars"]:
            warnings.append(f"자유 양식: 입력한 내용 {report['chars']}자 중 {report['learned']}자만 확실하게 "
                            "찾았습니다. 사진 속 글과 입력한 내용이 같은지(빠진 줄, 오타) 확인해 주세요.")
    if not resolved and not loose_glyphs and not copy_glyphs and not evaluation:
        raise ExtractError("글씨를 학습할 수 있는 페이지가 없습니다.\n" + "\n".join(warnings))

    if profile_id:
        directory = profile_dir(data_dir, profile_id)
        if not os.path.exists(os.path.join(directory, "profile.json")):
            raise ExtractError("이어서 추가할 글씨체 프로필을 찾지 못했습니다.")
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as f:
            profile = json.load(f)
    else:
        profile_id = new_id or secrets.token_hex(5)
        directory = profile_dir(data_dir, profile_id)
        os.makedirs(directory, exist_ok=True)
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
    for source, items in (("freeform", loose_glyphs), ("copy", copy_glyphs)):
        for g in items:
            name = f"g{len(profile['glyphs']):04d}.png"
            cv2.imwrite(os.path.join(directory, name), g.pop("mask"))
            g["file"] = name
            g["source"] = source
            g.pop("n", None)
            profile["glyphs"].append(g)
    if copy_report:
        profile["copy_report"] = {k: copy_report[k] for k in ("chars", "learned", "lines")}
        profile["copy_style"] = copy_report.get("style")
        profile["copy_lines"] = copy_lines[:24]
    if evaluation:
        kept = copysheet.save_evaluation(directory, evaluation)
        warnings.append(f"평가용 필사 시트 {kept}줄은 학습하지 않고 비교용으로 보관했습니다.")
    if report and report.get("ink"):
        inks.append(report["ink"])
    if inks:
        profile["ink"] = np.median(inks, axis=0).tolist()
    profile["report"] = report
    with open(os.path.join(directory, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    if not profile["glyphs"]:
        warnings.append("글자를 하나도 찾지 못했습니다. 펜이 너무 연하거나 칸이 비어 있지 않은지 확인해 주세요.")
    return profile, empty_all, warnings


def set_excluded(directory, files, excluded=True):
    """Stop (or resume) using particular glyph copies of a profile, e.g. an i written with its
    dot touching the stem.  The images are kept, so this can be undone."""
    path = os.path.join(directory, "profile.json")
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    known = {g["file"] for g in profile["glyphs"]}
    unknown = [name for name in files if name not in known]
    if unknown:
        raise ExtractError("모르는 글자 이미지입니다: " + ", ".join(unknown))
    current = set(profile.get("excluded", []))
    current = current | set(files) if excluded else current - set(files)
    profile["excluded"] = sorted(current)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    return profile["excluded"]


def set_letter_style(directory, ch, style):
    """Draw ``ch`` in another way built from the writer's own letters (e.g. ``"round"`` for a
    y made of their u and the tail of their g); ``None`` goes back to the template copies."""
    path = os.path.join(directory, "profile.json")
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    styles = profile.get("letter_styles", {})
    if style:
        styles[ch] = style
    else:
        styles.pop(ch, None)
    profile["letter_styles"] = styles
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    return styles
