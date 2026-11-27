"""How close is the program's handwriting to the writer's own?  A number, and where it differs.

The writer copies a held-out text by hand (the evaluation page of the copy sheets); the
program writes the same text.  Both go through the same extractor, so every character is
found the same way in both, and they are compared:

* shape -- each character's centre line against the writer's, as the share of stroke that
  lies close to the other's stroke (an F-score after scaling both to the same box).  People
  never write a letter twice the same, so this is divided by how alike the writer's own
  copies of a letter are: 100 % means "as close as your own writing is to itself".
* layout -- letter width, spacing, word gaps, wobble of baseline and tops, pen width and
  per-character size, each scored by how far it is from the writer's value.

A second shape test asks whether a simple classifier, shown the centre lines of the writer's
and the program's characters, can tell whose is whose (cross-validated by character); 50 %
accuracy means it cannot, which scores 100 %.

Some layout figures only count against the program when it is *less* neat than the writer:
the writer asked for even sizes, level tops and bottoms and one pen width (a monoline pen),
neater than quick handwriting.

The overall figure weights shape 60 % and layout 40 %.
"""
import itertools

import cv2
import numpy as np
from PIL import Image

from . import charsets, freeform, render, style
from .render import _thin

SIZE = 64             # characters are compared in a SIZE x SIZE box
TOLERANCE = 0.08      # a stroke point "matches" within this share of the box
SHAPE_WEIGHT = 0.6
# Layout figure -> a just-noticeable difference (in syllable heights; degrees for slant).  A
# figure scores exp(-(difference / jnd)^2): differences at the level of measuring noise cost
# almost nothing, one jnd costs 63 %, two cost 98 %.
LAYOUT = {"width": 0.05, "letter_gap": 0.02, "word_gap": 0.08, "height_spread": 0.03,
          "baseline_spread": 0.03, "top_spread": 0.03, "stroke": 0.008, "stroke_evenness": 0.06,
          "slant": 1.5}
# Dots and ticks have no shape worth comparing (the centre line of a dot is a point).
NO_SHAPE = set(".,'`·:;\"")
NEATER_IS_FINE = {"height_spread", "baseline_spread", "top_spread", "stroke_evenness"}
HOG_SIZE = 48


def _skeleton(mask):
    ys, xs = np.nonzero(mask > 0.3 * mask.max()) if mask.max() > 0 else ([], [])
    if len(ys) == 0:
        return None
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = m.shape
    s = (SIZE - 8) / max(h, w)
    m = cv2.resize(m, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE), np.float32)
    y0, x0 = (SIZE - m.shape[0]) // 2, (SIZE - m.shape[1]) // 2
    canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    binary = (canvas > 0.3 * canvas.max()).astype(np.uint8)
    return _thin(binary) > 0


def similarity(a, b):
    """F-score of two characters' centre lines matching within TOLERANCE (0..1)."""
    sa, sb = _skeleton(a), _skeleton(b)
    if sa is None or sb is None or not sa.any() or not sb.any():
        return 0.0
    da = cv2.distanceTransform((~sa).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 3)
    tol = TOLERANCE * SIZE
    precision = float((db[sa] <= tol).mean())
    recall = float((da[sb] <= tol).mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _hog(img, cell=8, bins=9):
    f = img.astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    b = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * bins).astype(int), bins - 1)
    out = [np.bincount(b[y:y + cell, x:x + cell].ravel(), mag[y:y + cell, x:x + cell].ravel(), bins)
           for y in range(0, f.shape[0], cell) for x in range(0, f.shape[1], cell)]
    out = np.array(out).ravel()
    return out / (np.linalg.norm(out) + 1e-6)


def _centre_line_features(mask):
    sk = _skeleton(mask)
    img = np.zeros((SIZE, SIZE), np.uint8)
    if sk is not None:
        img[sk] = 255
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    return _hog(cv2.resize(img, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA))


def distinguishability(yours, mine, folds=5, lam=100.0):
    """Cross-validated accuracy of a ridge classifier telling the writer's characters from the
    program's (same characters on both sides; folds split by character, so it must learn
    style, not which letter it is)."""
    X, y, group = [], [], []
    for n, m in yours.items():
        for other in mine.get(n, [])[:1]:
            X += [_centre_line_features(m), _centre_line_features(other)]
            y += [1.0, 0.0]
            group += [n, n]
    if len(X) < 40:
        return float("nan")
    X, y, group = np.array(X), np.array(y), np.array(group)
    ids = np.unique(group)
    np.random.default_rng(0).shuffle(ids)
    correct = 0
    for part in np.array_split(ids, folds):
        test = np.isin(group, part)
        mu = X[~test].mean(axis=0)
        A = X[~test] - mu
        w = np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (y[~test] - 0.5))
        correct += int(np.sum((((X[test] - mu) @ w) > 0) == (y[test] > 0.5)))
    return correct / len(y)


def chamfer(a, b):
    """Mean distance between two characters' centre lines (share of the box)."""
    sa, sb = _skeleton(a), _skeleton(b)
    if sa is None or sb is None or not sa.any() or not sb.any():
        return 0.5
    da = cv2.distanceTransform((~sa).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 3)
    return float(db[sa].mean() + da[sb].mean()) / 2 / SIZE


def slant(masks):
    """Mean lean of the near-vertical strokes in degrees (positive: tops lean right)."""
    total, weight = 0.0, 0.0
    for m in masks:
        f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
        gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.degrees(np.arctan(gy / gx))
        ok = (mag > 0.25 * mag.max()) & (np.abs(phi) < 30) & np.isfinite(phi)
        total += float((phi[ok] * mag[ok]).sum())
        weight += float(mag[ok].sum())
    return total / weight if weight else float("nan")


def stroke_evenness(masks):
    """Spread of the pen width along the strokes (std / mean of the local width)."""
    widths = []
    for m in masks:
        binary = (m > 0.5 * max(m.max(), 1e-6)).astype(np.uint8)
        if binary.sum() < 20:
            continue
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        sk = _thin(binary) > 0
        widths.extend((2 * dist[sk]).tolist())
    if len(widths) < 20:
        return float("nan")
    widths = np.array(widths)
    return float(widths.std() / widths.mean())


def _glyphs(images, text):
    glyphs, report = freeform.extract_freeform(images, text, single_line=True, work_width=None)
    masks = {g["n"]: g["mask"].astype(np.float32) / 255 for g in glyphs}
    return masks, report


def _layout(images, report):
    st = style.from_boxes(report["boxes"], report["spaces"])
    st["stroke"] = style.stroke_ratio(images, report["boxes"])
    return st


def _char_size_error(mine, yours):
    common = [c for c in yours["char_height"] if c in mine["char_height"]]
    if not common:
        return float("nan")
    return float(np.mean([abs(np.log(yours["char_height"][c] / mine["char_height"][c])) for c in common]))


def isolated(profile, chars, seed=0):
    """The program's drawing of each character on its own (same choices as in a line)."""
    r = render.Renderer(profile, render.Options(paper="blank", margin_line=False, seed=seed, line_pitch=10,
                                                size=0.5))
    out = {}
    for n, ch in enumerate(chars):
        mask = r.place(ch).mask
        if mask.max() > 0:
            out[n] = mask.astype(np.float32)
    return out


def _blur_to(image, factor):
    """Lose detail as a picture ``factor`` times smaller would (and keep the size)."""
    if factor >= 0.95:
        return image
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def evaluate(profile, images, lines, seeds=(0, 1), extra_own=None, source_px_per_mm=None):
    """Compare the writer's ``images`` (one per line of ``lines``) with the program's writing
    of the same lines.  ``extra_own``: more of the writer's glyph masks by character, used only
    to measure how much their own copies vary.  ``source_px_per_mm``: resolution of the photo
    the writer's lines came from; the program's writing is blurred to the same detail, so a
    sharper picture is not mistaken for a different hand.  Returns a dict of figures."""
    text = "\n".join(lines)
    chars = [c for c in charsets.normalize_text(text) if not c.isspace()]
    yours, your_report = _glyphs(images, text)
    detail = None
    if source_px_per_mm:
        from .copysheet import BAND_PX
        h = _syllable_height(your_report)
        if h:
            detail = h / BAND_PX * source_px_per_mm  # px a syllable had in the writer's photo
    your_style = _layout(images, your_report)
    your_style["slant"] = slant(list(yours.values()))
    your_style["stroke_evenness"] = stroke_evenness(list(yours.values()))

    # How far apart the writer's own copies of a character are.
    by_char = {}
    for n, m in yours.items():
        if chars[n] not in NO_SHAPE:
            by_char.setdefault(chars[n], []).append(m)
    for ch, masks in (extra_own or {}).items():
        if ch in by_char:
            by_char[ch] = by_char[ch] + list(masks)
    own = [chamfer(a, b) for masks in by_char.values()
           for a, b in itertools.islice(itertools.combinations(masks, 2), 6)]
    own_level = float(np.mean(own)) if len(own) >= 5 else float("nan")

    distances, per_char, layouts, renders, mine_all = [], {}, [], [], {}
    for seed in seeds:
        crops = style.render_lines(profile, lines, seed=seed)
        mine = isolated(profile, chars, seed)
        if detail:
            my_h = _syllable_height(_glyphs(crops, text)[1])
            factor = detail / my_h if my_h else 1.0
            crops = [_blur_to(c, factor) for c in crops]
            mine = {n: _blur_to(m, factor) for n, m in mine.items()}
        _, my_report = _glyphs(crops, text)
        # Shapes are compared on characters drawn one at a time: cutting them out of tightly
        # spaced lines would add the cutting errors to the program's side only.
        st = _layout(crops, my_report)
        st["slant"] = slant(list(mine.values()))
        st["stroke_evenness"] = stroke_evenness(list(mine.values()))
        layouts.append(st)
        renders.append((crops, my_report))
        for n, m in mine.items():
            mine_all.setdefault(n, []).append(m)
            if n in yours and chars[n] not in NO_SHAPE:
                d = chamfer(m, yours[n])
                distances.append(d)
                per_char.setdefault(chars[n], []).append(d)
    mine_level = float(np.mean(distances)) if distances else float("nan")
    closeness = min(1.0, own_level / mine_level) if distances and not np.isnan(own_level) else 0.0
    accuracy = distinguishability({n: m for n, m in yours.items() if chars[n] not in NO_SHAPE}, mine_all)
    hidden = 1.0 - 2.0 * max(0.0, accuracy - 0.5) if not np.isnan(accuracy) else closeness
    shape_score = (closeness + hidden) / 2

    layout_scores, figures = {}, {}
    for key, tol in LAYOUT.items():
        mine_v = float(np.nanmean([m[key] for m in layouts]))
        your_v = float(your_style[key])
        diff = abs(mine_v - your_v)
        if key in NEATER_IS_FINE and mine_v <= your_v:
            diff = 0.0
        if np.isnan(diff):
            continue
        layout_scores[key] = float(np.exp(-(diff / tol) ** 2))
        figures[key] = (your_v, mine_v)
    layout_score = float(np.mean(list(layout_scores.values()))) if layout_scores else 0.0
    total = SHAPE_WEIGHT * shape_score + (1 - SHAPE_WEIGHT) * layout_score
    worst = sorted(((float(np.mean(v)), ch) for ch, v in per_char.items()), reverse=True)[:15]
    return {"total": total, "shape": shape_score, "closeness": closeness, "own_level": own_level,
            "mine_level": mine_level, "accuracy": accuracy, "layout": layout_score,
            "layout_scores": layout_scores, "figures": figures, "compared": len(distances),
            "worst_chars": worst, "renders": renders[0][0], "render_report": renders[0][1],
            "your_report": your_report, "your_style": your_style}


def _syllable_height(report):
    hs = [b["box"][3] - b["box"][1] for b in report["boxes"] if charsets.is_syllable(b["char"])]
    return float(np.median(hs)) if hs else None


def side_by_side(images, your_report, renders, render_report, out_path, height=40):
    """Writer's line, then the program's line, for every line, at the same syllable height."""
    rows = []
    for imgs, report in ((images, your_report), (renders, render_report)):
        h = _syllable_height(report) or 40.0
        rows.append([(img, height / h) for img in imgs])
    out = []
    for (yours, s1), (mine, s2) in zip(*rows):
        for img, s in ((yours, s1), (mine, s2)):
            pil = Image.fromarray(img)
            out.append(pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))), Image.LANCZOS))
    if not out:
        return
    W = max(r.width for r in out)
    H = sum(r.height + 6 for r in out)
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in out:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_path)


def report_text(result):
    names = {"width": "글자 폭", "letter_gap": "글자 간격", "word_gap": "띄어쓰기", "height_spread": "크기 고르기",
             "baseline_spread": "아랫줄 맞춤", "top_spread": "윗줄 맞춤", "stroke": "펜 굵기",
             "stroke_evenness": "굵기 변화", "slant": "기울기"}
    lines = [f"일치도 {result['total'] * 100:.1f}%  (모양 {result['shape'] * 100:.1f}%, 배치 {result['layout'] * 100:.1f}%)",
             f"  모양: 글자 {result['compared']}개, 거리 {result['mine_level']:.4f} / 본인끼리 {result['own_level']:.4f}"
             f" -> {result['closeness'] * 100:.1f}%;  구별 정확도 {result['accuracy'] * 100:.1f}% (50%=구별 불가)"]
    for key, s in result["layout_scores"].items():
        yours, mine = result["figures"][key]
        lines.append(f"  {names.get(key, key):8s} {s * 100:5.1f}%   본인 {yours:.3f}  프로그램 {mine:.3f}")
    lines.append("  가장 다른 글자: " + " ".join(f"{ch}({d:.3f})" for d, ch in result["worst_chars"]))
    return "\n".join(lines)
