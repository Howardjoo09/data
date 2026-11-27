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

The overall figure weights shape 60 % and layout 40 %.
"""
import itertools

import cv2
import numpy as np
from PIL import Image

from . import charsets, freeform, style
from .render import _thin

SIZE = 64             # characters are compared in a SIZE x SIZE box
TOLERANCE = 0.08      # a stroke point "matches" within this share of the box
SHAPE_WEIGHT = 0.6
# Layout figure -> how far off (absolute, in syllable heights) costs about a third of its score.
LAYOUT = {"width": 0.08, "letter_gap": 0.05, "word_gap": 0.12, "height_spread": 0.03,
          "baseline_spread": 0.03, "top_spread": 0.03, "stroke": 0.012, "char_size": 0.08}


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


def evaluate(profile, images, lines, seeds=(0, 1), extra_own=None):
    """Compare the writer's ``images`` (one per line of ``lines``) with the program's writing
    of the same lines.  ``extra_own``: more of the writer's glyph masks by character, used only
    to measure how much their own copies vary.  Returns a dict of figures."""
    text = "\n".join(lines)
    chars = [c for c in charsets.normalize_text(text) if not c.isspace()]
    yours, your_report = _glyphs(images, text)
    your_style = _layout(images, your_report)

    # How alike the writer's own copies of a character are.
    by_char = {}
    for n, m in yours.items():
        by_char.setdefault(chars[n], []).append(m)
    for ch, masks in (extra_own or {}).items():
        if ch in by_char:
            by_char[ch] = by_char[ch] + list(masks)
    own = []
    for ch, masks in by_char.items():
        for a, b in itertools.islice(itertools.combinations(masks, 2), 6):
            own.append(similarity(a, b))
    own_level = float(np.mean(own)) if len(own) >= 5 else 0.75

    shape, per_char, layouts, renders = [], {}, [], []
    for seed in seeds:
        crops = style.render_lines(profile, lines, seed=seed)
        mine, my_report = _glyphs(crops, text)
        layouts.append(_layout(crops, my_report))
        renders.append(crops)
        for n, m in mine.items():
            if n in yours:
                s = similarity(m, yours[n])
                shape.append(s)
                per_char.setdefault(chars[n], []).append(s)
    shape_raw = float(np.mean(shape)) if shape else 0.0
    shape_score = min(1.0, shape_raw / own_level)

    layout_scores, figures = {}, {}
    for key, tol in LAYOUT.items():
        if key == "char_size":
            err = float(np.nanmean([_char_size_error(m, your_style) for m in layouts]))
            mine_v, your_v, diff = err, 0.0, err
        else:
            mine_v = float(np.nanmean([m[key] for m in layouts]))
            your_v = float(your_style[key])
            diff = abs(mine_v - your_v)
        if np.isnan(diff):
            continue
        layout_scores[key] = float(np.exp(-diff / tol))
        figures[key] = (your_v, mine_v)
    layout_score = float(np.mean(list(layout_scores.values()))) if layout_scores else 0.0
    total = SHAPE_WEIGHT * shape_score + (1 - SHAPE_WEIGHT) * layout_score
    worst = sorted(((float(np.mean(v)), ch) for ch, v in per_char.items()))[:15]
    return {"total": total, "shape": shape_score, "shape_raw": shape_raw, "own_level": own_level,
            "layout": layout_score, "layout_scores": layout_scores, "figures": figures,
            "compared": len(shape), "worst_chars": worst, "renders": renders[0],
            "your_style": your_style}


def side_by_side(images, renders, out_path, height=40):
    """Writer's line, then the program's line, for every line, scaled to one syllable height."""
    rows = []
    for yours, mine in zip(images, renders):
        for img in (yours, mine):
            alpha, _ = freeform._ink(img)
            ys = np.nonzero(alpha.max(axis=1) > 0.3)[0]
            if len(ys) == 0:
                continue
            band = img[max(0, ys.min() - 4):ys.max() + 5]
            hh = max(1, ys.max() - ys.min())
            s = 1.6 * height / hh
            pil = Image.fromarray(band)
            rows.append(pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))), Image.LANCZOS))
    if not rows:
        return
    W = max(r.width for r in rows)
    H = sum(r.height + 6 for r in rows)
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_path)


def report_text(result):
    names = {"width": "글자 폭", "letter_gap": "글자 간격", "word_gap": "띄어쓰기", "height_spread": "크기 고르기",
             "baseline_spread": "아랫줄 맞춤", "top_spread": "윗줄 맞춤", "stroke": "펜 굵기",
             "char_size": "글자별 크기"}
    lines = [f"일치도 {result['total'] * 100:.1f}%  (모양 {result['shape'] * 100:.1f}%, 배치 {result['layout'] * 100:.1f}%)",
             f"  모양: 글자 {result['compared']}개 비교, 평균 {result['shape_raw']:.3f} / 본인끼리 {result['own_level']:.3f}"]
    for key, s in result["layout_scores"].items():
        yours, mine = result["figures"][key]
        lines.append(f"  {names.get(key, key):8s} {s * 100:5.1f}%   본인 {yours:.3f}  프로그램 {mine:.3f}")
    lines.append("  가장 다른 글자: " + " ".join(f"{ch}({s:.2f})" for s, ch in result["worst_chars"]))
    return "\n".join(lines)
