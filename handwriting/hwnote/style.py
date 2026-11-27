"""Measure how someone writes -- size, spacing, alignment -- from handwriting of known text.

The same measurement is applied to the writer's real handwriting and to the program's
output, so the two can be compared number by number and the output tuned to match.
"""
import cv2
import numpy as np

from . import charsets, freeform


def measure(images, transcript):
    """Style figures of handwriting ``images`` whose text is ``transcript``.

    Lengths are relative to the median Hangul syllable height of the same line, so writing
    of any size or resolution can be compared.
    """
    _, report = freeform.extract_freeform(images, transcript)
    return from_boxes(report["boxes"], report["spaces"])


def from_boxes(boxes, spaces):
    by_line = {}
    for b in boxes:
        by_line.setdefault((b["page"], b["line"]), []).append(b)
    widths, heights, aspects, gaps, word_gaps, base_dev, size_dev, top_dev = [], [], [], [], [], [], [], []
    mid_dev = []
    shape_h = {}
    char_h = {}
    char_w = {}
    for items in by_line.values():
        items.sort(key=lambda b: b["n"])
        hang = [b for b in items if charsets.is_syllable(b["char"])]
        if len(hang) < 4:
            continue
        h = float(np.median([b["box"][3] - b["box"][1] for b in hang]))
        bottoms = np.array([b["box"][3] for b in hang], float)
        xs = np.array([(b["box"][0] + b["box"][2]) / 2 for b in hang], float)
        # Baseline wander: distance of each bottom from the line's straight fit.
        fit = np.polyfit(xs, bottoms, 1) if len(set(xs)) > 1 else [0, bottoms.mean()]
        base_dev.extend(list((bottoms - np.polyval(fit, xs)) / h))
        tops = np.array([b["box"][1] for b in hang], float)
        tfit = np.polyfit(xs, tops, 1) if len(set(xs)) > 1 else [0, tops.mean()]
        top_dev.extend(list((tops - np.polyval(tfit, xs)) / h))
        mids = (tops + bottoms) / 2
        mfit = np.polyfit(xs, mids, 1) if len(set(xs)) > 1 else [0, mids.mean()]
        mid_dev.extend(list((mids - np.polyval(mfit, xs)) / h))
        for b in hang:
            bw, bh = b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]
            widths.append(bw / h)
            heights.append(bh / h)
            aspects.append(bw / bh)
            shape_h.setdefault(shape_key(b["char"]), []).append(bh / h)
            char_h.setdefault(b["char"], []).append(bh / h)
            char_w.setdefault(b["char"], []).append(bw / h)
        size_dev.append(float(np.std([b["box"][3] - b["box"][1] for b in hang]) / h))
        for a, b in zip(items, items[1:]):
            if b["n"] != a["n"] + 1:
                continue  # a character between them was not measured
            gap = (b["box"][0] - a["box"][2]) / h
            if spaces[b["n"]]:
                word_gaps.append(gap)
            elif charsets.is_syllable(a["char"]) and charsets.is_syllable(b["char"]):
                gaps.append(gap)
    med = lambda v: float(np.median(v)) if len(v) else float("nan")  # noqa: E731
    return {
        "syllables": len(widths),
        "width": med(widths),                       # syllable ink width / height
        "aspect": med(aspects),
        "height_spread": float(np.std(heights)) if heights else float("nan"),
        "letter_gap": med(gaps),                    # between syllables of a word
        "letter_gap_spread": float(np.std(gaps)) if gaps else float("nan"),
        "word_gap": med(word_gaps),
        "baseline_spread": float(np.std(base_dev)) if base_dev else float("nan"),
        "top_spread": float(np.std(top_dev)) if top_dev else float("nan"),
        "middle_spread": float(np.std(mid_dev)) if mid_dev else float("nan"),
        # Height of each syllable shape relative to the line's typical syllable.
        "shape_height": {k: float(np.median(v)) for k, v in shape_h.items() if len(v) >= 3},
        "char_height": {k: float(np.median(v)) for k, v in char_h.items()},
        "char_width": {k: float(np.median(v)) for k, v in char_w.items()},
    }


def char_corrections(sample, rendered, key="char_height", limit=0.3):
    """Per-syllable size factors that make rendered writing match a sample of the writer's
    own sentences: a syllable written big and careful in its template box (often 으 or 의)
    is usually smaller or narrower in running writing.  Small differences are left alone."""
    out = {}
    for ch, v in sample.get(key, {}).items():
        mine = rendered.get(key, {}).get(ch)
        if mine and v:
            factor = v / mine
            if abs(np.log(factor)) > 0.08:
                out[ch] = float(np.clip(factor, 1 - limit, 1 + limit))
    return out


def shape_key(ch):
    from .compose import vowel_class
    _, vowel, final = charsets.decompose(ch)
    return f"{vowel_class(vowel)}{'+' if final else ''}"


def stroke_ratio(images, boxes):
    """Pen width relative to the median syllable height (2 x ink area / ink outline)."""
    heights = [b["box"][3] - b["box"][1] for b in boxes if charsets.is_syllable(b["char"])]
    if not heights:
        return float("nan")
    widths = []
    for image in images:
        alpha, _ = freeform._ink(image)
        binary = (alpha > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if perimeter:
            widths.append(2.0 * binary.sum() / perimeter)
    return float(np.median(widths)) / float(np.median(heights)) if widths else float("nan")


def measure_rendered(profile, lines, seed=0, exact=False):
    """Style figures of the program's own writing of ``lines`` (one line each).  By default
    the letters are cut back out of the image exactly as the writer's are, so both sides are
    measured the same way; ``exact`` uses where the program actually put every character."""
    from . import render
    from .score import slant
    boxes, spaces, masks, images = [], [], [], []
    n = 0
    for i, line in enumerate(lines):
        r = render.Renderer(profile, strip_options(seed + i))
        r.record = []
        page = np.asarray(r.render(line)[0].convert("RGB"))
        images.append(page)
        text = charsets.normalize_text(line)
        chars = [c for c in text if not c.isspace()]
        flags, prev_space = [], True
        for c in text:
            if c.isspace():
                prev_space = True
                continue
            flags.append(prev_space)
            prev_space = False
        for k, (c, rec) in enumerate(zip(chars, r.record)):
            if rec is not None:
                x0, y0, x1, y1, mask = rec
                boxes.append({"n": n + k, "char": c, "line": i, "page": i, "box": (x0, y0, x1, y1)})
                masks.append(mask)
        spaces.extend(flags)
        n += len(chars)
    if not exact:
        crops = [crop_ink(img) for img in images]
        _, rep = freeform.extract_freeform(crops, "\n".join(lines), single_line=True, work_width=None)
        boxes, spaces, images = rep["boxes"], rep["spaces"], crops
    st = from_boxes(boxes, spaces)
    st["stroke"] = stroke_ratio(images, boxes)
    st["slant"] = slant(masks)
    return st


def crop_ink(page, margin=12):
    ys, xs = np.nonzero(page.mean(axis=2) < 140)
    if len(ys) == 0:
        return page
    return page[max(0, ys.min() - margin):ys.max() + margin, max(0, xs.min() - margin):xs.max() + margin]


# Layout setting -> (figure it controls, first guess at how the figure moves with it, range,
# tolerance for picking the best tried layout).  Every figure grows with its setting.
LAYOUT_FIT = {
    "letter_spacing": ("letter_gap", "add_ink_gap", (0.2, 4.0), 0.02),
    "word_spacing": ("word_gap", "ratio", (0.5, 2.0), 0.05),
    "aspect": ("width", "ratio", (0.8, 1.25), 0.03),
    "stroke": ("stroke", "ratio", (0.6, 1.6), 0.005),
    "shear": ("slant", "degrees", (-0.2, 0.2), 0.5),
}


def calibrate_layout(profile_dir, target=None, lines=None, rounds=5, seeds=(0,)):
    """Fit the profile's letter spacing, word spacing, syllable width, pen width and slant to
    the writer's own (``target`` style figures, by default those measured on their copy
    sheets), by writing ``lines`` and measuring the result.  Each setting is moved by the
    secant through its last two tries; the best layout tried is kept.  Stores ``layout`` in
    the profile and returns it."""
    import json
    import os
    from . import render
    path = os.path.join(profile_dir, "profile.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    target = target or data.get("copy_style")
    lines = (lines or data.get("copy_lines") or [])[:14]
    if not target or not lines:
        return data.get("layout", {})
    goals = {key: target.get(figure, float("nan")) for key, (figure, _, _, _) in LAYOUT_FIT.items()}
    goals = {k: v for k, v in goals.items() if v is not None and not np.isnan(v)}
    layout = {"letter_spacing": 1.0, "word_spacing": 1.0, "aspect": 1.0, "stroke": 1.0, "shear": 0.0}
    history = {k: [] for k in goals}
    best, best_error = dict(layout), float("inf")
    for _ in range(rounds):
        data["layout"] = layout
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        profile = render.Profile(profile_dir)
        figures = [measure_rendered(profile, lines, seed) for seed in seeds]
        tried, error = dict(layout), 0.0
        for key, goal in goals.items():
            figure, guess, (lo, hi), tol = LAYOUT_FIT[key]
            got = float(np.nanmean([f[figure] for f in figures]))
            if np.isnan(got):
                continue
            error += abs(goal - got) / tol
            history[key].append((tried[key], got))
            value = tried[key]
            if len(history[key]) > 1:
                (p1, m1), (p2, m2) = history[key][-2:]
                slope = (m2 - m1) / (p2 - p1) if abs(p2 - p1) > 1e-6 else 0.0
            else:
                slope = 0.0
            if slope > 0:
                value = value + (goal - got) / slope
            elif guess == "add_ink_gap":  # the spacer keeps INK_GAP x spacing of room between letters
                value = value + (goal - got) / render.INK_GAP
            elif guess == "ratio" and got > 0:
                value = value * goal / got
            elif guess == "degrees":
                value = float(np.tan(np.arctan(value) + np.radians(goal - got)))
            layout[key] = float(np.clip(value, lo, hi))
        if error < best_error:
            best, best_error = tried, error
    data["layout"] = best
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return best


def strip_options(seed):
    """A long blank strip with room for one line of any length (lines must not wrap when the
    program's writing is compared line by line with the writer's)."""
    from . import render
    return render.Options(paper="blank", margin_line=False, seed=seed, line_pitch=10, size=0.5,
                          page_width=600.0, page_height=50.0)


def render_lines(profile, lines, seed=0):
    """The program's own writing of ``lines``, one image per line (for comparison)."""
    from . import render
    crops = []
    for i, line in enumerate(lines):
        r = render.Renderer(profile, strip_options(seed + i))
        page = np.asarray(r.render(line)[0].convert("RGB"))
        if (page.mean(axis=2) < 140).any():
            crops.append(crop_ink(page))
    return crops


def calibrate(profile_dir, images, transcript, rounds=2):
    """Learn per-syllable size corrections from the writer's own sentences and store them in
    the profile.  Returns ``(sample_style, rendered_style, corrections)``."""
    import json
    import os
    from . import render
    path = os.path.join(profile_dir, "profile.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sample = measure(images, transcript)
    lines = [l for l in charsets.normalize_text(transcript).split("\n") if l.strip()]
    data["char_scale"], data["char_width"] = {}, {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    corrections, widths = {}, {}
    for n in range(rounds):  # a second round refines what the first got roughly right
        profile = render.Profile(profile_dir)
        crops = render_lines(profile, lines, seed=1000 * n)
        _, rep = freeform.extract_freeform(crops, "\n".join(lines))
        rendered = from_boxes(rep["boxes"], rep["spaces"])
        for ch, factor in char_corrections(sample, rendered).items():
            corrections[ch] = float(np.clip(corrections.get(ch, 1.0) * factor, 0.8, 1.15))
        # Width relative to height and to the writing as a whole: only how this syllable's
        # shape differs from the rest (overall narrowness is the renderer's HANGUL_ASPECT).
        overall = sample["width"] / rendered["width"]
        for ch, factor in char_corrections(sample, rendered, "char_width").items():
            hf = char_corrections(sample, rendered).get(ch, 1.0)
            f = factor / hf / overall
            if abs(np.log(f)) > 0.08:
                widths[ch] = float(np.clip(widths.get(ch, 1.0) * f, 0.8, 1.2))
        data["char_scale"], data["char_width"] = corrections, widths
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    data["style_sample"] = {k: v for k, v in sample.items() if not isinstance(v, dict)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return sample, rendered, corrections
