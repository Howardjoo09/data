"""Fit a profile to the writer's copy sheets and measure how close the result is.

    python tools/tune_profile.py DATA_DIR PROFILE_ID [--out DIR] [--no-fit]

1. Letter spacing, word spacing and syllable width are fitted to the writer's own copy
   sheets (style.calibrate_layout), never to the evaluation page.
2. The program writes the evaluation page's text; hwnote/score.py compares it with the
   writer's evaluation page and prints the figures.  A side-by-side image is written to
   ``--out`` (writer's line above the program's, line by line).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hwnote import copysheet, extract, render, score, style  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data")
    parser.add_argument("profile")
    parser.add_argument("--out", default=".")
    parser.add_argument("--no-fit", action="store_true", help="only measure, keep the profile as it is")
    args = parser.parse_args()
    directory = extract.profile_dir(args.data, args.profile)
    t = time.time()
    if not args.no_fit:
        layout = style.calibrate_layout(directory)
        print("layout:", {k: round(v, 3) for k, v in layout.items()}, f"({time.time() - t:.0f}s)")
    images, lines = copysheet.load_evaluation(directory)
    if not images:
        print("평가용 시트가 없습니다 (필사 시트 1쪽).")
        return
    result = score.evaluate(render.Profile(directory), images, lines,
                            source_px_per_mm=copysheet.evaluation_resolution(directory))
    print(score.report_text(result))
    os.makedirs(args.out, exist_ok=True)
    score.side_by_side(images, result["your_report"], result["renders"], result["render_report"],
                       os.path.join(args.out, "side_by_side.png"))
    keep = {k: result[k] for k in ("total", "shape", "layout", "layout_scores", "figures", "accuracy",
                                   "closeness", "own_level", "mine_level")}
    with open(os.path.join(args.out, "score.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=1)
    print(f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
