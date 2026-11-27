"""Write the official template definitions kept in the repository (templates/<id>.json).

Templates made on the server get random IDs and live in the data folder; the official ones
have fixed IDs and are installed on every start, so a filled-in copy is always recognised.

    python tools/official_templates.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hwnote import template as T  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

# One page to try the writing tool (iPad or paper) before filling in the full template.
# The syllables cover every vowel position (right, below, mixed, wide), single and double
# finals, so jamo can already be cut from them.
PILOT = (
    "가너다려마비사아자차카타파하이"      # vowel on the right
    "개네대세"                          # wide vowels ㅐ ㅔ
    "고노두로무보소우주호"                # vowel underneath
    "과외위의"                          # mixed vowels
    "각난달람밥산잘한정문손을는들"           # with a final consonant
    "값닭꽃쌍"                          # double / compound
    "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"
    "0123456789"
    "ABCDNTXanx"
    ".,!?()-:/~="
    "≤≥→∞"
)

OFFICIAL = {
    100001: ("시험용 1쪽 템플릿", PILOT),
}


def build():
    os.makedirs(OUT, exist_ok=True)
    for template_id, (name, chars) in OFFICIAL.items():
        chars = list(chars)
        assert len(chars) == len(set(chars)), f"duplicate characters in {name}"
        pages = [chars[i:i + T.CELLS_PER_PAGE] for i in range(0, len(chars), T.CELLS_PER_PAGE)]
        info = {"id": template_id, "layout": 1, "name": name, "official": True, "pages": pages}
        with open(os.path.join(OUT, f"{template_id}.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        print(template_id, name, len(chars), "cells,", len(pages), "page(s)")


if __name__ == "__main__":
    build()
