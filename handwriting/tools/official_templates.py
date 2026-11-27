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

import copy_text  # noqa: E402  (same folder)

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

# Full template for the calculus notes (微積分 理論整理): every Hangul syllable of the
# document, with the frequent ones written two or three times so the same letter is not drawn
# identically every time (tiers by count in the document: >=60, 20-59, <20).
CALC_H3 = "수의이에하함서다한로정는면분가여대라열을증리고명법렴지므부선미도각를적연그"
CALC_H2 = ("점과속때으인자곡위은구와계시일성따간극급래방양모산식기발비아유단실즉직용놓무두역프능변"
           "합든소음있항값동삼등상쌍할나어또사치만질형")
CALC_H1 = ("닫제최힌갖조진판호개교록볼크우움같반평거길본임좌축접트결공러없주참표환감넓코타피관근니오"
           "재젠족존탄까립마순경된목바번운원집째찬키해활꼴데되들루림며보생안않얻예차체칙뀌런례론린복"
           "셈알중편것균내념댓덧려영입전절취탈터건낸더드띠른매몫솟심외작잡첫칭커컨향게꾸누닌둘떤련많"
           "문및밑배빼설스슷신싸완율응장져짓짝켜킨통특학홀히곱귀규귝난납높눗느닯답디락략력르름릴막맥"
           "멍못붙브석숫쉬약웃워읽잃준짐쪽착채충큰풀행현확황후")
GENERAL = "국화세요회저물너민했"            # common in other writing, absent from the notes
MORE_JAMO = "ㄲㄸㅃㅆㅉㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ"   # jamo not on the pilot page
LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
GREEK = "αβγδεθηλμπσφωΔΣ"
SYMBOLS = "()[]{}|=+-×÷±∓<>≤≥≠≈→⇒⇐⇔∞∫√∀∃∈⊂∪∩∅∴∵'·⋯⋮,.:;!?/~*■①②③④⑤⑥"
SECOND = "xnaislfmcdteobyhgupkr" + "012345" + "()=-+,.'|<>[]{}→∞∫" + "αθ"
THIRD = "xnaislf" + "12" + "()=-+"

CALC = (CALC_H3 + CALC_H2 + CALC_H1 + GENERAL + MORE_JAMO + LOWER + UPPER + DIGITS + GREEK + SYMBOLS
        + CALC_H3 + CALC_H2 + SECOND
        + CALC_H3 + THIRD)

OFFICIAL = {
    100001: ("시험용 1쪽 템플릿", PILOT, False),
    100002: ("미적분 노트용 전체 템플릿", CALC, True),
}


COPY_ID = 100003


def copy_sheet_pages():
    """Pages of the copy sheets: the evaluation page first (held out from learning, used to
    measure the result), then prose and formula pages alternating while both last."""
    per = T.COPY_ROWS
    chunk = lambda lines: [lines[i:i + per] for i in range(0, len(lines), per)]  # noqa: E731
    prose = chunk(T.wrap_copy_text(copy_text.PROSE))
    formulas = chunk(T.wrap_copy_text(copy_text.FORMULAS, width=150.0))
    evaluation = chunk(T.wrap_copy_text(copy_text.EVALUATION, width=150.0))
    assert len(evaluation) == 1, "the evaluation text must fit on one page"
    pages = evaluation[:]
    order = [prose[0]] + formulas[:1] + prose[1:2] + formulas[1:2] + prose[2:] + formulas[2:]
    pages.extend(order)
    return pages


def build():
    os.makedirs(OUT, exist_ok=True)
    pages = copy_sheet_pages()
    info = {"id": COPY_ID, "layout": 2, "name": "필사 시트", "official": True, "eval_pages": [0],
            "pages": pages}
    with open(os.path.join(OUT, f"{COPY_ID}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(COPY_ID, "copy sheets", sum(len(p) for p in pages), "lines,", len(pages), "page(s)")
    for template_id, (name, chars, variants) in OFFICIAL.items():
        chars = list(chars)
        if not variants:
            assert len(chars) == len(set(chars)), f"duplicate characters in {name}"
        pages = [chars[i:i + T.CELLS_PER_PAGE] for i in range(0, len(chars), T.CELLS_PER_PAGE)]
        info = {"id": template_id, "layout": 1, "name": name, "official": True, "variants": variants,
                "pages": pages}
        with open(os.path.join(OUT, f"{template_id}.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        print(template_id, name, len(chars), "cells,", len(pages), "page(s)")


if __name__ == "__main__":
    build()
