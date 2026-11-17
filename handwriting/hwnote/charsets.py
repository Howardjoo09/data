"""Character sets offered by the handwriting template, plus Hangul helpers."""
import unicodedata

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCT = ".,!?'\"()[]-:;/%~+=<>@#&*_·→"

CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ",
             "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JAMO = CHOSEONG + JUNGSEONG

# Jamo that can be assembled from simpler handwritten jamo when the user skipped them.
JAMO_PARTS = {
    "ㄲ": "ㄱㄱ", "ㄸ": "ㄷㄷ", "ㅃ": "ㅂㅂ", "ㅆ": "ㅅㅅ", "ㅉ": "ㅈㅈ",
    "ㄳ": "ㄱㅅ", "ㄵ": "ㄴㅈ", "ㄶ": "ㄴㅎ", "ㄺ": "ㄹㄱ", "ㄻ": "ㄹㅁ", "ㄼ": "ㄹㅂ",
    "ㄽ": "ㄹㅅ", "ㄾ": "ㄹㅌ", "ㄿ": "ㄹㅍ", "ㅀ": "ㄹㅎ", "ㅄ": "ㅂㅅ",
    "ㅐ": "ㅏㅣ", "ㅔ": "ㅓㅣ", "ㅒ": "ㅑㅣ", "ㅖ": "ㅕㅣ",
}
# Mixed vowels = a horizontal part (under the initial) + a vertical part (on the right).
MIXED_VOWELS = {"ㅘ": "ㅗㅏ", "ㅙ": "ㅗㅐ", "ㅚ": "ㅗㅣ", "ㅝ": "ㅜㅓ", "ㅞ": "ㅜㅔ", "ㅟ": "ㅜㅣ", "ㅢ": "ㅡㅣ"}
RIGHT_VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅣ")
BOTTOM_VOWELS = set("ㅗㅛㅜㅠㅡ")

# Frequent Hangul syllables, most common first (first 110 fill one template page).
_COMMON = """
이다는의에가하고을지서기로를사한리자대나일도어수시아그인정으들있국것적게해주부라상전우만소니장
제구성여내원은경면거연화동과보비생와무오신미중문세위실요마각공개회관저계등행방물진학모발선치법분식
결터본간너데운민단며했할때된될및까래려히았었던든통산업체명영안말년월날집람음감강건교육금노당독력류
록반번변복북불새석설속손스습승씨야약양언역예외용임입작재점조종직질차참책처천초최추출카타트특파편평
포표프필합항현형호확활후희힘겠됩렇럼런렸많좋없같알싶봤줘됐써쓰읽찾왔갔났줄볼갈올둘셋넷첫몇왜뭐좀잘
못더또꼭늘막곳곧길김박남녀님네냐뿐쪽짜째찬친침커컴코크큰태택테토투판팔페품풀피향허험혀협혹환황획
효흐흔넘넣놓누눈느늦담답덕덜돈돌두뒤드득듯땅떠떨뜻락란랑략량러레렵령례론료루르른름릴림립맛망맞매
머먹먼멀메몸몰믿밀밑바받밝밤밥배백버벌범별병봄봉삼살색섞섬섭셈송쉬쉽슬식심십싸씩악않앞애얼엄엇열
염옆옛온옮완왕왼욕움웃웠유은응익잃잊잠잡잔쟁절젊접져졌존졸좁죠준즈즉즐증짐짓찍착찰채철청쳐총춤충
취층칙칠칸켜쿠키탁탄털튼틀팀퍼펴폭픈핀함헌헤혁혜홈훈휴흘흰
"""
COMMON_HANGUL = "".join(dict.fromkeys(c for c in _COMMON if "가" <= c <= "힣"))

CHARSETS = {
    "upper": ("영어 대문자", UPPER),
    "lower": ("영어 소문자", LOWER),
    "digits": ("숫자", DIGITS),
    "punct": ("문장부호", PUNCT),
    "jamo": ("한글 자모 (없는 글자 조합용, 강력 추천)", JAMO),
    "hangul_basic": (f"자주 쓰는 한글 {min(110, len(COMMON_HANGUL))}자", COMMON_HANGUL[:110]),
    "hangul_more": (f"자주 쓰는 한글 추가 {max(0, len(COMMON_HANGUL) - 110)}자", COMMON_HANGUL[110:]),
}

# Typographic characters that the renderer writes with a simpler handwritten equivalent.
SUBSTITUTE = {
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "`": "'",
    "–": "-", "—": "-", "‐": "-", "−": "-", "•": "·", "∙": "·", "ㆍ": "·",
    "（": "(", "）": ")", "，": ",", "．": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    " ": " ", "　": " ", "\t": "    ",
}
EXPAND = {"…": "...", "⋯": "..."}

# Template labels for characters that look alike in print.
LABEL_HINTS = {
    "I": "대문자 i", "l": "소문자 L", "O": "대문자 o", "o": "소문자 o", "0": "숫자", "1": "숫자",
    "'": "작은따옴표", '"': "큰따옴표", "-": "줄표", "_": "밑줄", "·": "가운뎃점", ",": "쉼표", ".": "마침표", "→": "화살표",
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    for src, dst in EXPAND.items():
        text = text.replace(src, dst)
    return "".join(SUBSTITUTE.get(c, c) for c in text)


def is_syllable(ch):
    return "가" <= ch <= "힣"


def is_jamo(ch):
    return "ㄱ" <= ch <= "ㅣ"


def decompose(ch):
    """'각' -> ('ㄱ', 'ㅏ', 'ㄱ'); final is '' when absent."""
    code = ord(ch) - 0xAC00
    initial, rest = divmod(code, 588)
    medial, final = divmod(rest, 28)
    return CHOSEONG[initial], JUNGSEONG[medial], JONGSEONG[final]


def unique_chars(text):
    """Characters of ``text`` in first-seen order, without whitespace, after normalisation."""
    return "".join(dict.fromkeys(c for c in normalize_text(text) if not c.isspace() and c.isprintable()))


def build_charlist(set_keys, custom_text="", repeat=1):
    chars = []
    for key in set_keys:
        if key in CHARSETS:
            chars.extend(CHARSETS[key][1])
    chars.extend(unique_chars(custom_text))
    chars = list(dict.fromkeys(chars))
    return [c for c in chars for _ in range(max(1, min(3, int(repeat))))]
