"""Characters of the equation font in Hancom (HWP) PDFs.

Equations exported from HWP use a font whose glyphs sit in the Unicode private use area, so
PDF text extraction yields codes like U+E0FC instead of "x".  The table below was made by
looking at every such glyph in the calculus notes this program is mainly used for.

Values are the character to write by hand.  Pieces of tall brackets (⎧ ⎨ ⎩ ...) are kept as
the Unicode bracket-piece characters so the layout code can join them; FRACTION is the bar
of a fraction and OVERDOT the dot of a repeating decimal.
"""

FRACTION = "―"   # horizontal bar of a fraction (drawn as a line, not a glyph)
OVERDOT = "˙"    # ˙ above a repeating digit

PUA = {
    # Capital letters (italic in print, written plainly)
    0xE000: "A", 0xE001: "B", 0xE002: "C", 0xE003: "D", 0xE004: "E", 0xE005: "F", 0xE006: "G",
    0xE007: "H", 0xE008: "I", 0xE009: "J", 0xE00A: "K", 0xE00B: "L", 0xE00C: "M", 0xE00D: "N",
    0xE00E: "O", 0xE00F: "P", 0xE010: "Q", 0xE011: "R", 0xE012: "S", 0xE013: "T", 0xE014: "U",
    0xE015: "V", 0xE016: "W", 0xE017: "X", 0xE018: "Y", 0xE019: "Z",
    # Digits and plain symbols
    0xE034: "1", 0xE035: "2", 0xE036: "3", 0xE037: "4", 0xE038: "5", 0xE039: "6", 0xE03A: "7",
    0xE03B: "8", 0xE03C: "9", 0xE03D: "0", 0xE03E: "!",
    0xE044: "(", 0xE045: ")", 0xE046: "-", 0xE047: "=", 0xE048: "+", 0xE049: "[", 0xE04A: "]",
    0xE04B: "{", 0xE04C: "}", 0xE04D: "|", 0xE04F: ":", 0xE052: ",", 0xE053: ".", 0xE054: "/",
    0xE055: "<", 0xE056: ">",
    # Operators and arrows
    0xE05B: "∫", 0xE05C: "√", 0xE05F: OVERDOT, 0xE067: "Σ", 0xE06D: FRACTION, 0xE06E: "→",
    0xE06F: "⇐", 0xE070: "⇔", 0xE071: "⇒",
    # Pieces of tall braces and brackets
    0xE078: "⎧", 0xE079: "⎨", 0xE07A: "⎩", 0xE07B: "⎪", 0xE082: "⎫", 0xE083: "⎬", 0xE084: "⎭",
    0xE100: "⎡", 0xE101: "|", 0xE102: "⎤", 0xE103: "⎣", 0xE104: "⎦", 0xE105: "⎥",
    # Greek
    0xE088: "Δ", 0xE09D: "α", 0xE09E: "β", 0xE0A1: "ε", 0xE0A3: "η", 0xE0A4: "θ", 0xE0AC: "π",
    0xE10E: "ε",
    # Small letters (italic in print)
    0xE0E5: "a", 0xE0E6: "b", 0xE0E7: "c", 0xE0E8: "d", 0xE0E9: "e", 0xE0EA: "f", 0xE0EB: "g",
    0xE0EC: "h", 0xE0ED: "i", 0xE0EE: "j", 0xE0EF: "k", 0xE0F0: "l", 0xE0F1: "m", 0xE0F2: "n",
    0xE0F3: "o", 0xE0F4: "p", 0xE0F5: "q", 0xE0F6: "r", 0xE0F7: "s", 0xE0F8: "t", 0xE0F9: "u",
    0xE0FA: "v", 0xE0FB: "w", 0xE0FC: "x", 0xE0FD: "y", 0xE0FE: "z",
}

BRACKET_PIECES = set("⎧⎨⎩⎪⎫⎬⎭⎡⎤⎣⎦⎥")

# Printed characters written with a simpler handwritten equivalent.
PLAIN = {"∙": "·", "⋅": "·", "−": "-", "∆": "Δ", "′": "'", "‘": "'", "’": "'", "`": "'"}


def readable(ch):
    """The character a PDF code stands for (PUA equation glyphs decoded)."""
    code = ord(ch)
    if 0xE000 <= code <= 0xF8FF:
        return PUA.get(code, "")
    return PLAIN.get(ch, ch)
