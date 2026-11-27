"""A short template: as few boxes as possible for writing a given document well.

Hangul syllables need not all be written: an unwritten one is put together from jamo cut out
of written ones.  So a short template holds

0. the basic jamo on their own (ㄱ ... ㅎ, ㅏ ... ㅣ): when a jamo cannot be cut cleanly out of
   any written syllable, the composer uses these, so no syllable falls back to a font;
1. syllables that between them show every jamo the document needs, in each position it is
   needed in (an initial beside a vowel on the right and above one underneath, each vowel with
   and without a final, each final) -- chosen among the document's most frequent syllables;
   on longer templates every jamo is shown twice, as cutting one out can fail;
2. the document's other characters (Latin letters, digits, symbols), which cannot be put
   together from anything, most frequent first;
3. as many more of the document's most frequent syllables as fit, since written letters look
   best -- and if the document is short, second copies of its most used characters.

Without a document, ordinary Korean (charsets.COMMON_HANGUL in order) stands in for it.
"""
from collections import Counter

from . import charsets
from .compose import vowel_class

BASIC_MARKS = ".,()-="
BASIC_JAMO = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"


def _base_initial(jamo):
    """ㄲ is written as two ㄱ: the composer builds doubled consonants from the single one."""
    parts = charsets.JAMO_PARTS.get(jamo)
    return parts[0] if parts and parts[0] == parts[-1] else jamo


def features(ch):
    """What writing syllable ``ch`` teaches about the writer's jamo."""
    initial, vowel, final = charsets.decompose(ch)
    group = "right" if vowel_class(vowel) in ("right", "right_wide") else "below"
    out = {("I", _base_initial(initial), group), ("V", vowel, bool(final))}
    for part in charsets.JAMO_PARTS.get(final, final):  # ㄺ: ㄹ and ㄱ as finals
        out.add(("F", part))
    return out


def _generic_counts():
    """Stand-in frequencies for ordinary Korean when there is no document."""
    words = charsets.COMMON_HANGUL
    counts = Counter({c: len(words) - i for i, c in enumerate(words)})
    for i, c in enumerate("eatonisrhldcumfpgwybvkxjqz"):
        counts[c] = 60 - 2 * i
    for c in charsets.DIGITS:
        counts[c] = 40
    for c in BASIC_MARKS + "?!:'~":
        counts[c] = 50
    return counts


def choose(counts, cells, known=""):
    """Characters for a ``cells``-box template for a document whose characters are
    ``counts`` (a Counter; empty for ordinary Korean).  ``known``: characters already in the
    writer's handwriting, left out (their jamo are already known as well)."""
    counts = Counter(counts) or _generic_counts()
    known = set(known)
    syllables = [c for c, _ in counts.most_common() if charsets.is_syllable(c)]
    others = [c for c, _ in counts.most_common() if not charsets.is_syllable(c) and c not in known]
    needed = set().union(*(features(c) for c in syllables)) if syllables else set()
    for c in known:
        if charsets.is_syllable(c):
            needed -= features(c)
    # 0. The basic jamo, for the composer to fall back on.
    chosen = [j for j in BASIC_JAMO if j not in known] if syllables else []
    # 1. Cover every jamo feature, preferring frequent syllables (greedy set cover), twice
    #    over when there is room.
    core_budget = int(0.45 * cells)
    candidates = [c for c in syllables if c not in known]
    top = max((counts[c] for c in candidates), default=1)
    for _ in range(2 if cells >= 2 * 110 else 1):
        todo = set(needed)
        while todo and len(chosen) < core_budget:
            best, gain = None, 0.0
            for c in candidates:
                new = len(features(c) & todo)
                if new:
                    score = new + counts[c] / top  # a frequent syllable wins a tie
                    if score > gain:
                        best, gain = c, score
            if best is None:
                break
            chosen.append(best)
            candidates.remove(best)
            todo -= features(best)
    # 2./3. The rest by importance: other characters count three times (they have no
    # stand-in), syllables by how often they are used.
    rest = [(3 * counts[c], c) for c in others] + [(counts[c], c) for c in candidates]
    rest.sort(key=lambda t: -t[0])
    chosen += [c for _, c in rest[:max(0, cells - len(chosen))]]
    # 4. A short document leaves boxes over: second copies of its most used characters, so
    #    they come out a little different each time.
    again = [c for c, _ in counts.most_common() if c in chosen and c not in BASIC_JAMO]
    while len(chosen) < cells and again:
        chosen += again[:cells - len(chosen)]
    return chosen[:cells]


def coverage(counts, written, known=""):
    """Shares of the document's character uses that will be written from the template,
    put together from jamo, or (other characters not written) drawn from a font."""
    counts = Counter(counts) or _generic_counts()
    have = set(written) | set(known)
    total = sum(counts.values()) or 1
    direct = sum(n for c, n in counts.items() if c in have)
    composed = sum(n for c, n in counts.items() if c not in have and charsets.is_syllable(c))
    return {"direct": direct / total, "composed": composed / total,
            "font": (total - direct - composed) / total}
