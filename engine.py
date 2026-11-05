"""Platform-independent Hangul keystroke progression."""
from dataclasses import dataclass
import unicodedata

INITIALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"


def stages(character: str) -> tuple[str, ...]:
    code = ord(character) - 0xAC00
    if 0 <= code < 11172:
        initial, remainder = divmod(code, 588)
        _, final = divmod(remainder, 28)
        base = chr(ord(character) - final)
        return (INITIALS[initial], base, character) if final else (INITIALS[initial], base)
    return (character,)


@dataclass(frozen=True)
class Edit:
    erase: int
    text: str


class Player:
    def __init__(self, text: str = ""):
        self.reset(text)

    def reset(self, text: str):
        self.text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        self.index = 0
        self.stage = 0
        self.replace_previous = False
        self.active = False

    @property
    def done(self):
        return self.index >= len(self.text)

    def start(self):
        self.active = not self.done

    def pause(self):
        self.active = False

    def detach(self):
        """An external edit/cursor move makes replacing the old character unsafe."""
        self.replace_previous = False

    def peek(self):
        if not self.active or self.done:
            return None
        return Edit(int(self.replace_previous), stages(self.text[self.index])[self.stage])

    def commit(self):
        """Advance only after the corresponding input was successfully sent."""
        if self.peek() is None:
            return
        self.stage += 1
        if self.stage == len(stages(self.text[self.index])):
            self.index += 1
            self.stage = 0
            self.replace_previous = False
        else:
            self.replace_previous = True
        if self.done:
            self.active = False
