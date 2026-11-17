"""Get plain text out of uploaded documents."""
import re

import pymupdf


class TextError(ValueError):
    pass


SENTENCE_END = re.compile(r"[.!?。:다요죠음함됨임][\"')\]」』]?$")


def _join_lines(lines):
    out = ""
    for line in (l.strip() for l in lines):
        if not line:
            continue
        if not out:
            out = line
        elif re.search(r"[A-Za-z]-$", out) and line[:1].islower():
            out = out[:-1] + line  # word hyphenated across lines
        else:
            out += " " + line
    return out


def _block_paragraphs(block):
    """Split a PDF text block into paragraphs.

    A line that ends a sentence and stops well short of the block's right edge is taken as the
    end of a paragraph; every other line break is just the PDF's wrapping.
    """
    lines = []
    for line in block["lines"]:
        text = "".join(span["text"] for span in line["spans"]).strip()
        if text:
            lines.append((text, line["bbox"][2]))
    if not lines:
        return []
    right = max(x1 for _, x1 in lines)
    width = right - block["bbox"][0]
    paragraphs, current = [], []
    for i, (text, x1) in enumerate(lines):
        current.append(text)
        last = i == len(lines) - 1
        if last or (right - x1 > 0.15 * width and SENTENCE_END.search(text)):
            paragraphs.append(_join_lines(current))
            current = []
    return paragraphs


def extract_text(data, filename="", keep_lines=False):
    name = filename.lower()
    if data[:5] == b"%PDF-" or name.endswith(".pdf"):
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise TextError("PDF를 열 수 없습니다.") from exc
        paragraphs = []
        for page in doc:
            for block in page.get_text("dict", sort=True)["blocks"]:
                if block.get("type") != 0:  # image block
                    continue
                if keep_lines:
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"]).rstrip()
                        if text.strip():
                            paragraphs.append(text)
                else:
                    paragraphs.extend(_block_paragraphs(block))
        doc.close()
        if not paragraphs:
            raise TextError("PDF에서 글자를 찾지 못했습니다. 스캔한 이미지 PDF는 아직 지원하지 않습니다 "
                            "(글자를 복사할 수 있는 PDF만 가능).")
        return "\n".join(paragraphs)
    for encoding in ("utf-8-sig", "cp949", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise TextError("텍스트 파일의 인코딩을 알 수 없습니다 (UTF-8 또는 CP949로 저장해 주세요).")
