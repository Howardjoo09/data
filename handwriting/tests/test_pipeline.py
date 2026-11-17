"""End-to-end checks: template -> fake photo -> profile -> handwritten notes.

Run from the handwriting folder:  python -m unittest discover -s tests -v
"""
import io
import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from hwnote import charsets, compose, extract, render, template as T, textio  # noqa: E402
import make_sample  # noqa: E402


def font_file(directory):
    """The CJK font bundled with PyMuPDF stands in for handwriting in tests."""
    path = os.path.join(directory, "test_font.ttf")
    with open(path, "wb") as f:
        f.write(pymupdf.Font("korea").buffer)
    return path


class TemplateTests(unittest.TestCase):
    def test_page_code_roundtrip_and_checksum(self):
        for template_id, page in [(1, 0), (123456, 7), (0xFFFFFE, 255)]:
            bits = T.encode_bits(template_id, page)
            self.assertEqual(T.decode_bits(bits), (template_id, page))
            for i in (1, 20, 40):
                broken = list(bits)
                broken[i] ^= 1
                self.assertIsNone(T.decode_bits(broken))

    def test_charlist_dedupes_and_adds_text_characters(self):
        chars = charsets.build_charlist(["digits"], "1 가가 “AB”\n", repeat=2)
        self.assertEqual("".join(chars), "00112233445566778899가가\"\"AABB")

    def test_cells_stay_inside_page_and_markers(self):
        x, y = T.cell_box(T.CELLS_PER_PAGE - 1)
        self.assertLess(x + T.BOX, T.MARKERS[1][0])
        self.assertLess(y + T.BOX, T.MARKERS[2][1])


class ComposeTests(unittest.TestCase):
    def test_every_syllable_can_be_composed_from_basic_jamo(self):
        font = render._fallback_font(80)
        masks = {}
        for j in "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ":
            from PIL import Image, ImageDraw
            img = Image.new("L", (120, 120), 0)
            ImageDraw.Draw(img).text((60, 60), j, font=font, fill=255, anchor="mm")
            masks[j] = np.asarray(img, np.float32) / 255
        source = compose.JamoSource(masks.get)
        for code in range(0xAC00, 0xD7A4, 37):  # a spread over all 11,172 syllables
            mask = compose.compose_syllable(chr(code), source, 90, 95)
            self.assertIsNotNone(mask, chr(code))
            self.assertGreater(mask.sum(), 50, chr(code))
        self.assertIsNone(compose.compose_syllable("값", compose.JamoSource({}.get), 90, 95))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = cls.tmp.name
        sets = ["upper", "lower", "digits", "punct", "jamo", "hangul_basic"]
        cls.info, cls.photos = make_sample.make(cls.data, font_file(cls.data), sets, "광합성", seed=5, skip=0.0)
        cls.profile, cls.empty, cls.warnings = extract.build_profile(
            cls.data, [(p, f"page{n}.jpg") for n, p in enumerate(cls.photos)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_written_box_is_extracted(self):
        cells = sum(len(p) for p in self.info["pages"])
        self.assertEqual(self.warnings, [])
        self.assertEqual(self.empty, [])
        self.assertEqual(len(self.profile["glyphs"]), cells)
        self.assertEqual([g["char"] for g in self.profile["glyphs"]], [c for p in self.info["pages"] for c in p])

    def test_glyph_shapes_match_their_labels(self):
        glyphs = {g["char"]: g for g in self.profile["glyphs"]}
        self.assertGreater(glyphs["I"]["h"], 2 * glyphs["I"]["w"])        # tall and thin
        self.assertGreater(glyphs["-"]["w"], 2 * glyphs["-"]["h"])        # wide and flat
        self.assertGreater(glyphs["g"]["top"] + glyphs["g"]["h"], T.BASELINE + 1)  # descender
        self.assertLess(glyphs["a"]["top"] + glyphs["a"]["h"], T.BASELINE + 1)
        self.assertLess(abs(glyphs["."]["top"] + glyphs["."]["h"] - T.BASELINE), 1.0)
        for g in self.profile["glyphs"]:  # nothing of the printed box or baseline leaks in
            self.assertLess(g["w"], T.BOX - 2 * extract.INSET - 0.5, g["char"])

    def test_upside_down_photo_reads_the_same(self):
        image = extract.load_images(self.photos[0])[0]
        flipped = cv2.rotate(image, cv2.ROTATE_180)
        self.assertEqual(extract.read_page_id(extract.rectify(flipped)), (self.info["id"], 0))

    def test_non_template_image_is_rejected(self):
        ok, png = cv2.imencode(".png", np.full((800, 600, 3), 200, np.uint8))
        with self.assertRaises(extract.ExtractError):
            extract.build_profile(self.data, [(png.tobytes(), "photo.png")])

    def test_render_notes(self):
        directory = extract.profile_dir(self.data, self.profile["id"])
        profile = render.Profile(directory)
        text = "광합성 정리\n\n식물은 빛 에너지로 포도당을 만든다. (CO2 + H2O → glucose) 漢\n" + "긴 문장 " * 450
        renderer = render.Renderer(profile, render.Options(seed=1))
        pages = renderer.render(text)
        self.assertEqual(len(pages), 2)
        self.assertIn("빛", renderer.factory.composed)       # not in the template -> built from jamo
        self.assertEqual(renderer.factory.fallback, {"漢"})  # only the hanja had no handwriting
        ink = 255 - np.asarray(pages[0].convert("L"), np.float32)
        self.assertGreater((ink > 120).mean(), 0.01)
        pdf = render.pages_to_pdf(pages)
        self.assertEqual(pymupdf.open(stream=pdf, filetype="pdf").page_count, 2)

    def test_web_api(self):
        os.environ["HWNOTE_DATA"] = self.data
        import server
        server.DATA = self.data
        client = server.app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        r = client.post("/api/template", json={"sets": ["digits"], "text": "가나"})
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json["cells"], 12)
        with client.get(r.json["pdf"]) as pdf:
            self.assertTrue(pdf.data.startswith(b"%PDF"))
        r = client.post("/api/profile", data={"files": [(io.BytesIO(self.photos[0]), "p1.jpg")]},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.json)
        profile_id = r.json["id"]
        with client.get(r.json["sheet"]) as sheet:
            self.assertEqual(sheet.mimetype, "image/png")
        r = client.post("/api/render", json={"profile_id": profile_id, "text": "Hello 세계", "paper": "grid"})
        self.assertEqual(r.status_code, 200, r.json)
        with client.get(r.json["pdf"]) as pdf:
            self.assertTrue(pdf.data.startswith(b"%PDF"))
        self.assertEqual(client.post("/api/render", json={"profile_id": "../../etc", "text": "x"}).status_code, 404)
        self.assertEqual(client.get("/api/output/abc/../../x").status_code, 404)


class TextTests(unittest.TestCase):
    def test_pdf_paragraphs_are_rebuilt(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
        body = ("첫 문단은 아주 길어서 PDF 안에서 여러 줄로 자동 줄바꿈이 됩니다. " * 2 + "끝입니다.\n"
                "둘째 문단입니다.")
        page.insert_textbox(pymupdf.Rect(50, 50, 400, 400), body, fontname="kr", fontsize=11)
        text = textio.extract_text(doc.tobytes(), "doc.pdf")
        self.assertEqual(text.split("\n"), [body.split("\n")[0].strip(), "둘째 문단입니다."])

    def test_txt_encodings(self):
        self.assertEqual(textio.extract_text("한글".encode("cp949"), "a.txt"), "한글")
        self.assertEqual(textio.extract_text("﻿한글".encode("utf-8"), "a.txt"), "한글")


if __name__ == "__main__":
    unittest.main()
