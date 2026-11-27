"""End-to-end checks: template -> fake photo -> profile -> handwritten notes.

Run from the handwriting folder:  python -m unittest discover -s tests -v
"""
import glob
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

from hwnote import charsets, compose, extract, freeform, render, template as T, textio  # noqa: E402
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

    def test_official_templates_print_and_read_back(self):
        # Kept in the repository with fixed IDs, so a filled-in copy is recognised on any server.
        for path in sorted(glob.glob(os.path.join(ROOT, "templates", "*.json"))):
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(os.path.basename(path), f"{info['id']}.json")
            chars = [c for page in info["pages"] for c in page]
            if not info.get("variants"):
                self.assertEqual(len(chars), len(set(chars)))
            pdf = T.build_pdf(info)
            for n in range(len(info["pages"])):
                page = extract.rectify(T.render_page_image(pdf, n, 6))
                self.assertEqual(extract.read_page_id(page), (info["id"], n))

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


FREEFORM_TEXT = ("오늘은 생명과학 시간에 세포 호흡에 대해 배웠다. 세포 호흡은 포도당을 분해해서 에너지를 얻는 과정이다.\n"
                 "산소가 충분하면 ATP가 많이 만들어지지만, 부족하면 젖산 발효가 일어난다.\n"
                 "다음 주까지 광합성과 비교하는 표를 만들어 오기! (시험 범위: 1~3단원)")


def located(report, truth, transcript):
    """(correct, wrong): whether each learned character was cut from the right place."""
    chars = [c for c in transcript if not c.isspace()]
    correct = wrong = 0
    for n, (x, y) in report["positions"].items():
        hit = [c for c, x0, y0, x1, y1 in truth if x0 - 2 <= x <= x1 + 2 and y0 - 2 <= y <= y1 + 2]
        if hit and hit[0] == chars[n]:
            correct += 1
        else:
            wrong += 1
    return correct, wrong


class FreeformTests(unittest.TestCase):
    """Ordinary handwriting (no template) plus a transcript of what it says."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.font = font_file(cls.tmp.name)
        page, truth = make_sample.freeform_page(FREEFORM_TEXT, cls.font, seed=2)
        cls.photo, cls.truth = make_sample.freeform_photo(page, truth, seed=2, angle=-2.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_characters_are_found_in_the_right_place(self):
        image = extract.load_images(self.photo)[0]
        glyphs, report = freeform.extract_freeform([image], FREEFORM_TEXT)
        correct, wrong = located(report, self.truth, FREEFORM_TEXT)
        self.assertEqual(report["chars"], len(self.truth))
        self.assertGreaterEqual(correct, 0.95 * len(self.truth))
        self.assertLessEqual(wrong, 1)
        self.assertEqual(len(glyphs), report["learned"])

    def test_transcript_mistakes_are_skipped_not_learned_wrongly(self):
        typo = FREEFORM_TEXT.replace("분해해서", "분해서").replace("1~3단원", "1~3 단원까지")
        image = extract.load_images(self.photo)[0]
        _, report = freeform.extract_freeform([image], typo)
        correct, wrong = located(report, self.truth, typo)
        self.assertLessEqual(wrong, 2)
        self.assertGreater(correct, 0.8 * len(self.truth))
        self.assertLess(report["learned"], len(self.truth))  # words around the mistakes are left out

    def test_line_splitting_survives_crowded_lines(self):
        page, truth = make_sample.freeform_page(FREEFORM_TEXT, self.font, seed=3, hangul_mm=6.3, pitch_mm=7.5,
                                                gap=(0.0, 0.06))
        photo, truth = make_sample.freeform_photo(page, truth, seed=3)
        _, report = freeform.extract_freeform(extract.load_images(photo), FREEFORM_TEXT)
        correct, wrong = located(report, truth, FREEFORM_TEXT)
        self.assertGreaterEqual(correct, 0.9 * len(truth))
        self.assertLessEqual(wrong, 3)

    def test_unwritten_syllables_use_the_writers_own_jamo(self):
        data = os.path.join(self.tmp.name, "data")
        profile, _, warnings = extract.build_profile(data, [(self.photo, "notes.jpg")], transcript=FREEFORM_TEXT)
        self.assertEqual(warnings, [])
        renderer = render.Renderer(render.Profile(extract.profile_dir(data, profile["id"])), render.Options(seed=1))
        self.assertGreater(renderer.factory.harvested, 20)
        renderer.render("새 운 내 용")
        # 새·운·내 never appear in the sample but all their jamo do; 용 needs ㅛ, which is nowhere.
        self.assertTrue({"새", "운", "내"} <= renderer.factory.composed)
        self.assertEqual(renderer.factory.fallback, {"용"})

    def test_single_line_crop(self):
        # A crop holding one line of writing (letters fill most of its height) must still work.
        page, truth = make_sample.freeform_page("그러나, 약간의 수정사항을 고려하면 다음 정리를 얻을 수 있다.",
                                                self.font, seed=4)
        ys = [t[2] for t in truth] + [t[4] for t in truth]
        crop = page[int(min(ys)) - 6:int(max(ys)) + 6]
        ok, png = cv2.imencode(".png", crop[:, :, ::-1])
        text = "그러나, 약간의 수정사항을 고려하면 다음 정리를 얻을 수 있다."
        _, report = freeform.extract_freeform(extract.load_images(png.tobytes()), text)
        self.assertGreaterEqual(report["learned"], 0.8 * len(truth))

    def test_shape_check_catches_shifted_labels(self):
        from PIL import Image, ImageDraw
        font = render._fallback_font(90)

        def glyph(ch):
            img = Image.new("L", (130, 130), 0)
            ImageDraw.Draw(img).text((65, 65), ch, font=font, fill=255, anchor="mm")
            return np.asarray(img, np.float32) / 255

        self.assertTrue(freeform.shape_ok("위", glyph("위"), 100))
        self.assertFalse(freeform.shape_ok("위", glyph("우"), 100))   # ㅣ missing
        self.assertTrue(freeform.shape_ok("우", glyph("우"), 100))
        self.assertFalse(freeform.shape_ok("우", glyph("위"), 100))   # a neighbour's ㅣ attached
        self.assertFalse(freeform.shape_ok("가", glyph("."), 100))    # far too short for Hangul
        self.assertTrue(freeform.shape_ok("(", glyph("("), 100))
        self.assertFalse(freeform.shape_ok("(", glyph("ㅌ"), 100))

    def test_short_glyph_among_syllables_is_dropped(self):
        tall = {"char": "가", "line": 0, "top": 0, "bottom": 100}
        found = [dict(tall, n=i) for i in range(4)] + [{"char": "나", "line": 0, "top": 40, "bottom": 100, "n": 9}]
        self.assertEqual([g["n"] for g in freeform._even_heights(found)], [0, 1, 2, 3])  # 'an' learned as 나

    def test_harvest_rejects_jamo_joined_into_one_stroke(self):
        from PIL import Image, ImageDraw
        img = Image.new("L", (100, 100), 0)
        ImageDraw.Draw(img).line([(10, 20), (45, 20), (45, 80), (75, 80), (75, 5)], fill=255, width=7)
        self.assertIsNone(compose.harvest("가", np.asarray(img, np.float32) / 255))

    def test_convert_api(self):
        import server
        data = os.path.join(self.tmp.name, "web")
        server.DATA = data
        client = server.app.test_client()
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_font(fontname="kr", fontbuffer=pymupdf.Font("korea").buffer)
        page.insert_textbox(pymupdf.Rect(50, 50, 500, 300), "광합성 정리\n빛 에너지로 포도당을 만든다.",
                            fontname="kr", fontsize=12)
        form = {"document": (io.BytesIO(doc.tobytes()), "notes.pdf"),
                "handwriting": [(io.BytesIO(self.photo), "hand.jpg")],
                "transcript": FREEFORM_TEXT, "paper": "grid", "margin_line": "0"}
        r = client.post("/api/convert", data=form, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.json)
        self.assertEqual(r.json["text_source"], "document")
        self.assertIn("광합성 정리", r.json["text"])
        self.assertGreater(r.json["profile"]["freeform"]["learned"], 0.95 * len(self.truth))
        with client.get(r.json["pdf"]) as pdf:
            self.assertTrue(pdf.data.startswith(b"%PDF"))
        first_id = r.json["profile"]["id"]
        # The same handwriting again is recognised and not learned twice.
        form = {"text": "다시 변환", "handwriting": [(io.BytesIO(self.photo), "hand.jpg")], "transcript": FREEFORM_TEXT}
        r = client.post("/api/convert", data=form, content_type="multipart/form-data")
        self.assertEqual(r.json["profile"]["id"], first_id)
        # A saved handwriting style can be reused without uploading again.
        r = client.post("/api/convert", data={"text": "저장된 글씨체", "profile_id": first_id})
        self.assertEqual(r.status_code, 200, r.json)
        # Free-form handwriting without its text cannot be learned.
        r = client.post("/api/convert", data={"text": "x", "handwriting": [(io.BytesIO(self.photo), "hand.jpg")]},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("내용", r.json["error"])
        r = client.post("/api/convert", data={"text": "x"})
        self.assertEqual(r.status_code, 400)


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
