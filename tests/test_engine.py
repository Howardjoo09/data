import unittest

from engine import Edit, Player, stages


class ProgressTests(unittest.TestCase):
    def test_hangul_boundaries_and_compound_letters(self):
        self.assertEqual(stages("각"), ("ㄱ", "가", "각"))
        self.assertEqual(stages("가"), ("ㄱ", "가"))
        self.assertEqual(stages("꽜"), ("ㄲ", "꽈", "꽜"))
        self.assertEqual(stages("힣"), ("ㅎ", "히", "힣"))

    def test_document_reconstruction(self):
        original = "한글과 English 123!\n괜찮아 😀"
        p = Player(original)
        p.start()
        document = ""
        steps = 0
        while (edit := p.peek()) is not None:
            if edit.erase:
                document = document[:-edit.erase]
            document += edit.text
            p.commit()
            steps += 1
        self.assertEqual(document, original)
        self.assertEqual(steps, sum(len(stages(c)) for c in original))
        self.assertTrue(p.done)
        self.assertFalse(p.active)

    def test_pause_preserves_exact_stage(self):
        p = Player("각")
        p.start()
        p.commit()
        p.pause()
        self.assertIsNone(p.peek())
        self.assertEqual((p.index, p.stage), (0, 1))
        p.start()
        self.assertEqual(p.peek(), Edit(1, "가"))
        p.commit()
        p.pause()
        p.start()
        self.assertEqual(p.peek(), Edit(1, "각"))

    def test_manual_edit_is_never_erased(self):
        p = Player("각")
        p.start()
        p.commit()
        p.pause()
        p.detach()
        p.start()
        self.assertEqual(p.peek(), Edit(0, "가"))

    def test_no_progress_until_commit(self):
        p = Player("가")
        p.start()
        self.assertEqual(p.peek(), p.peek())
        self.assertEqual((p.index, p.stage), (0, 0))

    def test_normalization_and_reset(self):
        p = Player("가\r\nA")
        self.assertEqual(p.text, "가\nA")
        p.start()
        p.commit()
        p.reset("새")
        self.assertEqual((p.index, p.stage, p.active), (0, 0, False))
        empty = Player()
        empty.start()
        self.assertIsNone(empty.peek())


if __name__ == "__main__":
    unittest.main()
