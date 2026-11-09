"""Tests for the text clean-up heuristics, built from real failures seen in conversations.

    cd desktop && python -m unittest discover tests      (pure Python: no torch, no models)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textfix import (COMPRESSION_MAX, clean_translation, collapse_repeats,  # noqa: E402
                     compression_ratio, is_hallucination)


class CollapseRepeats(unittest.TestCase):
    def test_whisper_sentence_loop(self):
        raw = ("Mais, alors, on ne peut pas le Internet, on ne peut pas le Internet. "
               + "On ne peut pas le Internet. " * 5)
        self.assertGreater(compression_ratio(raw), COMPRESSION_MAX)
        self.assertEqual(collapse_repeats(raw), "Mais, alors, on ne peut pas le Internet.")

    def test_phrase_loop(self):
        raw = "De akkor nem használhatjuk az internetet, " + "nem használhatjuk az internetet, " * 4
        self.assertEqual(collapse_repeats(raw.strip(", ") + "."), "De akkor nem használhatjuk az internetet.")

    def test_keeps_normal_text(self):
        for t in ["Le 15 Pro. Non, c'est le 13.", "Oui, oui, oui.", "Il n'y a pas de choix en fait."]:
            self.assertEqual(collapse_repeats(t), t)


class CleanTranslation(unittest.TestCase):
    def test_subtitle_dash(self):
        self.assertEqual(clean_translation("- Van iPhone-otok?"), "Van iPhone-otok?")
        self.assertEqual(clean_translation("- Szia!"), "Szia!")

    def test_two_alternatives(self):
        self.assertEqual(clean_translation("- Köszönjük. - Köszönöm."), "Köszönjük.")

    def test_keeps_real_dashes(self):
        for t in ["Ez egy 2017-16-os modell - azt hiszem, nem tudom.", "Igen - nem."]:
            self.assertEqual(clean_translation(t), t)


class Hallucinations(unittest.TestCase):
    def test_subtitle_credits(self):
        self.assertTrue(is_hallucination("Sous-titrage Société Radio-Canada"))
        self.assertTrue(is_hallucination("Köszönöm a figyelmet!"))
        self.assertFalse(is_hallucination("Merci."))


if __name__ == "__main__":
    unittest.main()
