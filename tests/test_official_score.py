import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "training"), str(Path(__file__).resolve().parents[1] / "src")]
from evaluate_dev import official_score


class OfficialScoreTests(unittest.TestCase):
    def test_empty_gold(self):
        self.assertEqual(official_score([], [], 0, True), 1.0)
        self.assertEqual(official_score([{"name": "x"}], [], 0, True), 0.0)

    def test_hash_mismatch_or_no_match_is_zero(self):
        self.assertEqual(official_score([{}], [{}], 1, False), 0.0)
        self.assertEqual(official_score([{}], [{}], 0, True), 0.0)

    def test_f1(self):
        # 2 gold, 3 predicted, 2 matched: P=2/3, R=1 -> F1=0.8
        self.assertAlmostEqual(official_score([{}] * 3, [{}] * 2, 2, True), 0.8)
        self.assertEqual(official_score([{}] * 2, [{}] * 2, 2, True), 1.0)


if __name__ == "__main__":
    unittest.main()
