import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.length_stats import measure_lengths, summarize


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return list("".join(item["content"] for item in messages))


class LengthStatsTests(unittest.TestCase):
    def test_uses_local_tokenizer_and_reports_limits(self):
        rows = [{"messages": [{"role": "user", "content": "abc"}], "target": "de"}]
        lengths, method = measure_lengths(rows, CharacterTokenizer())
        report = summarize(lengths, method, "train")
        self.assertEqual(method, "tokenizer")
        self.assertEqual(report["percentiles"]["max"], 5)
        self.assertEqual(report["over_limit_percent"]["2048"], 0.0)

    def test_falls_back_to_character_estimate(self):
        rows = [{"messages": [{"role": "user", "content": "12345678"}], "target": "1234"}]
        lengths, method = measure_lengths(rows)
        self.assertEqual(lengths, [12])
        self.assertIn("character estimate", method)

    def test_warns_when_sequences_exceed_context(self):
        report = summarize([100, 9000], "tokenizer", "train")
        self.assertEqual(report["over_limit_percent"]["8192"], 50.0)
        self.assertIn("exceed 8192", report["recommendation"])


if __name__ == "__main__":
    unittest.main()
