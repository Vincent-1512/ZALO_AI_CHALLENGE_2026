import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_asr_train_data import build_train_records


class AsrNoisyDataTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "train_1", "audio": "a.wav", "audio_text": "Tắt đèn",
                    "context": {"installed_apps": {"a": {"tools": ["a.one", "a.two"]}}},
                    "history": [{"role": "user", "audio": "old.wav", "audio_text": "Mở cửa"}],
                    "label": {"tool_calls": [{"name": "a.one", "arguments": {}}],
                              "tool_results": [{"ok": True}]}}
        self.catalog = {"apps": [{"tools": [{"name": "a.one"}, {"name": "a.two"}]}]}
        self.asr = {"train_1": {"a.wav": "tắt dèn", "old.wav": "mở cữa"}}

    def test_fake_asr_is_used_deterministically_and_keeps_targets(self):
        records_a, info = build_train_records([self.row], self.catalog, self.asr, 1.0, 7, False)
        records_b, _ = build_train_records([self.row], self.catalog, self.asr, 1.0, 7, False)
        self.assertEqual(records_a, records_b)
        request = json.loads(records_a[0]["messages"][1]["content"])
        self.assertEqual(request["request"], "tắt dèn")
        self.assertEqual(request["history"][0]["text"], "mở cữa")
        self.assertEqual(records_a[0]["target"], '{"action":"call","name":"a.one","arguments":{}}')
        self.assertEqual(info["asr_selected_samples"], 1)

    def test_zero_probability_uses_gold_and_schema_order_adds_same_target_copy(self):
        records, info = build_train_records([self.row], self.catalog, self.asr, 0.0, 7, True)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["target"], records[1]["target"])
        self.assertEqual(records[2]["target"], records[3]["target"])
        request = json.loads(records[0]["messages"][1]["content"])
        self.assertEqual(request["request"], "Tắt đèn")
        self.assertEqual(info["schema_permutations"], True)

    def test_alt_asr_used_for_selected_fraction(self):
        alt = {"train_1": {"a.wav": "tắc đèn", "old.wav": "mờ cửa"}}
        records, info = build_train_records([self.row], self.catalog, self.asr, 1.0, 7, False, alt, 1.0)
        self.assertEqual(json.loads(records[0]["messages"][1]["content"])["request"], "tắc đèn")
        self.assertEqual(info["alt_asr_samples"], 1)
        records, info = build_train_records([self.row], self.catalog, self.asr, 1.0, 7, False, alt, 0.0)
        self.assertEqual(json.loads(records[0]["messages"][1]["content"])["request"], "tắt dèn")
        self.assertEqual(info["alt_asr_samples"], 0)
        with self.assertRaises(ValueError):
            build_train_records([self.row], self.catalog, self.asr, 1.0, 7, False, alt, 1.5)

    def test_probability_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            build_train_records([self.row], self.catalog, self.asr, 1.1)


if __name__ == "__main__":
    unittest.main()
