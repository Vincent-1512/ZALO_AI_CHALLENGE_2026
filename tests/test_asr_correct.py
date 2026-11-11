import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lifemate.asr_correct import correct_transcript, context_vocabulary


class AsrCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "context": {
                "user_profile": {"name": "Sếp Dũng", "city": "Đà Nẵng"},
                "installed_apps": {"calendar": {"tools": ["calendar.create_event"]}},
            }
        }
        self.catalog = {"apps": [{"id": "calendar", "name_vi": "Lịch",
            "tools": [{"name": "calendar.create_event", "description": "Tạo sự kiện",
                "parameters": {"properties": {"status": {"enum": ["confirmed", "cancelled"]}}}}]}]}

    def test_repairs_missing_vietnamese_diacritic_from_context(self):
        corrected, changes = correct_transcript("Gọi cho Sếp Dung", self.sample, self.catalog)
        self.assertEqual(corrected, "Gọi cho Sếp Dũng")
        self.assertEqual(changes[0]["before"], "Sếp Dung")
        self.assertEqual(changes[0]["after"], "Sếp Dũng")

    def test_does_not_correct_names_absent_from_context(self):
        text = "Nhắn cho Anh Minh"
        corrected, changes = correct_transcript(text, self.sample, self.catalog)
        self.assertEqual(corrected, text)
        self.assertEqual(changes, [])

    def test_exact_context_phrase_is_unchanged(self):
        text = "Gọi cho Sếp Dũng"
        corrected, changes = correct_transcript(text, self.sample, self.catalog)
        self.assertEqual(corrected, text)
        self.assertEqual(changes, [])

    def test_vocabulary_includes_context_tools_and_enum_values(self):
        terms = context_vocabulary(self.sample, self.catalog)
        self.assertIn("Sếp Dũng", terms)
        self.assertIn("calendar.create_event", terms)
        self.assertIn("cancelled", terms)
        self.assertIn("status", terms)


if __name__ == "__main__":
    unittest.main()
