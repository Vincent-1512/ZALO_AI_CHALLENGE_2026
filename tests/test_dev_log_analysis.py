import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_dev_log import analyze, error_rates


class DevLogAnalysisTests(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation_but_keeps_accents(self):
        self.assertEqual(error_rates("Xin, Chào!", "xin chào"), (0.0, 0.0))
        self.assertGreater(error_rates("xin chao", "xin chào")[0], 0)

    def test_analyzes_actual_top_level_log_schema(self):
        row = {"id": "dev_1", "state_match": False, "end_reason": "parse_failure",
               "asr_text": "bat den", "gold_text": "bật đèn", "schema_retries": 0,
               "gold_calls": [{"name": "home.light", "arguments": {"on": True}}],
               "predicted_calls": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = analyze(path, 0.5)
        result = report["samples_detail"][0]
        self.assertIn("PARSE_FAIL", result["labels"])
        self.assertIn("ASR_ERROR", result["labels"])
        self.assertIn("MISSING_CALL", result["labels"])
        self.assertEqual(report["error_sample_count"], 1)
        self.assertEqual(report["asr_error_samples"], 1)

    def test_wrong_order_classification(self):
        from scripts.analyze_dev_log import classify

        record = {"asr_text": "", "gold_text": "", "end_reason": "stop",
                  "gold_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}],
                  "predicted_calls": [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]}
        self.assertEqual(classify(record, 0.5), ["WRONG_ORDER"])

    def test_classifies_missing_and_extra_calls_separately(self):
        from scripts.analyze_dev_log import classify

        record = {"gold_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}],
                 "predicted_calls": [{"name": "a", "arguments": {}}, {"name": "c", "arguments": {}}]}
        self.assertEqual(classify(record, 0.5), ["WRONG_TOOL_NAME"])
        record["predicted_calls"] = [{"name": "a", "arguments": {}}]
        self.assertEqual(classify(record, 0.5), ["MISSING_CALL"])
        record["gold_calls"] = [{"name": "a", "arguments": {}}]
        record["predicted_calls"] = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
        self.assertEqual(classify(record, 0.5), ["EXTRA_CALL"])

    def test_classifies_arguments_only_when_tool_matches(self):
        from scripts.analyze_dev_log import classify

        record = {"gold_calls": [{"name": "a", "arguments": {"x": 1}}],
                  "predicted_calls": [{"name": "a", "arguments": {"x": 2}}]}
        self.assertEqual(classify(record, 0.5), ["WRONG_ARGS"])

    def test_successful_schema_retry_is_still_reported(self):
        from scripts.analyze_dev_log import classify

        record = {"end_reason": "stop", "schema_retries": 1,
                  "gold_calls": [], "predicted_calls": []}
        self.assertEqual(classify(record, 0.5), ["SCHEMA_REJECTED"])


if __name__ == "__main__":
    unittest.main()
