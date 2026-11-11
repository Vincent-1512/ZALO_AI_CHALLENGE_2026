import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifemate import runner
from lifemate.prompts import build_messages


class InferenceDefaultsTests(unittest.TestCase):
    def test_default_prompt_fixture_is_unchanged(self):
        sample = {"id": "x", "audio": "a.wav", "context": {"timezone": "UTC",
                  "installed_apps": {"calendar": {"tools": ["calendar.create_event"]}}}, "history": []}
        catalog = {"apps": [{"id": "calendar", "name_vi": "Calendar", "description": "",
                   "tools": [{"name": "calendar.create_event", "description": "Create", "kind": "write",
                              "parameters": {}}]}]}
        messages = build_messages(sample, "make event", catalog)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload, {
            "request": "make event", "context": {"timezone": "UTC"},
            "available_tools": [
                {"name": "calendar.create_event", "description": "Create", "kind": "write", "parameters": {}},
                {"name": "sys.ask_missing_info"},
            ], "history": [], "steps_this_request": [],
        })

    def test_default_submission_jsonl_shape_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "public_test.jsonl"
            source.write_text('{"id":"p1","audio":"a.wav"}\n', encoding="utf-8")
            output = root / "submission.jsonl"
            config = {"limits": {"max_tool_steps": 8}}
            with mock.patch.object(runner, "infer_one", return_value=([{"name": "a.b", "arguments": {}}], "hash")):
                runner._write_predictions(source, output, SimpleNamespace(), None, {}, None, None, config)
            self.assertEqual(output.read_text(encoding="utf-8"),
                '{"id": "p1", "tool_calls": [{"name": "a.b", "arguments": {}}], "state_hash": "hash"}\n')
            self.assertEqual(list(root.glob("*.log")), [])

    def test_asr_correction_is_not_called_when_flag_is_off(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "a.wav"
            audio.write_bytes(b"audio")
            sample = {"audio": "a.wav", "history": []}
            asr = SimpleNamespace(transcribe_many=lambda paths: ["Sếp Dung"])
            with mock.patch.object(runner, "correct_transcript") as correction:
                result = runner._transcribe_sample(sample, Path(directory), asr)
            self.assertEqual(result, ("Sếp Dung", {"a.wav": "Sếp Dung"}))
            correction.assert_not_called()


if __name__ == "__main__":
    unittest.main()
