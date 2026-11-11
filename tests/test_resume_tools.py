import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate import runner
from lifemate.models import SpeechRecognizer, _asr_model_fingerprint
import train_lora


class FakeEnv:
    def __init__(self, ok=True):
        self.ok = ok

    def validate(self, name, arguments):
        return {"ok": self.ok, "error": None if self.ok else "bad"}


class EndReasonTests(unittest.TestCase):
    def step(self, actions, env_ok=True):
        queue = list(actions)
        profile = {"policy_seconds": 0.0, "tool_seconds": 0.0}
        with mock.patch.object(runner, "_next_action", side_effect=lambda *a, **k: queue.pop(0)):
            result = runner._action_for_step(None, FakeEnv(env_ok), {}, "", {}, {}, [], profile)
        return result, profile

    def test_stop_vs_parse_failure_are_distinct(self):
        self.assertEqual(self.step([{"action": "stop"}])[1]["end_reason"], "stop")
        self.assertEqual(self.step([None])[1]["end_reason"], "parse_failure")

    def test_valid_call_is_returned_unchanged(self):
        call = {"action": "call", "name": "a.b", "arguments": {}}
        result, profile = self.step([call])
        self.assertEqual(result, call)
        self.assertNotIn("end_reason", profile)

    def test_schema_rejection_counts_retry_and_reason(self):
        call = {"action": "call", "name": "a.b", "arguments": {}}
        result, profile = self.step([call, call], env_ok=False)
        self.assertIsNone(result)
        self.assertEqual(profile["schema_retries"], 1)
        self.assertEqual(profile["end_reason"], "schema_rejected_after_retry")

    def test_profile_none_keeps_old_behaviour(self):
        queue = [{"action": "stop"}]
        with mock.patch.object(runner, "_next_action", side_effect=lambda *a, **k: queue.pop(0)):
            self.assertIsNone(runner._action_for_step(None, FakeEnv(), {}, "", {}, {}, [], None))


class AsrCacheTests(unittest.TestCase):
    def make(self, cache_path):
        asr = SpeechRecognizer.__new__(SpeechRecognizer)
        asr._cache_path = str(cache_path)
        asr._cache_meta = {"model": "m"}
        asr._cache = {}
        asr.calls = 0

        def fake(paths, max_new_tokens):
            asr.calls += 1
            return [f"text-{Path(p).name}" for p in paths]

        asr._transcribe_uncached = fake
        return asr

    def test_second_identical_call_hits_cache_and_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"12345")
            cache = Path(folder) / "cache" / "asr.json"
            asr = self.make(cache)
            self.assertEqual(asr.transcribe_many([audio]), ["text-a.wav"])
            self.assertEqual(asr.transcribe_many([audio]), ["text-a.wav"])
            self.assertEqual(asr.calls, 1)
            stored = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(stored["meta"], {"model": "m"})
            self.assertEqual(len(stored["items"]), 1)

    def test_same_size_audio_content_change_misses_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "a.wav"
            audio.write_bytes(b"12345")
            asr = self.make(Path(folder) / "cache.json")
            asr.transcribe_many([audio])
            audio.write_bytes(b"abcde")  # same byte length, different audio content
            asr.transcribe_many([audio])
            self.assertEqual(asr.calls, 2)

    def test_model_fingerprint_tracks_config_and_weight_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            (model / "preprocessor_config.json").write_text("{\"x\":1}")
            (model / "model.safetensors").write_bytes(b"weights")
            first = _asr_model_fingerprint(model)
            (model / "config.json").write_text("{\"x\":1}")
            second = _asr_model_fingerprint(model)
            self.assertNotEqual(first, second)
            self.assertEqual(second["weights"], [("model.safetensors", 7)])

    def test_cache_disabled_always_calls_model(self):
        asr = SpeechRecognizer.__new__(SpeechRecognizer)
        asr._cache_path = None
        asr.calls = 0

        def fake(paths, max_new_tokens):
            asr.calls += 1
            return ["x"]

        asr._transcribe_uncached = fake
        asr.transcribe_many([Path("a")]); asr.transcribe_many([Path("a")])
        self.assertEqual(asr.calls, 2)


class TrainArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        with mock.patch.object(sys, "argv", ["train_lora.py", *extra]):
            return train_lora.parse_args()

    def test_defaults_are_unchanged(self):
        args = self.parse()
        self.assertEqual((args.max_steps, args.save_total_limit, args.no_load_best, args.warm_start_adapter),
                         (0, 3, False, None))
        self.assertFalse(args.no_gradient_checkpointing)

    def test_no_gradient_checkpointing_flag_parses(self):
        self.assertTrue(self.parse("--no-gradient-checkpointing").no_gradient_checkpointing)

    def test_warm_start_and_resume_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parse("--warm-start-adapter", "a", "--resume-from-checkpoint", "b")

    def test_new_flags_parse(self):
        args = self.parse("--max-steps", "508", "--save-total-limit", "5", "--no-load-best")
        self.assertEqual((args.max_steps, args.save_total_limit, args.no_load_best), (508, 5, True))


if __name__ == "__main__":
    unittest.main()
