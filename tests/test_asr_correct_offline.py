import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval_asr_correct_offline import evaluate
from lifemate.models import _asr_model_fingerprint, asr_cache_key


class OfflineAsrCorrectionTests(unittest.TestCase):
    def test_reads_cached_transcript_and_reports_wer_before_after(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "data" / "train"
            split.mkdir(parents=True)
            audio = split / "a.wav"
            audio.write_bytes(b"audio bytes")
            sample = {"id": "train_1", "audio": "a.wav", "audio_text": "Gọi Sếp Dũng",
                "context": {"user_profile": {"name": "Sếp Dũng"}, "installed_apps": {}}, "history": []}
            (split / "train.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
            (split / "train_tools_catalog.json").write_text('{"apps": []}', encoding="utf-8")
            sft_dev = root / "dev.jsonl"
            sft_dev.write_text('{"sample_id":"train_1"}\n', encoding="utf-8")
            model = root / "asr"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            cache = root / "cache.json"
            cache.write_text(json.dumps({"meta": {"model_fingerprint": _asr_model_fingerprint(model)},
                "items": {asr_cache_key([audio]): ["Gọi Sếp Dung"]}}), encoding="utf-8")
            report = evaluate(root / "data", sft_dev, cache, model)
        self.assertEqual(report["samples"], 1)
        self.assertGreater(report["wer_before"], 0)
        self.assertEqual(report["wer_after"], 0)
        self.assertEqual(report["worsened_count"], 0)


if __name__ == "__main__":
    unittest.main()
