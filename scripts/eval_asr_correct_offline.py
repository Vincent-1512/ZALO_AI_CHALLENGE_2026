#!/usr/bin/env python3
"""Measure context correction WER from an existing, model-fingerprinted ASR cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lifemate.asr_correct import correct_transcript
from lifemate.data import read_jsonl, resolve_audio_path, resolve_split
from lifemate.models import _asr_model_fingerprint, asr_cache_key


def _audio_refs(sample: dict) -> list[str]:
    history = [turn["audio"] for turn in sample.get("history", [])
               if turn.get("role") == "user" and turn.get("audio")]
    return [sample["audio"], *reversed(history)]


def _rates(pairs: list[tuple[str, str]]) -> dict[str, float]:
    from scripts.analyze_dev_log import error_rates

    if not pairs:
        return {"wer": 0.0, "cer": 0.0}
    rates = [error_rates(predicted, gold) for predicted, gold in pairs]
    return {"wer": sum(item[0] for item in rates) / len(rates),
            "cer": sum(item[1] for item in rates) / len(rates)}


def evaluate(data_root: Path, sft_dev: Path, cache_path: Path, model_path: Path, limit: int = 0) -> dict:
    split = resolve_split(data_root, "train")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_fingerprint = json.loads(json.dumps(_asr_model_fingerprint(model_path)))
    cache_fingerprint = cache.get("meta", {}).get("model_fingerprint")
    if cache_fingerprint != expected_fingerprint:
        raise ValueError("ASR cache model fingerprint does not match --model-path")
    cached_items = cache.get("items", {})
    dev_ids = {row["sample_id"] for row in read_jsonl(sft_dev)}
    catalog_path = split.split_dir / "train_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    samples = [row for row in read_jsonl(split.split_dir / "train.jsonl") if row["id"] in dev_ids]
    if limit > 0:
        samples = samples[:limit]
    before, after, worsened, details = [], [], [], []
    max_new_tokens = 128
    for sample in samples:
        paths = [resolve_audio_path(ref, split.split_dir) for ref in _audio_refs(sample)]
        key = asr_cache_key(paths, max_new_tokens)
        if key not in cached_items:
            raise KeyError(f"ASR cache miss for {sample['id']}; cache a matching evaluate_dev run first")
        raw = cached_items[key][0]
        corrected, changes = correct_transcript(raw, sample, catalog)
        gold = sample.get("audio_text", "")
        before.append((raw, gold)); after.append((corrected, gold))
        old_wer = _rates([(raw, gold)])["wer"]
        new_wer = _rates([(corrected, gold)])["wer"]
        detail = {"id": sample["id"], "asr_text": raw, "corrected_text": corrected,
                  "gold_text": gold, "wer_before": old_wer, "wer_after": new_wer,
                  "corrections": changes}
        details.append(detail)
        if new_wer > old_wer:
            worsened.append(detail)
    return {"samples": len(samples), "wer_before": _rates(before)["wer"],
            "wer_after": _rates(after)["wer"], "cer_before": _rates(before)["cer"],
            "cer_after": _rates(after)["cer"], "worsened_count": len(worsened),
            "worsened_samples": worsened, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/train+public_test"))
    parser.add_argument("--sft-dev", type=Path, default=Path("data/processed/sft/dev.jsonl"))
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, default=Path("models/PhoWhisper-medium"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(args.data_root, args.sft_dev, args.asr_cache, args.asr_model, args.limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
