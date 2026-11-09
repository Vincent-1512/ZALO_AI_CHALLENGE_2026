#!/usr/bin/env python3
"""Prepare SFT data with deterministic ASR-noisy requests; does not train a model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate.data import read_jsonl, resolve_audio_path, resolve_split
from lifemate.models import SpeechRecognizer
from prepare_sft import make_records, split_rows, write_jsonl


def _rng(seed: int, sample_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _schema_variant(record: dict, seed: int) -> dict | None:
    variant = copy.deepcopy(record)
    payload = json.loads(variant["messages"][1]["content"])
    tools = payload.get("available_tools", [])
    if len(tools) <= 1:
        return None
    rng = _rng(seed, str(record["sample_id"]))
    shuffled = list(tools)
    rng.shuffle(shuffled)
    if shuffled == tools:
        shuffled = shuffled[1:] + shuffled[:1]
    payload["available_tools"] = shuffled
    variant["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return variant


def build_train_records(rows: list[dict], catalog: dict, asr_by_sample: dict[str, dict[str, str]],
                        probability: float = 0.5, seed: int = 2026,
                        schema_permutations: bool = True) -> tuple[list[dict], dict]:
    """Build gold/ASR request variants and optional same-label schema-order copies."""
    if not 0 <= probability <= 1:
        raise ValueError("ASR probability must be between 0 and 1")
    records, selected, variants = [], 0, 0
    for row in rows:
        prepared = copy.deepcopy(row)
        if _rng(seed, str(row["id"])).random() < probability:
            transcript_map = asr_by_sample.get(row["id"], {})
            if row.get("audio") not in transcript_map:
                raise KeyError(f"Missing ASR transcript for {row['id']} current audio")
            prepared["audio_text"] = transcript_map[row["audio"]]
            for turn in prepared.get("history", []):
                if turn.get("role") == "user" and turn.get("audio") in transcript_map:
                    turn["audio_text"] = transcript_map[turn["audio"]]
            selected += 1
        for record in make_records([prepared], catalog):
            records.append(record)
            if schema_permutations:
                variant = _schema_variant(record, seed)
                if variant is not None:
                    records.append(variant)
                    variants += 1
    return records, {"samples": len(rows), "asr_selected_samples": selected,
                     "records": len(records), "schema_variant_records": variants,
                     "schema_permutations": schema_permutations}


def _transcribe_rows(rows: list[dict], split_dir: Path, asr: SpeechRecognizer) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        refs = [row["audio"], *reversed([turn["audio"] for turn in row.get("history", [])
                                        if turn.get("role") == "user" and turn.get("audio")])]
        paths = [resolve_audio_path(ref, split_dir) for ref in refs]
        output[row["id"]] = dict(zip(refs, asr.transcribe_many(paths)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/train+public_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/asr_noisy_sft"))
    parser.add_argument("--asr-model", type=Path, default=Path("models/PhoWhisper-medium"))
    parser.add_argument("--asr-cache", type=Path, required=True)
    parser.add_argument("--asr-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-schema-permutation", action="store_true")
    parser.add_argument("--include-dev-holdout", action="store_true",
                        help="Final-fit mode: train on train+dev+holdout rows (dev/holdout metrics become optimistic)")
    args = parser.parse_args()
    if not 0 <= args.asr_probability <= 1:
        parser.error("--asr-probability must be between 0 and 1")
    os.environ["LIFEMATE_ASR_CACHE"] = str(args.asr_cache.resolve())
    split = resolve_split(args.data_root, "train")
    catalog = json.loads((split.split_dir / "train_tools_catalog.json").read_text(encoding="utf-8"))
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    partitions = split_rows(rows, args.seed, (0.8, 0.1, 0.1))
    fit_rows = list(partitions["train"])
    if args.include_dev_holdout:
        fit_rows += list(partitions["dev"]) + list(partitions["holdout"])
    if args.asr_probability > 0:
        asr = SpeechRecognizer(args.asr_model)
        asr_by_sample = _transcribe_rows(fit_rows, split.split_dir, asr)
    else:
        asr_by_sample = {}
    train_records, info = build_train_records(
        fit_rows, catalog, asr_by_sample, args.asr_probability,
        args.seed, not args.no_schema_permutation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_records)
    manifest = {"seed": args.seed, "asr_probability": args.asr_probability,
                "train": info, "dev_samples": len(partitions["dev"]),
                "holdout_samples": len(partitions["holdout"]),
                "include_dev_holdout": args.include_dev_holdout,
                "dev_and_holdout_use_gold_transcripts": True,
                "schema_permutation_note": "Each eligible trajectory row gets one same-target copy, matching notebook order permutation."}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
