from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import load_catalog, read_jsonl, resolve_split
from lifemate.prompts import action_json, build_messages


def normalized_request(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\W+", " ", text, flags=re.UNICODE).strip()


def split_rows(rows: list[dict], seed: int, ratios: tuple[float, float, float]):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalized_request(row.get("audio_text", ""))].append(row)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    total = len(rows)
    train_goal = round(total * ratios[0])
    dev_goal = round(total * ratios[1])
    partitions = {"train": [], "dev": [], "holdout": []}
    for group in shuffled:
        target = "train" if len(partitions["train"]) < train_goal else None
        if target is None:
            target = "dev" if len(partitions["dev"]) < dev_goal else "holdout"
        partitions[target].extend(group)
    return partitions


def make_records(rows: list[dict], catalog: dict) -> list[dict]:
    records = []
    for row in rows:
        calls = row["label"].get("tool_calls", [])
        results = row["label"].get("tool_results", [])
        if len(calls) != len(results):
            raise ValueError(f"Gold calls/results count differs for sample {row['id']}")
        for step in range(len(calls) + 1):
            messages = build_messages(
                row,
                row.get("audio_text", ""),
                catalog,
                prior_calls=calls[:step],
                prior_results=results[:step],
            )
            target = {"action": "stop"} if step == len(calls) else {
                "action": "call",
                "name": calls[step]["name"],
                "arguments": calls[step].get("arguments", {}),
            }
            records.append({"sample_id": row["id"], "messages": messages, "target": action_json(target)})
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_partitions(output_dir: Path, partitions: dict, catalog: dict, seed: int, ratios: tuple):
    manifest = {"seed": seed, "ratios": ratios, "samples": {}, "records": {}}
    for name, subset in partitions.items():
        records = make_records(subset, catalog)
        write_jsonl(output_dir / f"{name}.jsonl", records)
        manifest["samples"][name] = len(subset)
        manifest["records"][name] = len(records)
        ids = "\n".join(sorted(row["id"] for row in subset)).encode("utf-8")
        manifest[f"{name}_id_sha256"] = hashlib.sha256(ids).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train-only LifeMate next-action SFT records.")
    parser.add_argument("--data-root", default="data/raw/train+public_test")
    parser.add_argument("--output-dir", default="data/processed/sft")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    args = parser.parse_args()
    ratios = (args.train_ratio, args.dev_ratio, args.holdout_ratio)
    if abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) <= 0:
        raise ValueError("Split ratios must be positive and sum to 1")

    split = resolve_split(args.data_root, "train")
    catalog = load_catalog(split.split_dir)
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    partitions = split_rows(rows, args.seed, ratios)
    output_dir = Path(args.output_dir)
    manifest = write_partitions(output_dir, partitions, catalog, args.seed, ratios)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
