#!/usr/bin/env python3
"""Hold out whole apps to measure zero-shot use of unseen tools (public test adds 11 apps absent from train).

train.jsonl : samples whose gold AND history never touch a held-out app; held-out apps are removed from
              installed_apps so their schemas never appear in training prompts.
dev_zs.jsonl: samples whose gold calls use a held-out app (full app list, evaluated on unseen schemas).
dev_id.jsonl: ~10% of the remaining samples, same distribution as train (regression check).
Records are SFT records (sample_id/messages/target) like training/prepare_sft.py output."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate.data import load_catalog, read_jsonl, resolve_split
from prepare_sft import make_records, split_rows, write_jsonl


def apps_of_calls(calls: list[dict]) -> set[str]:
    return {call["name"].split(".")[0] for call in calls if "." in call.get("name", "")}


def gold_apps(row: dict) -> set[str]:
    return apps_of_calls(row["label"].get("tool_calls", [])) - {"sys"}


def history_apps(row: dict) -> set[str]:
    apps: set[str] = set()
    for turn in row.get("history", []):
        apps |= apps_of_calls(turn.get("tool_calls", []))
    return apps - {"sys"}


def choose_held_out(rows: list[dict], count: int, seed: int, min_samples: int) -> list[str]:
    usage = Counter(app for row in rows for app in gold_apps(row))
    eligible = sorted(app for app, n in usage.items() if n >= min_samples)
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} apps have >= {min_samples} gold-using samples")
    return sorted(random.Random(seed).sample(eligible, count))


def strip_apps(row: dict, apps: set[str]) -> dict:
    stripped = copy.deepcopy(row)
    installed = stripped["context"]["installed_apps"]
    stripped["context"]["installed_apps"] = {k: v for k, v in installed.items() if k not in apps}
    return stripped


def split_zero_shot(rows: list[dict], held_out: set[str], seed: int, id_dev_ratio: float = 0.1):
    dev_zs, remaining, dropped = [], [], 0
    for row in rows:
        if gold_apps(row) & held_out:
            dev_zs.append(row)
        elif history_apps(row) & held_out:
            dropped += 1   # history would leak/depend on the held-out schema
        else:
            remaining.append(row)
    parts = split_rows(remaining, seed, (1 - id_dev_ratio, id_dev_ratio, 1e-9))
    train = [strip_apps(row, held_out) for row in parts["train"]]
    dev_id = parts["dev"] + parts["holdout"]
    return train, dev_id, dev_zs, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/train+public_test")
    parser.add_argument("--output-dir", default="data/processed/zero_shot")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-held-out-apps", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--held-out-apps", nargs="*", help="Explicit app ids instead of random choice")
    args = parser.parse_args()
    split = resolve_split(args.data_root, "train")
    catalog = load_catalog(split.split_dir)
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    held_out = sorted(args.held_out_apps or choose_held_out(rows, args.num_held_out_apps, args.seed, args.min_samples))
    train, dev_id, dev_zs, dropped = split_zero_shot(rows, set(held_out), args.seed)
    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", make_records(train, catalog))
    write_jsonl(out / "dev_id.jsonl", make_records(dev_id, catalog))
    write_jsonl(out / "dev_zs.jsonl", make_records(dev_zs, catalog))
    ids = lambda part: hashlib.sha256("\n".join(sorted(r["id"] for r in part)).encode()).hexdigest()
    manifest = {"seed": args.seed, "held_out_apps": held_out,
                "samples": {"train": len(train), "dev_id": len(dev_id), "dev_zs": len(dev_zs), "dropped_history_leak": dropped},
                "id_sha256": {"train": ids(train), "dev_id": ids(dev_id), "dev_zs": ids(dev_zs)}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
