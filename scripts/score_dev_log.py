#!/usr/bin/env python3
"""CPU-only: recompute the OFFICIAL per-sample score from an existing dev per-sample log.

Replays the logged predicted calls and the gold calls in fresh ToolEnvs (train split) and applies the
published formula (full-state hash must match AND call F1). No model/GPU needed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate.data import import_tool_env, load_catalog, read_jsonl, resolve_split
from evaluate_dev import maximum_matching, official_score


def replay_hash(tool_env, split, sample_id, calls):
    env = tool_env(split.split_name, sample_id, root=split.env_root)
    for call in calls:
        env.call(call["name"], call.get("arguments", {}))
    return env.get_state_hash()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="dev per-sample JSONL (needs id, gold_calls, predicted_calls)")
    parser.add_argument("--data-root", default="data/raw/train+public_test")
    parser.add_argument("--out", help="optional JSONL with per-sample official scores")
    args = parser.parse_args()
    split = resolve_split(args.data_root, "train")
    tool_env = import_tool_env(split.split_dir)
    catalog = load_catalog(split.split_dir)
    index = {t["name"]: t for a in catalog["apps"] for t in a["tools"]}
    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    total, hash_ok_n, out = 0.0, 0, []
    for row in rows:
        gold, pred = row["gold_calls"], row["predicted_calls"]
        ok = replay_hash(tool_env, split, row["id"], pred) == replay_hash(tool_env, split, row["id"], gold)
        matched = maximum_matching(pred, gold, index)
        score = official_score(pred, gold, matched, ok)
        total += score
        hash_ok_n += ok
        out.append({"id": row["id"], "hash_match": ok, "matched": matched, "score": round(score, 4),
                    "gold": len(gold), "pred": len(pred), "end_reason": row.get("end_reason")})
    n = len(rows)
    print(f"samples={n} official_score≈{100 * total / n:.2f} full_hash_match={hash_ok_n / n:.3f}")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
