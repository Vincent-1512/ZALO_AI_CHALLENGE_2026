#!/usr/bin/env python3
"""Report prompt+response SFT lengths, using a local tokenizer when available."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lifemate.data import read_jsonl


def _serialized(record: dict) -> str:
    messages = "".join(
        f"<{message['role']}>{message['content']}</{message['role']}>"
        for message in record["messages"]
    )
    return messages + "<assistant>" + record["target"] + "</assistant>"


def measure_lengths(records: list[dict], tokenizer=None) -> tuple[list[int], str]:
    lengths = []
    if tokenizer is not None:
        try:
            for row in records:
                sequence = row["messages"] + [{"role": "assistant", "content": row["target"]}]
                encoded = tokenizer.apply_chat_template(sequence, tokenize=True, add_generation_prompt=False)
                lengths.append(len(encoded[0]) if hasattr(encoded, "shape") and len(encoded.shape) > 1 else len(encoded))
            return lengths, "tokenizer"
        except Exception:
            lengths.clear()
    # Approximate only: Unicode character count / 4; not an exact tokenizer estimate.
    return [math.ceil(len(_serialized(row)) / 4) for row in records], "character estimate (~4 chars/token)"


def summarize(lengths: list[int], method: str, split: str) -> dict:
    ordered = sorted(lengths)
    if not ordered:
        return {"split": split, "samples": 0, "method": method, "percentiles": {},
                "over_limit_percent": {str(limit): 0.0 for limit in (2048, 4096, 8192)},
                "recommendation": "No records found."}
    percentile = lambda q: ordered[min(round((len(ordered) - 1) * q), len(ordered) - 1)]
    overflow = {str(limit): sum(value > limit for value in ordered) / len(ordered) * 100
                for limit in (2048, 4096, 8192)}
    if overflow["8192"]:
        advice = "Some sequences exceed 8192; inspect them and choose an explicit context policy before reducing max length."
    elif ordered[-1] > percentile(0.5) * 2:
        advice = "Length variance may cause padding; group_by_length is a candidate to benchmark, with no speed gain assumed."
    else:
        advice = "These lengths alone do not justify reducing max length; test group_by_length only if padding is a measured bottleneck."
    return {"split": split, "samples": len(ordered), "method": method,
            "percentiles": {"p50": percentile(0.5), "p90": percentile(0.9),
                            "p99": percentile(0.99), "max": ordered[-1]},
            "over_limit_percent": overflow, "recommendation": advice}


def load_local_tokenizer(model_path: Path):
    if not model_path.is_dir():
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    except Exception as error:
        print(f"Local tokenizer unavailable ({type(error).__name__}); using character estimate.", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/sft/train.jsonl"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("data/processed/sft/dev.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tokenizer = load_local_tokenizer(args.model_path)
    reports = []
    for name, path in (("train", args.train_jsonl), ("dev", args.dev_jsonl)):
        rows = list(read_jsonl(path))
        lengths, method = measure_lengths(rows, tokenizer)
        reports.append(summarize(lengths, method, name))
    result = {"reports": reports, "note": "No training run was performed; estimated lengths do not prove a speed improvement."}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
