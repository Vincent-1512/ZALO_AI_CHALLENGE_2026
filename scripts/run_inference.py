from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.runner import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline LifeMate inference baseline.")
    parser.add_argument("--data", default="data/raw/train+public_test")
    parser.add_argument("--split", default="public_test")
    parser.add_argument("--output", default="result/submission.jsonl")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--asr-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--adapter")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--asr-cache", help="Optional JSON file that memoises ASR transcripts across runs")
    parser.add_argument("--asr-correct", action="store_true", help="Repair ASR spans from current sample context (optional)")
    parser.add_argument("--num-samples", type=int, default=1, help="Best-of-k action candidates per step; default keeps greedy inference")
    parser.add_argument("--temperature", type=float, help="Sampling temperature; valid only with --num-samples > 1")
    parser.add_argument("--per-sample-log", help="Optional JSONL with ASR repairs and candidate diagnostics")
    args = parser.parse_args()
    if args.asr_cache:
        import os
        os.environ["LIFEMATE_ASR_CACHE"] = str(Path(args.asr_cache).resolve())
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.temperature is not None and args.num_samples == 1:
        parser.error("--temperature is only used with --num-samples > 1")
    if args.temperature is not None and args.temperature <= 0:
        parser.error("--temperature must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_inference(
        args.data,
        args.split,
        args.output,
        config,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        adapter_path=args.adapter,
        allow_cpu=args.allow_cpu,
        asr_correct=args.asr_correct,
        num_samples=args.num_samples,
        temperature=args.temperature,
        per_sample_log=args.per_sample_log,
    )


if __name__ == "__main__":
    main()
