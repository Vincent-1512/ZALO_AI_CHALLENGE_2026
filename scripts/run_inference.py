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
    args = parser.parse_args()
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
    )


if __name__ == "__main__":
    main()
