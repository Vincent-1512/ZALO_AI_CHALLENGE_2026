from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DataMate inference entrypoint")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"),
                        help="Mounted split directory containing questions.jsonl, calendar.csv and data/")
    parser.add_argument("--base-model", type=Path, required=True,
                        help="Local Qwen base model directory; network access is disabled")
    parser.add_argument("--adapter", type=Path, required=True, help="Selected PEFT adapter directory")
    parser.add_argument("--output", type=Path, default=Path("result/submission.jsonl"))
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/baseline.json")
    args = parser.parse_args()
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Local base model directory does not exist: {args.base_model}")
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"PEFT adapter directory does not exist: {args.adapter}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from datamate.cli import main as run_cli
    run_cli([
        "public-predict", "--data-root", str(args.data_dir), "--config", str(args.config),
        "--output", str(args.output), "--adapter", str(args.adapter),
        "--model-path", str(args.base_model),
    ])


if __name__ == "__main__":
    main()
