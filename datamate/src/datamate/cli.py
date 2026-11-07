from __future__ import annotations

import argparse
import json
from pathlib import Path

from .augment import augment_verified_plans
from .modeling import load_model, load_tokenizer, set_seed
from .plan_labels import build_verified_plans
from .prepare_data import prepare_data
from .runner import evaluate_split, run_public


def _runtime(config: dict, adapter: str | None, model_path: str | None):
    set_seed(config["seed"])
    tokenizer = load_tokenizer(config["model_name"], config["model_revision"], model_path)
    model = load_model(config["model_name"], config["model_revision"], quantized=True, local_path=model_path)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DataMate train/evaluate/predict utilities")
    parser.add_argument("action", choices=["prepare", "build-plans", "augment", "evaluate", "public-predict"])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/sft"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/current"))
    parser.add_argument("--output", type=Path, default=Path("result/submission.jsonl"))
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.action == "prepare":
        result = prepare_data(args.data_root, args.processed_dir, config["dev_business_ids"])
    elif args.action == "augment":
        result = augment_verified_plans(args.processed_dir / "train_verified_plans.jsonl",
                                        args.processed_dir / "train_augmented_plans.jsonl")
    elif args.action in {"build-plans", "evaluate", "public-predict"}:
        model, tokenizer = _runtime(config, args.adapter, args.model_path)
        if args.action == "build-plans":
            result = build_verified_plans(args.data_root, args.processed_dir, model, tokenizer,
                                          config["model_name"], config["model_revision"], args.limit, config["seed"],
                                          config["max_new_tokens"], config["max_context_tokens"])
        elif args.action == "evaluate":
            result = evaluate_split(model, tokenizer, args.processed_dir / "dev_questions.jsonl",
                                    args.data_root / "train", args.output_dir, args.limit,
                                    config["max_new_tokens"], config["max_context_tokens"])
        else:
            result = run_public(model, tokenizer, args.data_root, args.output, args.limit,
                                config["max_new_tokens"], config["max_context_tokens"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
