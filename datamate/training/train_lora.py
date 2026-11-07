from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from datamate.io import load_business_package, read_jsonl  # noqa: E402
from datamate.modeling import load_model, load_tokenizer, select_compute_dtype, set_seed  # noqa: E402
from datamate.query_plan import make_prompt  # noqa: E402
from datamate.runner import predict_records  # noqa: E402
from datamate.scoring import score_records  # noqa: E402


class TokenizedPlans(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.column_names = ["input_ids", "attention_mask", "labels", "length"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | str) -> dict[str, Any] | list[Any]:
        if isinstance(index, str):
            return [record[index] for record in self.records]
        return self.records[index]


class PlanCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in rows],
            padding=True, return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for index, row in enumerate(rows):
            labels[index, :len(row["labels"])] = torch.tensor(row["labels"], dtype=torch.long)
        batch["labels"] = labels
        return batch


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)


def tokenize_plan_records(rows: list[dict[str, Any]], data_root: Path, tokenizer: Any,
                          max_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_root = data_root / "train"
    package_cache = {}
    encoded, lengths, dropped = [], [], []
    for row in rows:
        business_id = row["business_id"]
        if business_id not in package_cache:
            package_cache[business_id] = load_business_package(split_root, business_id)
        package = package_cache[business_id]
        messages = make_prompt(row, package)
        prompt_ids = _chat_ids(tokenizer, messages, generation=True)
        target = json.dumps(row["plan"], ensure_ascii=False, separators=(",", ":"))
        full_messages = messages + [{"role": "assistant", "content": target}]
        input_ids = _chat_ids(tokenizer, full_messages, generation=False)
        if input_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prompt tokens are not a prefix of the target sequence")
        if len(input_ids) > max_length:
            dropped.append({"id": row["id"], "prompt_tokens": len(prompt_ids), "sequence_tokens": len(input_ids)})
            continue
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        encoded.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels,
                        "length": len(input_ids)})
        lengths.append(len(input_ids))
    if not encoded:
        raise ValueError(f"No verified plans fit max_sequence_length={max_length}; increase it or reduce prompt context")
    summary = {
        "accepted": len(encoded), "dropped_too_long": len(dropped),
        "p50_tokens": statistics.median(lengths), "p95_tokens": sorted(lengths)[min(len(lengths) - 1, math.ceil(.95 * len(lengths)) - 1)],
        "max_tokens": max(lengths), "dropped_ids": dropped,
    }
    return encoded, summary


class UtilityEarlyStopping(TrainerCallback):
    def __init__(self, dev_rows: list[dict[str, Any]], data_root: Path, tokenizer: Any,
                 output_dir: Path, max_new_tokens: int, max_context_tokens: int,
                 patience: int, min_delta: float):
        self.dev_rows = dev_rows
        self.split_root = data_root / "train"
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.patience = patience
        self.min_delta = min_delta
        self.package_cache = {}
        self.last_eval_seconds = 0.0
        self.metrics_path = output_dir / "dev_epoch_metrics.jsonl"
        self.best_path = output_dir / "best-adapter"
        self.best_score = float("-inf")
        self.stale = 0
        if self.metrics_path.exists():
            history = read_jsonl(self.metrics_path)
            for result in history:
                if result.get("utility_mean", float("-inf")) >= self.best_score + min_delta:
                    self.best_score = result["utility_mean"]
                    self.stale = 0
                else:
                    self.stale += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        was_training = model.training
        model.eval()
        started = time.monotonic()
        predictions, _ = predict_records(model, self.tokenizer, self.dev_rows, self.split_root,
                                         self.max_new_tokens, self.package_cache, self.max_context_tokens)
        report = score_records(predictions, self.dev_rows)
        self.last_eval_seconds = time.monotonic() - started
        report.update({"epoch": float(state.epoch or 0), "elapsed_seconds": round(self.last_eval_seconds, 2)})
        score = report["utility_mean"]
        if score >= self.best_score + self.min_delta or self.best_score == float("-inf"):
            self.best_score, self.stale = score, 0
            model.save_pretrained(self.best_path, safe_serialization=True)
            self.tokenizer.save_pretrained(self.best_path)
            report["best"] = True
        else:
            self.stale += 1
            report["best"] = False
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"dev epoch={state.epoch:.2f} utility={score:.4f} best={self.best_score:.4f} stale={self.stale}/{self.patience}")
        if was_training:
            model.train()
        if self.stale >= self.patience:
            control.should_training_stop = True
        return control


class TrainingETA(TrainerCallback):
    def __init__(self, utility_callback: UtilityEarlyStopping):
        self.utility_callback = utility_callback
        self.started = 0.0
        self.last_log_at = 0.0
        self.last_step = 0
        self.last_epoch = 0
        self.step_intervals: list[float] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_log_at = time.monotonic()
        self.last_step = state.global_step
        self.last_epoch = int(state.epoch or 0)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.monotonic()
        completed_steps = state.global_step - self.last_step
        if completed_steps > 0:
            elapsed = max(0.0, now - self.last_log_at)
            current_epoch = int(state.epoch or 0)
            if current_epoch > self.last_epoch:
                elapsed = max(0.0, elapsed - self.utility_callback.last_eval_seconds)
            self.step_intervals.append(elapsed / completed_steps)
            step_seconds = statistics.median(self.step_intervals[-5:])
            remaining_steps = max(0, state.max_steps - state.global_step)
            remaining_epochs = max(0, math.ceil(args.num_train_epochs - (state.epoch or 0)))
            train_remaining = remaining_steps * step_seconds
            evaluation_remaining = remaining_epochs * self.utility_callback.last_eval_seconds
            eta_minutes = (train_remaining + evaluation_remaining) / 60
            total_elapsed = (now - self.started) / 60
            print(f"training ETA≈{eta_minutes:.1f} min (measured step median={step_seconds:.2f}s; elapsed={total_elapsed:.1f} min)")
        self.last_log_at, self.last_step = now, state.global_step
        self.last_epoch = int(state.epoch or 0)
        return control


def _latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[1]))
    return str(checkpoints[-1]) if checkpoints else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_training(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(config["seed"])
    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    if args.max_dev_samples:
        dev_rows = dev_rows[:args.max_dev_samples]
    if not train_rows or not dev_rows:
        raise ValueError("Training and dev data must both be non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "train_sha256": _sha256(args.train_jsonl),
        "dev_sha256": _sha256(args.dev_jsonl),
        "config_sha256": _sha256(args.config),
        "max_train_samples": args.max_train_samples,
        "max_dev_samples": args.max_dev_samples,
        "max_sequence_length": args.max_sequence_length or config["max_sequence_length"],
        "epochs": args.epochs or config["training"]["epochs"],
        "micro_batch_size": args.micro_batch_size or config["training"]["micro_batch_size"],
        "gradient_accumulation_steps": args.gradient_accumulation_steps or config["training"]["gradient_accumulation_steps"],
        "max_steps": args.max_steps,
        "model": config["model_name"],
        "revision": config["model_revision"],
    }
    manifest_path = args.output_dir / "data_manifest.json"
    resume_requested = args.resume or bool(args.resume_from_checkpoint)
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume_requested:
            raise FileExistsError(f"Run already initialized at {args.output_dir}; use --resume or a new RUN_ID")
        if previous_manifest != run_manifest:
            raise ValueError("Resume inputs/config changed; use the original files or create a new RUN_ID")
    elif resume_requested:
        raise FileNotFoundError("Cannot resume: this output directory has no data_manifest.json")
    manifest_needs_write = not manifest_path.exists()
    tokenizer = load_tokenizer(config["model_name"], config["model_revision"])
    encoded, token_report = tokenize_plan_records(train_rows, args.data_root, tokenizer,
                                                  args.max_sequence_length or config["max_sequence_length"])
    (args.output_dir / "tokenization_report.json").write_text(
        json.dumps(token_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model = load_model(config["model_name"], config["model_revision"], quantized=True, training=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    training = config["training"]
    bf16 = select_compute_dtype() == torch.bfloat16
    max_steps = args.max_steps if args.max_steps > 0 else -1
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs or training["epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=args.micro_batch_size or training["micro_batch_size"],
        gradient_accumulation_steps=args.gradient_accumulation_steps or training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_grad_norm=training["max_grad_norm"],
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=training["logging_steps"],
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=training["save_total_limit"],
        save_safetensors=True,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=2,
        report_to=[],
        seed=config["seed"],
        data_seed=config["seed"],
        label_names=["labels"],
    )
    callback = UtilityEarlyStopping(dev_rows, args.data_root, tokenizer, args.output_dir,
                                    config["max_new_tokens"], config["max_context_tokens"],
                                    training["early_stopping_patience"],
                                    training["early_stopping_min_delta"])
    eta_callback = TrainingETA(callback)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedPlans(encoded),
        data_collator=PlanCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[callback, eta_callback],
    )
    if manifest_needs_write:
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resume = args.resume_from_checkpoint or (_latest_checkpoint(args.output_dir) if args.resume else None)
    trainer.train(resume_from_checkpoint=resume)
    last_path = args.output_dir / "last-adapter"
    trainer.model.save_pretrained(last_path, safe_serialization=True)
    tokenizer.save_pretrained(last_path)
    trainer.save_state()
    report = {"train_records": len(encoded), "tokenization": token_report,
              "best_utility_mean": callback.best_score, "best_adapter": str(callback.best_path),
              "last_adapter": str(last_path), "resumed_from": resume,
              "training_minutes": round((time.monotonic() - eta_callback.started) / 60, 2)}
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on executor-verified query plans")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/baseline.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/raw")
    parser.add_argument("--train-jsonl", type=Path, default=PROJECT / "data/processed/sft/train_augmented_plans.jsonl")
    parser.add_argument("--dev-jsonl", type=Path, default=PROJECT / "data/processed/sft/dev_questions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
