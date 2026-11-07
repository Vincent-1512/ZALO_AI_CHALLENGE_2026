from __future__ import annotations

import argparse
import json
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode_record(record: dict, tokenizer, max_length: int) -> dict:
    if not 0 < max_length <= 16384:
        raise ValueError("Sequence length must be between 1 and 16384")
    messages = [dict(message) for message in record["messages"]]
    payload = json.loads(messages[1]["content"])
    while True:
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        response = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": record["target"]}],
            tokenize=True, add_generation_prompt=False,
        )
        if len(response) <= max_length:
            break
        if not payload.get("history"):
            raise ValueError(
                f"{record.get('sample_id')}: schema/request/steps + full target require "
                f"{len(response)} tokens after removing history; budget={max_length}"
            )
        payload["history"].pop(0)
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if response[:len(prompt_ids)] != prompt_ids or len(response) <= len(prompt_ids):
        raise ValueError(f"{record.get('sample_id')}: incompatible tokenizer assistant prefix")
    return {
        "input_ids": response, "attention_mask": [1] * len(response),
        "labels": [-100] * len(prompt_ids) + response[len(prompt_ids):],
    }


def minimum_record_length(record, tokenizer):
    messages = [dict(message) for message in record["messages"]]
    payload = json.loads(messages[1]["content"])
    payload["history"] = []
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return len(tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["target"]}],
        tokenize=True, add_generation_prompt=False,
    ))


@dataclass
class CausalPaddingCollator:
    tokenizer: object

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        inputs, masks, labels = [], [], []
        for feature in features:
            padding = width - len(feature["input_ids"])
            inputs.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the LifeMate Qwen next-action policy.")
    parser.add_argument("--train-jsonl", default="data/processed/sft/train.jsonl")
    parser.add_argument("--dev-jsonl", default="data/processed/sft/dev.jsonl")
    parser.add_argument("--model-path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="models/lifemate-qwen3-4b-lora")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.001)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-dev-samples", type=int, default=0)
    parser.add_argument("--auto-context", action="store_true", help="Choose context from actual train/dev token lengths")
    parser.add_argument("--preflight-only", action="store_true", help="Check data/tokenizer without loading model or CUDA")
    parser.add_argument("--preflight-report")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--save-steps", type=int, default=0,
                        help="Optional maximum checkpoint interval; default derives it from evals per epoch")
    parser.add_argument("--longest-first", action="store_true", help="Smoke: exercise longest training record first")
    args = parser.parse_args()
    if not 0 < args.max_sequence_length <= 16384:
        parser.error("max sequence length must be in 1..16384")
    if min(args.gradient_accumulation_steps, args.lora_rank,
           args.lora_alpha, args.evals_per_epoch, args.early_stopping_patience) < 1:
        parser.error("save, accumulation, LoRA, eval and patience values must be positive")
    if args.save_steps < 0:
        parser.error("save steps must be zero (automatic) or positive")
    if not 0 <= args.lora_dropout < 1 or args.early_stopping_threshold < 0:
        parser.error("LoRA dropout must be in [0, 1) and early-stopping threshold non-negative")
    return args


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def limit_sample_groups(records: list[dict], sample_limit: int) -> list[dict]:
    if sample_limit <= 0:
        return records
    selected = list(dict.fromkeys(row["sample_id"] for row in records))[:sample_limit]
    allowed = set(selected)
    limited = [row for row in records if row["sample_id"] in allowed]
    print(f"Using {len(selected)} samples and {len(limited)} next-action records")
    return limited


def load_encoded_data(args, tokenizer):
    rows = {
        "train": limit_sample_groups(load_records(Path(args.train_jsonl)), args.limit_train_samples),
        "dev": limit_sample_groups(load_records(Path(args.dev_jsonl)), args.limit_dev_samples),
    }
    if not all(rows.values()):
        raise ValueError("Both train and dev records must be non-empty")
    report = {"context_ceiling": args.max_sequence_length, "splits": {}}
    minimums = []
    for split, records in rows.items():
        lengths = [minimum_record_length(row, tokenizer) for row in records]
        minimums.extend(lengths)
        report["splits"][split] = {
            "samples": len({row["sample_id"] for row in records}), "records": len(records),
            "max_tokens_without_history": max(lengths),
            "data_sha256": hashlib.sha256(Path(getattr(args, f"{split}_jsonl")).read_bytes()).hexdigest(),
        }
    if args.auto_context:
        candidates = sorted({2048, 4096, 6144, 8192, 12288, 16384, args.max_sequence_length})
        selected = next((n for n in candidates if max(minimums) <= n <= args.max_sequence_length), None)
        if selected is None:
            raise ValueError(f"Full schema/request/steps + target need up to {max(minimums)} tokens; "
                             f"ceiling={args.max_sequence_length}. No records were dropped.")
        args.max_sequence_length = selected
    report["selected_context"] = args.max_sequence_length
    encoded = {split: [encode_record(row, tokenizer, args.max_sequence_length) for row in records]
               for split, records in rows.items()}
    for split, records in encoded.items():
        lengths = sorted(len(row["input_ids"]) for row in records)
        report["splits"][split].update(max_encoded_tokens=max(lengths),
            p95_encoded_tokens=lengths[math.ceil(len(lengths) * .95) - 1])
    if args.longest_first:
        encoded["train"].sort(key=lambda row: len(row["input_ids"]), reverse=True)
    path = Path(args.preflight_report or Path(args.output_dir) / "preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return encoded["train"], encoded["dev"]


def load_quantized_model(model_path: str, compute_dtype, args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quantization, dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()}, local_files_only=True,
        attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    base.config.use_cache = False
    adapter_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, adapter_config)
    model.print_trainable_parameters()
    return model


def make_training_args(args, compute_dtype, torch, eval_steps):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )


def run_training(args) -> None:
    import torch
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, dev_data = load_encoded_data(args, tokenizer)
    if args.preflight_only:
        return
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if native_bf16 else torch.float16
    print(f"Training context={args.max_sequence_length}; compute dtype={compute_dtype}", flush=True)
    from response_loss import ResponseOnlyTrainer
    from attention_backend import check_t4_attention, training_attention_context
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    check_t4_attention(max(len(row["input_ids"]) for row in train_data + dev_data), model_config)
    torch.cuda.empty_cache()
    model = load_quantized_model(args.model_path, compute_dtype, args)
    eval_steps = max(1, math.ceil(len(train_data) / args.gradient_accumulation_steps / args.evals_per_epoch))
    if args.save_steps:
        eval_steps = min(args.save_steps, eval_steps)
    trainer = ResponseOnlyTrainer(
        model=model,
        longest_first=args.longest_first,
        args=make_training_args(args, compute_dtype, torch, eval_steps),
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=CausalPaddingCollator(tokenizer),
        processing_class=tokenizer,
    )
    if not args.disable_early_stopping:
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))
    with training_attention_context():
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(final_path)
    print(f"Saved final LoRA adapter to {final_path}")


def main() -> None:
    args = parse_args()
    import torch

    if not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU. Enable GPU in Colab runtime settings.")
    run_training(args)


if __name__ == "__main__":
    main()
