from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def trim_history(messages: list[dict], tokenizer, max_length: int):
    payload = json.loads(messages[1]["content"])
    while True:
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) <= max_length:
            return prompt_ids
        if not payload["history"]:
            raise ValueError(f"Prompt alone exceeds max sequence length ({len(prompt_ids)} > {max_length})")
        payload["history"].pop(0)


def encode_record(record: dict, tokenizer, max_length: int) -> dict:
    messages = [dict(message) for message in record["messages"]]
    prompt_ids = trim_history(messages, tokenizer, max_length)
    response = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": record["target"]}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if len(response) > max_length:
        raise ValueError(f"Target exceeds max sequence length for {record.get('sample_id')}")
    if response[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template changed between prompt and completed message")
    return {
        "input_ids": response,
        "attention_mask": [1] * len(response),
        "labels": [-100] * len(prompt_ids) + response[len(prompt_ids) :],
    }


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
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-dev-samples", type=int, default=0)
    return parser.parse_args()


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
    train_rows = limit_sample_groups(load_records(Path(args.train_jsonl)), args.limit_train_samples)
    dev_rows = limit_sample_groups(load_records(Path(args.dev_jsonl)), args.limit_dev_samples)
    train_data = [encode_record(row, tokenizer, args.max_sequence_length) for row in train_rows]
    dev_data = [encode_record(row, tokenizer, args.max_sequence_length) for row in dev_rows]
    if not train_data or not dev_data:
        raise ValueError("Both train and dev records must be non-empty")
    return train_data, dev_data


def load_quantized_model(model_path: str, compute_dtype):
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
        model_path, quantization_config=quantization, torch_dtype=compute_dtype,
        device_map="auto", local_files_only=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.config.use_cache = False
    adapter_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, adapter_config)
    model.print_trainable_parameters()
    return model


def make_training_args(args, compute_dtype, torch):
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
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
    from transformers import AutoTokenizer, Trainer

    seed_everything(args.seed)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, dev_data = load_encoded_data(args, tokenizer)
    model = load_quantized_model(args.model_path, compute_dtype)
    trainer = Trainer(
        model=model,
        args=make_training_args(args, compute_dtype, torch),
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=CausalPaddingCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(final_path)
    print(f"Saved final LoRA adapter to {final_path}")


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; upload this workspace and train data to Vax.ai.")
    run_training(args)


if __name__ == "__main__":
    main()
