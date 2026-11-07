from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_tokenizer(model_name: str, revision: str, local_path: str | None = None):
    source = local_path if local_path else model_name
    local_only = bool(local_path and Path(local_path).is_dir())
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if local_only else revision, use_fast=True, local_files_only=local_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str, revision: str, quantized: bool = True, training: bool = False,
               local_path: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to load the DataMate model")
    dtype = select_compute_dtype()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    ) if quantized else None
    source = local_path if local_path else model_name
    local_only = bool(local_path and Path(local_path).is_dir())
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=None if local_only else revision,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=local_only,
    )
    model.config.use_cache = not training
    model.eval()
    return model


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_text(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int = 512,
                  max_context_tokens: int = 8192) -> str:
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    device = model.get_input_embeddings().weight.device
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)
    model_context = getattr(model.config, "max_position_embeddings", 32768)
    max_context = min(model_context, max_context_tokens)
    if input_ids.shape[-1] + max_new_tokens > max_context:
        raise ValueError(f"Prompt uses {input_ids.shape[-1]} tokens; context limit is {max_context}")
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
