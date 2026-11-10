#!/usr/bin/env python3
"""LoRA fine-tune a Whisper-style ASR (PhoWhisper) on train audio -> `audio_text`.

Uses the same train/dev/holdout partition as the policy data (prepare_sft.split_rows, seed 2026),
so the ASR never sees dev audio unless --include-dev-holdout is given (final fit).
Reports WER before and after on the dev audio, then writes a merged model directory that
lifemate.models.SpeechRecognizer can load unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import unicodedata
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate.data import read_jsonl, resolve_audio_path, resolve_split  # noqa: E402
from prepare_sft import split_rows  # noqa: E402


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def corpus_wer(hyps: list[str], refs: list[str]) -> float:
    errors = sum(edit_distance(normalize(h), normalize(r)) for h, r in zip(hyps, refs))
    words = sum(len(normalize(r)) for r in refs)
    return errors / max(1, words)


def audio_pairs(rows: list[dict], split_dir: Path) -> list[tuple[str, str]]:
    """(audio path, transcript) for current audio and user-history audio; de-duplicated."""
    seen, pairs = set(), []
    for row in rows:
        items = [(row.get("audio"), row.get("audio_text"))]
        items += [(t.get("audio"), t.get("audio_text")) for t in row.get("history", [])
                  if t.get("role") == "user"]
        for ref, text in items:
            if not ref or not text or ref in seen:
                continue
            seen.add(ref)
            pairs.append((str(resolve_audio_path(ref, split_dir)), text.strip()))
    return pairs


def read_audio(path: str, rate: int = 16000):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != rate:
        ratio = Fraction(rate, sr)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
    return audio


def transcribe(model, processor, paths: list[str], device: str, batch_size: int = 8) -> list[str]:
    import torch

    model.eval()
    out: list[str] = []
    for start in range(0, len(paths), batch_size):
        audios = [read_audio(p) for p in paths[start:start + batch_size]]
        inputs = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
        feats = inputs.input_features.to(device, dtype=model.dtype)
        with torch.inference_mode():
            gen = model.generate(feats, language="vi", task="transcribe", do_sample=False,
                                 num_beams=1, max_new_tokens=128)
        out.extend(t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/raw/train+public_test"))
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="merged model is written here")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026, help="partition seed (must match policy data)")
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=0, help="debug: cap train pairs")
    p.add_argument("--include-dev-holdout", action="store_true",
                   help="final fit: also train on dev+holdout audio (dev WER becomes optimistic)")
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = resolve_split(args.data_root, "train")
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    parts = split_rows(rows, args.seed, (0.8, 0.1, 0.1))
    fit_rows = list(parts["train"])
    if args.include_dev_holdout:
        fit_rows += list(parts["dev"]) + list(parts["holdout"])
    train_pairs = audio_pairs(fit_rows, split.split_dir)
    dev_pairs = audio_pairs(parts["dev"], split.split_dir)
    if not args.include_dev_holdout:
        train_audio = {a for a, _ in train_pairs}
        dev_pairs = [d for d in dev_pairs if d[0] not in train_audio]
    if args.limit:
        train_pairs = train_pairs[:args.limit]
        dev_pairs = dev_pairs[:max(8, args.limit // 4)]
    print(f"train_pairs={len(train_pairs)} dev_pairs={len(dev_pairs)}", flush=True)

    processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True)
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="vi", task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.base_model), dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    model.config.use_cache = False

    dev_paths = [a for a, _ in dev_pairs]
    dev_refs = [t for _, t in dev_pairs]
    model.config.use_cache = True
    t0 = time.time()
    wer_before = corpus_wer(transcribe(model, processor, dev_paths, device), dev_refs)
    print(f"WER dev TRƯỚC fine-tune = {wer_before:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [q for q in model.parameters() if q.requires_grad]
    for q in params:
        q.data = q.data.float()  # fp32 adapter weights, bf16 base
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_pairs) / (args.batch_size * args.grad_accum))
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    start_id = model.config.decoder_start_token_id

    def collate(batch):
        feats = processor.feature_extractor([read_audio(a) for a, _ in batch], sampling_rate=16000,
                                            return_tensors="pt").input_features
        ids = [tok(t).input_ids for _, t in batch]
        ids = [x[1:] if x and x[0] == start_id else x for x in ids]
        width = max(len(x) for x in ids)
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        for i, x in enumerate(ids):
            labels[i, :len(x)] = torch.tensor(x)
        return feats, labels

    step = 0
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(train_pairs)))
        random.shuffle(order)
        micro, running = 0, 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [train_pairs[i] for i in order[start:start + args.batch_size]]
            feats, labels = collate(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_features=feats.to(device, dtype=torch.bfloat16),
                             labels=labels.to(device)).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total:
                    print(f"epoch={epoch + 1}/{args.epochs} step={step}/{total} "
                          f"loss={running / (10 * args.grad_accum):.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                    running = 0.0
        print(f"hết epoch {epoch + 1}", flush=True)

    model = model.merge_and_unload()
    model.config.use_cache = True
    for q in model.parameters():
        q.data = q.data.to(torch.bfloat16)
    hyps = transcribe(model, processor, dev_paths, device)
    wer_after = corpus_wer(hyps, dev_refs)
    print(f"WER dev SAU fine-tune = {wer_after:.4f} (trước {wer_before:.4f})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    processor.save_pretrained(str(args.output_dir))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "base_model": str(args.base_model), "output_dir": str(args.output_dir),
        "train_pairs": len(train_pairs), "dev_pairs": len(dev_pairs),
        "wer_dev_before": wer_before, "wer_dev_after": wer_after,
        "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
        "include_dev_holdout": args.include_dev_holdout,
        "examples": [{"ref": r, "hyp": h} for r, h in list(zip(dev_refs, hyps))[:20]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"đã lưu model gộp tại {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
