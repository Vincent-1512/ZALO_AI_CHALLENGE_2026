#!/usr/bin/env python3
"""Dev WER of a Qwen3-ASR model on the SAME dev audio used by training/finetune_asr.py.

Run it in a separate venv that has `qwen-asr` installed (it may pin other library versions):
  python -m venv --system-site-packages /workspace/venv_qwen && /workspace/venv_qwen/bin/pip install -U qwen-asr
Compare the printed WER with the 'WER dev TRUOC' line from finetune_asr.py (PhoWhisper-large baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from finetune_asr import audio_pairs, corpus_wer  # noqa: E402
from lifemate.data import read_jsonl, resolve_split  # noqa: E402
from prepare_sft import split_rows  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="HF id or local dir")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    split = resolve_split(args.data_root, "train")
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    parts = split_rows(rows, args.seed, (0.8, 0.1, 0.1))
    train_audio = {a for a, _ in audio_pairs(parts["train"], split.split_dir)}
    dev = [d for d in audio_pairs(parts["dev"], split.split_dir) if d[0] not in train_audio]
    if args.limit:
        dev = dev[:args.limit]
    paths, refs = [a for a, _ in dev], [t for _, t in dev]
    print(f"dev_pairs={len(dev)} model={args.model}", flush=True)

    asr = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0",
                                        max_inference_batch_size=args.batch_size, max_new_tokens=128)
    hyps, t0 = [], time.time()
    for start in range(0, len(paths), args.batch_size):
        out = asr.transcribe(audio=paths[start:start + args.batch_size], language="Vietnamese")
        hyps.extend(o.text.strip() for o in out)
    seconds = time.time() - t0
    wer = corpus_wer(hyps, refs)
    print(f"WER dev Qwen3-ASR = {wer:.4f}  ({seconds / max(1, len(paths)):.2f}s/audio)", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "model": args.model, "dev_pairs": len(dev), "wer_dev": wer,
        "seconds_per_audio": seconds / max(1, len(paths)),
        "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:20]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
