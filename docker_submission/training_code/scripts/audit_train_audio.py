#!/usr/bin/env python3
"""Transcribe every train audio with an ASR model and flag suspicious samples. Never deletes data.

For each sample (current audio + user-history audio) writes: gold text, ASR text, WER, and flags
  - high_wer      : ASR disagrees strongly with the gold transcript (bad audio or wrong label)
  - too_short     : gold transcript has < 2 words
  - noop_request  : the sample's gold label has NO tool calls (out-of-scope / nothing to do).
                    These are KEPT for policy training (empty gold scores only if prediction is empty).
Outputs <out>.jsonl (per audio) and <out>.summary.json (counts, WER percentiles).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from lifemate.data import read_jsonl, resolve_audio_path, resolve_split  # noqa: E402
from lifemate.models import SpeechRecognizer  # noqa: E402
from finetune_asr import edit_distance, normalize  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="path prefix, e.g. runs/audit_train")
    p.add_argument("--high-wer", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    split = resolve_split(args.data_root, "train")
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    items, seen = [], set()
    for row in rows:
        noop = not row.get("label", {}).get("tool_calls")
        pairs = [(row.get("audio"), row.get("audio_text"), "current")]
        pairs += [(t.get("audio"), t.get("audio_text"), "history") for t in row.get("history", [])
                  if t.get("role") == "user"]
        for ref, text, kind in pairs:
            if not ref or not text or ref in seen:
                continue
            seen.add(ref)
            items.append({"id": row["id"], "audio": ref, "kind": kind, "gold": text.strip(),
                          "noop_request": bool(noop and kind == "current")})
    print(f"audio_files={len(items)}", flush=True)

    asr = SpeechRecognizer(args.asr_model, batch_size=8)
    paths = [resolve_audio_path(i["audio"], split.split_dir) for i in items]
    hyps: list[str] = []
    for start in range(0, len(paths), 64):
        hyps.extend(asr.transcribe_many(paths[start:start + 64]))
        print(f"transcribed={len(hyps)}/{len(paths)}", flush=True)

    counts = {"high_wer": 0, "too_short": 0, "noop_request": 0}
    wers = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for item, hyp in zip(items, hyps):
            ref_w, hyp_w = normalize(item["gold"]), normalize(hyp)
            wer = edit_distance(hyp_w, ref_w) / max(1, len(ref_w))
            flags = []
            if wer >= args.high_wer:
                flags.append("high_wer")
            if len(ref_w) < 2:
                flags.append("too_short")
            if item["noop_request"]:
                flags.append("noop_request")
            for f in flags:
                counts[f] += 1
            wers.append(wer)
            fh.write(json.dumps({**item, "asr": hyp, "wer": round(wer, 4), "flags": flags},
                                ensure_ascii=False) + "\n")
    wers.sort()
    pct = lambda q: wers[min(len(wers) - 1, int(q * len(wers)))] if wers else None  # noqa: E731
    summary = {"audio_files": len(items), "flag_counts": counts,
               "wer_mean": sum(wers) / max(1, len(wers)),
               "wer_p50": pct(0.5), "wer_p90": pct(0.9), "wer_p99": pct(0.99),
               "note": "No data was removed. noop_request samples must stay in policy training."}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
