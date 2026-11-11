#!/usr/bin/env python3
"""Diagnostic: zero-shot WER of hynt/Zipformer-30M-RNNT-6000h on the 154 train-dev samples,
compared with the PhoWhisper-large-ft transcripts already stored in
artifacts/merge_reports/reports/dev_merged_samples.jsonl. CPU is enough (30M params).

NOTE: the model card says cc-by-nc-nd-4.0 (non-commercial, no derivatives). This script only
measures; do not ship or fine-tune the model before checking the licence against the contest rules.
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

REPO = "hynt/Zipformer-30M-RNNT-6000h"
FILES = ["bpe.model", "encoder-epoch-20-avg-10.int8.onnx",
         "decoder-epoch-20-avg-10.int8.onnx", "joiner-epoch-20-avg-10.int8.onnx"]


def words(text):
    text = unicodedata.normalize("NFC", (text or "").lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, row[0] = row[:], i
        for j in range(1, len(hyp) + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
    return row[-1]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--cache", default="/tmp/zipformer_hynt")
    ap.add_argument("--out", default="reports/zipformer_dev")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)

    import numpy as np
    import sentencepiece as spm
    import sherpa_onnx
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    local = {name: hf_hub_download(REPO, name, local_dir=str(cache)) for name in FILES}
    sp = spm.SentencePieceProcessor(model_file=local["bpe.model"])
    tokens = cache / "tokens.txt"
    tokens.write_text("".join(f"{sp.id_to_piece(i)} {i}\n" for i in range(sp.get_piece_size())), encoding="utf-8")
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=local["encoder-epoch-20-avg-10.int8.onnx"], decoder=local["decoder-epoch-20-avg-10.int8.onnx"],
        joiner=local["joiner-epoch-20-avg-10.int8.onnx"], tokens=str(tokens),
        num_threads=args.threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")

    split_dir = root / "data/raw/train+public_test/train"
    train = {row["id"]: row for row in read_jsonl(split_dir / "train.jsonl")}
    dev_ids = [row["sample_id"] for row in read_jsonl(root / "data/processed/sft/dev.jsonl")]
    stored = {row["id"]: row for row in read_jsonl(root / "artifacts/merge_reports/reports/dev_merged_samples.jsonl")}
    dev_ids = [i for i in dict.fromkeys(dev_ids) if i in train and i in stored]
    print(f"{len(dev_ids)} dev samples", flush=True)

    results, started = [], time.perf_counter()
    for n, sample_id in enumerate(dev_ids, 1):
        wav, rate = sf.read(split_dir / train[sample_id]["audio"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        stream = recognizer.create_stream()
        stream.accept_waveform(rate, np.ascontiguousarray(wav))
        recognizer.decode_stream(stream)
        zip_text = stream.result.text
        gold = train[sample_id]["audio_text"]
        ref, z, f = words(gold), words(zip_text), words(stored[sample_id]["asr_text"])
        results.append({"id": sample_id, "gold": gold, "zipformer": zip_text, "large_ft": stored[sample_id]["asr_text"],
                        "n_ref": len(ref), "e_zip": edits(ref, z), "e_ft": edits(ref, f)})
        if n % 25 == 0:
            print(f"{n}/{len(dev_ids)} {time.perf_counter() - started:.0f}s", flush=True)

    total = sum(r["n_ref"] for r in results)
    summary = {
        "samples": len(results),
        "corpus_wer_zipformer": sum(r["e_zip"] for r in results) / total,
        "corpus_wer_large_ft": sum(r["e_ft"] for r in results) / total,
        "corpus_wer_best_of_two_per_sample": sum(min(r["e_zip"], r["e_ft"]) for r in results) / total,
        "zipformer_better": sum(r["e_zip"] < r["e_ft"] for r in results),
        "large_ft_better": sum(r["e_ft"] < r["e_zip"] for r in results),
        "tie": sum(r["e_ft"] == r["e_zip"] for r in results),
        "note": "WER uses lowercase + punctuation stripped; digits vs spelled-out numbers count as errors for both models.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(str(out) + "_samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n--- 6 mẫu Zipformer đúng hơn large_ft nhiều nhất ---")
    for r in sorted(results, key=lambda r: r["e_zip"] - r["e_ft"])[:6]:
        print(f"GOLD: {r['gold']}\n ZIP : {r['zipformer']}\n FT  : {r['large_ft']}\n")


if __name__ == "__main__":
    main()
