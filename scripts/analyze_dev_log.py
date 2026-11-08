#!/usr/bin/env python3
"""Analyze evaluate_dev.py per-sample JSONL logs without model dependencies.

Normalization for WER/CER applies Unicode NFC, lowercase, and removes punctuation
while preserving Vietnamese accents; WER tokenizes on whitespace and CER measures
the resulting string including its normalized spaces. A sample may have multiple
error labels: ASR_ERROR is independent of downstream call errors. ASR_ERROR uses
WER >= --asr-error-wer. Parse/schema/max-step labels come from end_reason. Call
labels compare the ordered gold/predicted call lists: a same-multiset permutation
is WRONG_ORDER; otherwise differing tool names/arguments are matched by position,
and list-length differences yield MISSING_CALL/EXTRA_CALL. Any unmatched residual
call is WRONG_ARGS. OK means no other label applies.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def error_rates(asr_text: str, gold_text: str) -> tuple[float, float]:
    asr, gold = normalize(asr_text), normalize(gold_text)
    asr_words, gold_words = asr.split(), gold.split()
    wer = edit_distance(asr_words, gold_words) / max(1, len(gold_words))
    cer = edit_distance(asr, gold) / max(1, len(gold))
    return wer, cer


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def classify(record: dict, asr_error_wer: float) -> list[str]:
    tags = []
    asr_text = record.get("asr_text") or ""
    gold_text = record.get("gold_text") or ""
    wer, _ = error_rates(asr_text, gold_text)
    if gold_text and wer >= asr_error_wer:
        tags.append("ASR_ERROR")

    end_reason = record.get("end_reason", "unknown")
    if end_reason in {"parse_failure", "retry_not_a_call"} or end_reason.startswith("exception:"):
        tags.append("PARSE_FAIL")
    if end_reason == "schema_rejected_after_retry" or record.get("schema_retries", 0) > 0:
        tags.append("SCHEMA_REJECTED")
    if end_reason == "max_steps":
        tags.append("MAX_STEPS")

    gold = record.get("gold_calls", []) or []
    predicted = record.get("predicted_calls", []) or []
    gold_keys = [canonical(x) for x in gold]
    predicted_keys = [canonical(x) for x in predicted]
    if predicted_keys != gold_keys:
        if Counter(predicted_keys) == Counter(gold_keys):
            tags.append("WRONG_ORDER")
        else:
            remaining_gold = Counter(gold_keys)
            remaining_predicted = Counter(predicted_keys)
            for key in gold_keys:
                if remaining_gold[key] and remaining_predicted[key]:
                    remaining_gold[key] -= 1
                    remaining_predicted[key] -= 1
            unmatched_gold = [json.loads(key) for key, count in remaining_gold.items() for _ in range(count)]
            unmatched_predicted = [json.loads(key) for key, count in remaining_predicted.items() for _ in range(count)]
            paired = min(len(unmatched_gold), len(unmatched_predicted))
            wrong_tool = wrong_args = False
            for gold_call, predicted_call in zip(unmatched_gold[:paired], unmatched_predicted[:paired]):
                if gold_call.get("name") == predicted_call.get("name"):
                    wrong_args = True
                elif canonical(gold_call.get("arguments", {})) == canonical(predicted_call.get("arguments", {})):
                    wrong_tool = True
                else:
                    wrong_tool = True
                    wrong_args = True
            if wrong_tool:
                tags.append("WRONG_TOOL_NAME")
            if wrong_args:
                tags.append("WRONG_ARGS")
            if len(unmatched_gold) > paired:
                tags.append("MISSING_CALL")
            if len(unmatched_predicted) > paired:
                tags.append("EXTRA_CALL")
    return tags or ["OK"]


def wer_bucket(wer: float) -> str:
    if wer <= 0.10:
        return "low (<=0.10)"
    if wer <= 0.40:
        return "medium (0.10–0.40)"
    return "high (>0.40)"


def analyze(path: Path, threshold: float) -> dict:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results, by_tag, buckets = [], defaultdict(list), defaultdict(list)
    for row in records:
        wer, cer = error_rates(row.get("asr_text") or "", row.get("gold_text") or "")
        tags = classify(row, threshold)
        result = {"id": row.get("id"), "wer": wer, "cer": cer, "state_match": row.get("state_match"),
                  "end_reason": row.get("end_reason", "unknown"), "labels": tags,
                  "asr_text": row.get("asr_text", ""), "gold_text": row.get("gold_text", ""),
                  "gold_calls": row.get("gold_calls", []), "predicted_calls": row.get("predicted_calls", [])}
        results.append(result)
        buckets[wer_bucket(wer)].append(result)
        for tag in tags:
            by_tag[tag].append(result)
    wrong = [row for row in results if row["labels"] != ["OK"]]
    asr_wrong = sum("ASR_ERROR" in row["labels"] for row in wrong)
    grouped_state = {}
    for bucket, rows in buckets.items():
        scored = [row for row in rows if isinstance(row["state_match"], bool)]
        grouped_state[bucket] = {
            "samples": len(rows), "state_match_count": sum(row["state_match"] for row in scored),
            "state_match_rate": sum(row["state_match"] for row in scored) / len(scored) if scored else None,
        }
    return {"samples": len(results),
            "mean_wer": sum(row["wer"] for row in results) / len(results) if results else 0.0,
            "mean_cer": sum(row["cer"] for row in results) / len(results) if results else 0.0,
            "wer_normalization": "NFC, lowercase, punctuation removed; whitespace collapsed; accents retained",
            "cer_unit": "Unicode code points after same normalization", "asr_error_wer_threshold": threshold,
            "error_sample_count": len(wrong), "asr_error_samples": asr_wrong,
            "asr_error_share_of_error_samples": asr_wrong / len(wrong) if wrong else 0.0,
            "label_counts": dict(Counter(tag for row in results for tag in row["labels"])),
            "state_match_by_wer": grouped_state,
            "examples_by_label": {tag: rows[:10] for tag, rows in sorted(by_tag.items()) if tag != "OK"},
            "samples_detail": results}


def markdown(report: dict) -> str:
    lines = ["# Dev log analysis", "", f"Samples: {report['samples']}",
             f"Mean WER: {report['mean_wer']:.3f}; mean CER: {report['mean_cer']:.3f}.",
             f"Normalization: {report['wer_normalization']}; CER: {report['cer_unit']}.", "",
             f"ASR_ERROR share among error samples: {report['asr_error_samples']}/{report['error_sample_count']} "
             f"({report['asr_error_share_of_error_samples']:.1%}).", "",
             "## Label counts", "", "| Label | Samples |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(report["label_counts"].items()))
    lines += ["", "## State match by WER", "", "| WER group | Samples | State match |", "|---|---:|---:|"]
    for key, value in report["state_match_by_wer"].items():
        rate = "n/a" if value["state_match_rate"] is None else f"{value['state_match_rate']:.1%}"
        lines.append(f"| {key} | {value['samples']} | {rate} ({value['state_match_count']}) |")
    for label, rows in report["examples_by_label"].items():
        lines += ["", f"## Examples: {label}", ""]
        for row in rows:
            lines.append(f"- `{row['id']}` WER={row['wer']:.3f} state_match={row['state_match']} "
                          f"end={row['end_reason']}: ASR={row['asr_text']!r}; gold={row['gold_text']!r}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="evaluate_dev.py per-sample JSONL")
    parser.add_argument("--output-json", default="reports/runs/dev_log_analysis.json")
    parser.add_argument("--output-md", default="reports/runs/dev_log_analysis.md")
    parser.add_argument("--asr-error-wer", type=float, default=0.5)
    args = parser.parse_args()
    report = analyze(Path(args.input), args.asr_error_wer)
    json_path, md_path = Path(args.output_json), Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}; ASR_ERROR share={report['asr_error_share_of_error_samples']:.1%}")


if __name__ == "__main__":
    main()
