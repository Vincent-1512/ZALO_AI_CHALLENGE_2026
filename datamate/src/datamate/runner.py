from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .io import load_business_package, read_jsonl, write_jsonl
from .planning import plan_and_execute
from .scoring import score_records


def predict_records(model: Any, tokenizer: Any, questions: list[dict[str, Any]], split_root: Path,
                    max_new_tokens: int = 512,
                    package_cache: dict[str, Any] | None = None,
                    max_context_tokens: int = 8192) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    package_cache = package_cache if package_cache is not None else {}
    predictions, diagnostics = [], []
    started = time.monotonic()
    for index, question in enumerate(questions, start=1):
        business_id = question["business_id"]
        diagnostic = {"id": question["id"], "business_id": business_id, "status": "ok"}
        sample_started = time.monotonic()
        try:
            if business_id not in package_cache:
                package_cache[business_id] = load_business_package(split_root, business_id)
            package = package_cache[business_id]
            plan, prediction, _, attempts = plan_and_execute(
                model, tokenizer, question, package,
                max_new_tokens=max_new_tokens, max_context_tokens=max_context_tokens,
            )
            prediction = {"id": question["id"], **prediction}
            diagnostic.update({"plan": plan, "attempts": attempts})
        except Exception as error:
            prediction = {"id": question["id"], "type": "abstain"}
            diagnostic.update({"status": "abstain", "error": f"{type(error).__name__}: {error}",
                               "attempts": getattr(error, "attempts", 1)})
        predictions.append(prediction)
        diagnostic["elapsed_seconds"] = round(time.monotonic() - sample_started, 4)
        diagnostics.append(diagnostic)
        if index % 25 == 0 or index == len(questions):
            print(f"predicted={index}/{len(questions)} elapsed={time.monotonic() - started:.1f}s")
    return predictions, diagnostics


def evaluate_split(model: Any, tokenizer: Any, questions_path: Path, split_root: Path,
                   output_dir: Path, max_samples: int = 0, max_new_tokens: int = 512,
                   max_context_tokens: int = 8192) -> dict[str, Any]:
    rows = read_jsonl(questions_path)
    if max_samples:
        rows = rows[:max_samples]
    predictions, diagnostics = predict_records(model, tokenizer, rows, split_root, max_new_tokens,
                                               max_context_tokens=max_context_tokens)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
    report = score_records(predictions, rows)
    report["seconds"] = sum(float(item.get("elapsed_seconds", 0)) for item in diagnostics)
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_public(model: Any, tokenizer: Any, data_root: Path, output_path: Path,
               max_samples: int = 0, max_new_tokens: int = 512,
               max_context_tokens: int = 8192) -> dict[str, Any]:
    split_root = data_root / "public_test" if (data_root / "public_test/questions.jsonl").is_file() else data_root
    questions = read_jsonl(split_root / "questions.jsonl")
    if max_samples:
        questions = questions[:max_samples]
    predictions, diagnostics = predict_records(model, tokenizer, questions, split_root, max_new_tokens,
                                               max_context_tokens=max_context_tokens)
    write_jsonl(output_path, predictions)
    diagnostic_path = output_path.with_suffix(".diagnostics.jsonl")
    write_jsonl(diagnostic_path, diagnostics)
    return {"samples": len(predictions), "output": str(output_path), "diagnostics": str(diagnostic_path)}
