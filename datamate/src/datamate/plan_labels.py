from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from .io import load_business_package, read_jsonl, write_jsonl
from .planning import plan_and_execute
from .scoring import answer_matches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if not max_samples or max_samples >= len(rows):
        return rows
    random.Random(seed).shuffle(rows)
    return rows[:max_samples]


def build_verified_plans(data_root: Path, processed_dir: Path, model: Any, tokenizer: Any,
                         model_name: str, revision: str, max_samples: int = 0,
                         seed: int = 2026, max_new_tokens: int = 384,
                         max_context_tokens: int = 8192) -> dict[str, Any]:
    """Generate plans for train only; persist only execution-verified records for SFT."""
    train_dir = data_root / "train"
    source_rows = read_jsonl(processed_dir / "train_questions.jsonl")
    source_questions = train_dir / "questions.jsonl"
    source_answers = train_dir / "answers.jsonl"
    rows = _selection(source_rows, max_samples, seed)
    train_archive = data_root.parent / "archives" / "train.zip"
    if train_archive.is_file():
        raw_data_signature = sha256_file(train_archive)
    else:
        raw_files = sorted(({"path": path.relative_to(train_dir).as_posix(), "bytes": path.stat().st_size,
                             "mtime_ns": path.stat().st_mtime_ns}
                            for path in (train_dir / "data").rglob("*") if path.is_file()),
                           key=lambda item: item["path"])
        raw_data_signature = hashlib.sha256(json.dumps(raw_files, sort_keys=True).encode()).hexdigest()
    source_dir = Path(__file__).resolve().parent
    pipeline_hash = hashlib.sha256("".join(
        sha256_file(source_dir / filename) for filename in
        ("io.py", "query_plan.py", "executor.py", "planning.py", "scoring.py")
    ).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps({
        "model": model_name,
        "revision": revision,
        "seed": seed,
        "limit": max_samples,
        "questions": sha256_file(source_questions),
        "answers": sha256_file(source_answers),
        "raw_data": raw_data_signature,
        "pipeline": pipeline_hash,
        "selected_ids": [row["id"] for row in rows],
    }, sort_keys=True).encode()).hexdigest()
    progress_path = processed_dir / f"plan_progress_{max_samples or 'all'}_{signature[:10]}.jsonl"
    metadata_path = progress_path.with_suffix(".meta.json")
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError(f"Progress metadata changed; choose a new output directory: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps({"signature": signature, "model": model_name,
                                             "revision": revision, "seed": seed,
                                             "selected": len(rows)}, indent=2) + "\n", encoding="utf-8")
    previous = read_jsonl(progress_path) if progress_path.exists() else []
    completed = {record["id"] for record in previous}
    package_cache = {}
    package_errors = {}
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    new_records = 0
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, row in enumerate(rows, start=1):
            if row["id"] in completed:
                continue
            business_id = row["business_id"]
            result: dict[str, Any] = {"id": row["id"], "business_id": business_id, "status": "invalid"}
            raw = ""
            try:
                if business_id in package_errors:
                    raise RuntimeError(package_errors[business_id])
                if business_id not in package_cache:
                    print(f"loading business={business_id}; parse/cache its tables", flush=True)
                    try:
                        package_cache[business_id] = load_business_package(train_dir, business_id)
                    except Exception as error:
                        package_errors[business_id] = f"{type(error).__name__}: {error}"
                        raise
                package = package_cache[business_id]
                plan, prediction, raw, attempts = plan_and_execute(
                    model, tokenizer, row, package, max_new_tokens, max_context_tokens,
                    repair_attempts=1,
                )
                expected = row["gold"]
                result.update({"status": "verified" if answer_matches(prediction, expected) else "mismatch",
                               "plan": plan, "prediction": prediction,
                               "gold": expected, "question": row["question"],
                               "as_of_date": row["as_of_date"], "attempts": attempts})
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["attempts"] = getattr(error, "attempts", 1)
                raw = getattr(error, "raw_response", raw)
                if raw:
                    result["raw_response"] = raw[:4000]
            progress.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            completed.add(row["id"])
            new_records += 1
            if new_records % 10 == 0:
                elapsed = time.monotonic() - started
                rate = new_records / max(elapsed, 1e-9)
                remaining = max(0, len(rows) - len(completed))
                eta_minutes = remaining / max(rate, 1e-9) / 60
                print(f"processed={len(completed)}/{len(rows)} elapsed={elapsed:.0f}s rate={rate:.3f}/s eta≈{eta_minutes:.1f}min")
    all_records = read_jsonl(progress_path)
    verified = [record for record in all_records if record.get("status") == "verified"]
    write_jsonl(processed_dir / "train_verified_plans.jsonl", verified)
    counts = {status: sum(record.get("status") == status for record in all_records)
              for status in ("verified", "mismatch", "invalid")}
    report = {"pipeline_version": 2, "signature": signature,
              "processed": len(all_records), "selected": len(rows),
              "counts": counts, "verified_rate": counts["verified"] / max(len(all_records), 1),
              "progress_file": str(progress_path),
              "verified_file": str(processed_dir / "train_verified_plans.jsonl")}
    report_path = processed_dir / "plan_generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
