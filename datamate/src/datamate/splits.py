from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


DEFAULT_DEV_BUSINESSES = ["biz_101", "biz_103", "biz_111", "biz_115", "biz_123", "biz_126"]


def load_labeled_train(split_root: Path) -> list[dict[str, Any]]:
    questions = read_jsonl(split_root / "questions.jsonl")
    answers = read_jsonl(split_root / "answers.jsonl")
    question_by_id = {row["id"]: row for row in questions}
    answer_by_id = {row["id"]: row for row in answers}
    if len(question_by_id) != len(questions) or len(answer_by_id) != len(answers):
        raise ValueError("Duplicate question/answer IDs found")
    missing = question_by_id.keys() - answer_by_id.keys()
    extra = answer_by_id.keys() - question_by_id.keys()
    if missing or extra:
        raise ValueError(f"Question/answer ID mismatch: missing={len(missing)}, extra={len(extra)}")
    return [{**question_by_id[key], "gold": answer_by_id[key]} for key in question_by_id]


def choose_dev_businesses(records: list[dict[str, Any]], fraction: float = 0.2, seed: int = 2026,
                          fixed_ids: list[str] | None = None) -> list[str]:
    available = sorted({row["business_id"] for row in records})
    if fixed_ids is not None:
        missing = sorted(set(fixed_ids) - set(available))
        if missing:
            raise ValueError(f"Configured dev business IDs not found: {missing}")
        return sorted(set(fixed_ids))
    count = max(1, round(len(available) * fraction))
    return sorted(random.Random(seed).sample(available, count))


def group_split(records: list[dict[str, Any]], dev_businesses: list[str]) -> tuple[list, list]:
    dev_set = set(dev_businesses)
    train_rows = [row for row in records if row["business_id"] not in dev_set]
    dev_rows = [row for row in records if row["business_id"] in dev_set]
    if not train_rows or not dev_rows:
        raise ValueError("Business-group split produced an empty train or dev set")
    overlap = {row["business_id"] for row in train_rows} & {row["business_id"] for row in dev_rows}
    if overlap:
        raise AssertionError(f"Business leakage in split: {sorted(overlap)}")
    return train_rows, dev_rows


def write_split_manifest(path: Path, train_rows: list, dev_rows: list, dev_businesses: list[str]) -> dict:
    manifest = {
        "seed": 2026,
        "dev_businesses": sorted(dev_businesses),
        "train_businesses": sorted({row["business_id"] for row in train_rows}),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "train_ids": [row["id"] for row in train_rows],
        "dev_ids": [row["id"] for row in dev_rows],
        "train_type_counts": dict(Counter(row["gold"]["type"] for row in train_rows)),
        "dev_type_counts": dict(Counter(row["gold"]["type"] for row in dev_rows)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_group_split(data_root: Path, output_dir: Path,
                        dev_businesses: list[str] | None = None) -> dict:
    train_root = data_root / "train"
    records = load_labeled_train(train_root)
    chosen = choose_dev_businesses(records, fixed_ids=dev_businesses or DEFAULT_DEV_BUSINESSES)
    train_rows, dev_rows = group_split(records, chosen)
    manifest = write_split_manifest(output_dir / "split_manifest.json", train_rows, dev_rows, chosen)
    write_jsonl(output_dir / "train_questions.jsonl", train_rows)
    write_jsonl(output_dir / "dev_questions.jsonl", dev_rows)
    return manifest
