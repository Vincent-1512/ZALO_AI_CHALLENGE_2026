from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .splits import prepare_group_split


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(data_root: Path, processed_dir: Path, dev_businesses: list[str] | None = None) -> dict[str, Any]:
    train_root = data_root / "train"
    manifest = prepare_group_split(data_root, processed_dir, dev_businesses)
    rows = read_jsonl(train_root / "questions.jsonl")
    inventory = {"train_questions": len(rows), "train_businesses": {}, "train_questions_sha256": _hash(train_root / "questions.jsonl"),
                 "train_answers_sha256": _hash(train_root / "answers.jsonl"), "split": manifest}
    for business_id in sorted({row["business_id"] for row in rows}):
        business_dir = train_root / "data" / business_id
        files = sorted(path for path in business_dir.iterdir() if path.is_file())
        data_files = [path for path in files if path.suffix.lower() in {".csv", ".xlsx", ".sqlite", ".db"}]
        inventory["train_businesses"][business_id] = {
            "source_file_count": len(data_files),
            "source_bytes": sum(path.stat().st_size for path in data_files),
            "extensions": {suffix: sum(path.suffix.lower() == suffix for path in data_files)
                           for suffix in sorted({path.suffix.lower() for path in data_files})},
            "files": [path.name for path in data_files],
            "readme_present": (business_dir / "README.md").is_file(),
        }
    public_root = data_root / "public_test"
    public_questions = read_jsonl(public_root / "questions.jsonl")
    inventory["public_questions"] = len(public_questions)
    inventory["public_businesses"] = len({row["business_id"] for row in public_questions})
    inventory["public_question_ids_unique"] = len({row["id"] for row in public_questions}) == len(public_questions)
    if not inventory["public_question_ids_unique"]:
        raise ValueError("Duplicate public question IDs found")
    if any("answer" in row or "gold" in row for row in public_questions):
        raise ValueError("Public test questions contain answer-like fields; stop and inspect the data split")
    output = processed_dir / "data_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"train_questions": len(rows), "train_businesses": len(inventory["train_businesses"]),
            "public_questions": len(public_questions), "dev_questions": manifest["dev_count"],
            "train_questions_out": str(processed_dir / "train_questions.jsonl"),
            "dev_questions_out": str(processed_dir / "dev_questions.jsonl"),
            "inventory_out": str(output)}
