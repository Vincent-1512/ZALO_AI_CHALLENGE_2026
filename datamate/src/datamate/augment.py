from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl


SAFE_REWRITES = (
    (re.compile(r"\bbao nhiu\b", re.IGNORECASE), "bao nhiêu"),
    (re.compile(r"\bbao nhiêu\b", re.IGNORECASE), "là bao nhiêu"),
    (re.compile(r"\btổng giá trị\b", re.IGNORECASE), "tổng cộng giá trị"),
    (re.compile(r"\btháng này\b", re.IGNORECASE), "trong tháng này"),
)


def make_paraphrase(question: str) -> str | None:
    for pattern, replacement in SAFE_REWRITES:
        if pattern.search(question):
            variant = pattern.sub(replacement, question, count=1)
            if variant.strip() != question.strip():
                return variant
    return None


def augment_verified_plans(input_path: Path, output_path: Path) -> dict[str, Any]:
    originals = read_jsonl(input_path)
    augmented = []
    seen = {record["question"].strip().casefold() for record in originals}
    for record in originals:
        variant = make_paraphrase(record["question"])
        if not variant or variant.strip().casefold() in seen:
            continue
        seen.add(variant.strip().casefold())
        augmented.append({**record, "id": f"aug_{record['id']}", "parent_id": record["id"],
                          "question": variant, "augmentation": "deterministic_verified_paraphrase"})
    combined = originals + augmented
    write_jsonl(output_path, combined)
    report = {"originals": len(originals), "augmented": len(augmented),
              "total": len(combined), "augmentation_ratio": len(augmented) / max(1, len(originals)),
              "output": str(output_path)}
    output_path.with_suffix(".report.json").write_text(
        __import__("json").dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
