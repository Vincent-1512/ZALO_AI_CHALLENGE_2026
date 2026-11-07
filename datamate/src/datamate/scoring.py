from __future__ import annotations

import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


def normalize_string(value: Any) -> str:
    value = unicodedata.normalize("NFC", str(value)).casefold()
    return "".join(character for character in value if character.isalnum())


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return False


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _same_number(left, right)
    if isinstance(left, str) and isinstance(right, str):
        return normalize_string(left) == normalize_string(right)
    return left == right


def answer_matches(prediction: dict[str, Any], gold: dict[str, Any]) -> bool:
    gold_type = gold.get("type")
    if prediction.get("type") != gold_type:
        return False
    if gold_type == "unanswerable":
        return True
    expected = gold.get("answer")
    actual = prediction.get("answer")
    if gold_type in {"number", "percent"}:
        return _same_number(actual, expected)
    if gold_type == "boolean":
        return isinstance(actual, bool) and actual is expected
    if gold_type == "string":
        candidates = [expected] + (gold.get("alias") if isinstance(gold.get("alias"), list) else [])
        return any(_same_value(actual, candidate) for candidate in candidates)
    if gold_type == "date":
        return isinstance(actual, str) and actual == expected
    if gold_type == "list":
        if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
            return False
        aliases = gold.get("alias") if isinstance(gold.get("alias"), list) else []
        for index, (left, right) in enumerate(zip(actual, expected)):
            accepted = [right]
            if index < len(aliases) and isinstance(aliases[index], list):
                accepted.extend(aliases[index])
            if not any(_same_value(left, candidate) for candidate in accepted):
                return False
        return True
    if gold_type == "table":
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        if actual.get("columns") != expected.get("columns"):
            return False
        actual_rows, expected_rows = actual.get("rows"), expected.get("rows")
        if not isinstance(actual_rows, list) or not isinstance(expected_rows, list) or len(actual_rows) != len(expected_rows):
            return False
        aliases = gold.get("alias") if isinstance(gold.get("alias"), list) else []
        remaining = list(enumerate(expected_rows))
        for row in actual_rows:
            match_index = None
            for index, (gold_index, candidate) in enumerate(remaining):
                if len(row) != len(candidate) or len(row) == 0:
                    continue
                first_column = [candidate[0]]
                if gold_index < len(aliases) and isinstance(aliases[gold_index], list):
                    first_column.extend(aliases[gold_index])
                if (any(_same_value(row[0], accepted) for accepted in first_column)
                        and all(_same_value(a, b) for a, b in zip(row[1:], candidate[1:]))):
                    match_index = index
                    break
            if match_index is None:
                return False
            remaining.pop(match_index)
        return not remaining
    return actual == expected


def score_records(predictions: list[dict[str, Any]], gold_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute DataMate's +1 / -1 / 0 answer utility and diagnostics."""
    predicted_by_id = {row["id"]: row for row in predictions if isinstance(row.get("id"), str)}
    if len(predicted_by_id) != len([row for row in predictions if isinstance(row.get("id"), str)]):
        raise ValueError("Duplicate prediction IDs")
    utility, correct, wrong, abstain, missing, invalid = 0, 0, 0, 0, 0, 0
    for gold in gold_records:
        prediction = predicted_by_id.get(gold["id"])
        if prediction is None:
            utility += 0
            missing += 1
        elif prediction.get("type") in {None, "abstain"}:
            abstain += 1
        elif answer_matches(prediction, gold["gold"] if "gold" in gold else gold):
            utility += 1
            correct += 1
        else:
            utility -= 1
            wrong += 1
        if prediction is not None and not isinstance(prediction.get("type"), str):
            invalid += 1
    count = len(gold_records)
    return {
        "samples": count,
        "utility_sum": utility,
        "utility_mean": utility / count if count else 0.0,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "missing": missing,
        "invalid": invalid,
        "coverage": (correct + wrong) / count if count else 0.0,
        "exact_accuracy": correct / count if count else 0.0,
    }
