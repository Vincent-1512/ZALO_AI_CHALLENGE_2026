from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd

from .query_plan import validate_plan


def _clean_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value) if value.is_integer() else value
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", "")
    if not text:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"(?i)(vnd|đ|dong|₫|%|\s)", "", text)
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if "," in text and "." in text:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    elif "." in text and len(text.rsplit(".", 1)[-1]) == 3:
        text = text.replace(".", "")
    elif text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + ("." + parts[-1] if len(parts[-1]) <= 2 else parts[-1])
    number = float(text)
    return -number if negative else number


def _normalized_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).casefold().strip()


def _filter(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    column, op, value = condition["column"], condition["op"], condition.get("value")
    series = frame[column]
    if op == "is_null":
        return frame[series.isna()]
    if op == "not_null":
        return frame[series.notna()]
    if op in {"contains", "not_contains"}:
        needle = _normalized_text(value)
        mask = series.map(lambda item: needle in _normalized_text(item) if pd.notna(item) else False)
        return frame[~mask if op == "not_contains" else mask]
    date_values = value if isinstance(value, list) else [value]
    if op in {"eq", "ne", "in", "not_in"} and date_values and all(
        isinstance(item, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", item) for item in date_values
    ):
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        targets = {pd.Timestamp(item) for item in date_values}
        mask = parsed.isin(targets)
        return frame[~mask if op in {"ne", "not_in"} else mask]
    if op in {"eq", "ne", "in", "not_in"}:
        values = value if isinstance(value, list) else [value]
        target = {_normalized_text(item) for item in values}
        mask = series.map(lambda item: _normalized_text(item) in target if pd.notna(item) else False)
        return frame[~mask if op in {"ne", "not_in"} else mask]
    if op == "between":
        low, high = value
        if isinstance(low, str) and isinstance(high, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", low) and re.match(r"^\d{4}-\d{2}-\d{2}$", high):
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
            return frame[(parsed >= pd.Timestamp(low)) & (parsed <= pd.Timestamp(high))]
        numeric = series.map(_number)
        return frame[(numeric >= _number(low)) & (numeric <= _number(high))]
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        target: Any = pd.Timestamp(value)
        compare = lambda left, right: {
            "gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right
        }[op]
        return frame[parsed.map(lambda item: compare(item, target) if pd.notna(item) else False)]
    numeric = series.map(_number)
    threshold = _number(value)
    mask = {"gt": numeric > threshold, "gte": numeric >= threshold,
            "lt": numeric < threshold, "lte": numeric <= threshold}[op]
    return frame[mask]


def _aggregate(frame: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    group_by = plan.get("group_by", [])
    metrics = plan.get("metrics", [])
    if not metrics:
        return frame
    if not group_by:
        result: dict[str, list[Any]] = {}
        for metric in metrics:
            column, op, alias = metric.get("column"), metric["op"], metric["as"]
            series = frame[column] if column else pd.Series([1] * len(frame), index=frame.index)
            numeric = series.map(_number) if op in {"sum", "mean"} else None
            if op == "count_rows":
                value = len(frame)
            elif op == "count":
                value = int(series.notna().sum())
            elif op == "nunique":
                value = int(series.dropna().nunique())
            elif op == "sum":
                value = float(numeric.sum()) if numeric.notna().any() else 0
            elif op == "mean":
                value = float(numeric.mean()) if numeric.notna().any() else None
            elif op == "min":
                value = series.min() if not series.dropna().empty else None
            else:
                value = series.max() if not series.dropna().empty else None
            result[alias] = [value]
        return pd.DataFrame(result)
    grouped = frame.groupby(group_by, dropna=False, sort=False)
    output = grouped.size().rename("__rows").reset_index()
    for metric in metrics:
        column, op, alias = metric.get("column"), metric["op"], metric["as"]
        if op == "count_rows":
            values = grouped.size()
        elif op == "count":
            values = grouped[column].count()
        elif op == "nunique":
            values = grouped[column].nunique(dropna=True)
        elif op in {"sum", "mean"}:
            numeric_frame = frame.assign(**{column: frame[column].map(_number)})
            numeric_group = numeric_frame.groupby(group_by, dropna=False, sort=False)[column]
            values = numeric_group.sum(min_count=1) if op == "sum" else numeric_group.mean()
        elif op == "min":
            values = grouped[column].min()
        else:
            values = grouped[column].max()
        values = values.rename(alias).reset_index()
        output = output.drop(columns=[alias], errors="ignore").merge(values, on=group_by, how="left", sort=False)
    return output.drop(columns=["__rows"], errors="ignore")


def _derive(frame: pd.DataFrame, operations: list[dict[str, Any]]) -> pd.DataFrame:
    for operation in operations:
        left = frame[operation["left"]].map(_number)
        right = frame[operation["right"]].map(_number)
        op = operation["op"]
        if op == "add":
            result = left + right
        elif op == "subtract":
            result = left - right
        elif op == "multiply":
            result = left * right
        elif op == "divide":
            result = left.div(right.where(right != 0))
        else:
            result = (left - right).div(right.where(right != 0)) * 100
        frame[operation["as"]] = result
    return frame


def _compare(left: Any, op: str, right: Any) -> bool:
    if op in {"gt", "gte", "lt", "lte"}:
        left, right = _number(left), _number(right)
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[op]
    equal = _normalized_text(left) == _normalized_text(right)
    return equal if op == "eq" else not equal


def execute_plan(plan: dict[str, Any], package: Any) -> dict[str, Any]:
    validate_plan(plan, package.catalog)
    table_names = plan["tables"]
    frame = package.tables[table_names[0]].copy()
    joins = {frozenset((join["left"].split(".", 1)[0], join["right"].split(".", 1)[0])): join
             for join in plan.get("joins", [])}
    included = {table_names[0]}
    for table_id in table_names[1:]:
        if table_id not in included:
            join = joins.get(frozenset((table_names[0], table_id)))
            if join is None:
                join = next((item for pair, item in joins.items() if table_id in pair and pair & included), None)
            if join is None:
                raise ValueError(f"No declared join path to {table_id}")
            left, right = join["left"], join["right"]
            if left not in frame.columns:
                left, right = right, left
            frame = frame.merge(package.tables[table_id], left_on=left, right_on=right, how="inner", sort=False)
            included.add(table_id)
    for condition in plan.get("filters", []):
        frame = _filter(frame, condition)
    frame = _aggregate(frame, plan)
    frame = _derive(frame, plan.get("derive", []))
    for column in plan.get("select", []):
        if column not in frame.columns:
            raise ValueError(f"Selected column is not available after aggregation: {column}")
    available = set(frame.columns)
    for order in plan.get("sort", []):
        if order["column"] not in available:
            raise ValueError(f"Sort column is not available after aggregation: {order['column']}")
    if plan.get("sort"):
        frame = frame.sort_values(
            [item["column"] for item in plan["sort"]],
            ascending=[item["direction"] == "asc" for item in plan["sort"]],
            kind="mergesort", na_position="last",
        )
    if plan.get("limit") is not None:
        frame = frame.head(plan["limit"])
    if plan.get("select"):
        frame = frame[plan["select"]]
    return _format_output(frame.reset_index(drop=True), plan["output"])


def _format_output(frame: pd.DataFrame, output: dict[str, Any]) -> dict[str, Any]:
    answer_type = output["type"]
    if answer_type == "abstain":
        return {"type": "abstain"}
    if answer_type == "unanswerable":
        result = {"type": "unanswerable"}
        if isinstance(output.get("reason"), str):
            result["reason"] = output["reason"]
        return result
    if answer_type == "table":
        columns = output["columns"]
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Table output columns unavailable: {sorted(missing)}")
        rows = [[_clean_scalar(value) for value in row] for row in frame[columns].itertuples(index=False, name=None)]
        return {"type": "table", "answer": {"columns": columns, "rows": rows}}
    column = output["column"]
    if column not in frame.columns:
        raise ValueError(f"Output column unavailable: {column}")
    values = [_clean_scalar(value) for value in frame[column].tolist()]
    values = [value for value in values if value is not None]
    if answer_type == "list":
        return {"type": answer_type, "answer": values}
    if not values:
        raise ValueError("Plan produced no non-null value")
    value = values[0]
    if answer_type == "boolean" and output.get("compare"):
        compare = output["compare"]
        value = _compare(value, compare["op"], compare["value"])
    elif answer_type == "boolean":
        if isinstance(value, str):
            normalized = _normalized_text(value)
            if normalized in {"true", "1", "yes", "co", "có", "con", "còn", "da", "đã"}:
                value = True
            elif normalized in {"false", "0", "no", "khong", "không", "chua", "chưa", "het", "hết"}:
                value = False
            else:
                raise ValueError(f"Cannot map {value!r} to a boolean")
        elif isinstance(value, (int, float, bool)):
            value = bool(value)
        else:
            raise ValueError(f"Cannot map {type(value).__name__} to a boolean")
    elif answer_type == "date":
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
        if pd.isna(parsed):
            raise ValueError(f"Output is not a parseable date: {value!r}")
        value = parsed.date().isoformat()
    elif answer_type == "percent":
        decimals = output.get("decimals", 2)
        quantum = Decimal(1).scaleb(-decimals)
        value = float(Decimal(str(_number(value))).quantize(quantum, rounding=ROUND_HALF_UP))
    elif answer_type == "number":
        number = _number(value)
        decimals = output.get("decimals", 0 if number.is_integer() else 2)
        quantum = Decimal(1).scaleb(-decimals)
        rounded = Decimal(str(number)).quantize(quantum, rounding=ROUND_HALF_UP)
        value = int(rounded) if decimals == 0 else float(rounded)
    return {"type": answer_type, "answer": _clean_scalar(value)}
