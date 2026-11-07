from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

PLAN_VERSION = 1
PLAN_KEYS = {"version", "tables", "joins", "filters", "group_by", "metrics", "derive", "select", "sort", "limit", "output"}
FILTER_OPS = {"eq", "ne", "contains", "not_contains", "in", "not_in", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"}
METRIC_OPS = {"count_rows", "count", "sum", "mean", "min", "max", "nunique"}
SORT_DIRECTIONS = {"asc", "desc"}
OUTPUT_TYPES = {"number", "percent", "boolean", "string", "list", "table", "date", "unanswerable", "abstain"}


def _strict_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"{label}: unsupported keys {sorted(extra)}")


def parse_plan(text: str) -> dict[str, Any]:
    """Parse a single JSON plan, allowing only a surrounding Markdown fence."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response does not contain a JSON object")
    try:
        plan = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid plan JSON: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    return plan


def normalize_plan(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize common equivalent model outputs without guessing ambiguous fields."""
    result = json.loads(json.dumps(plan))
    tables = {table["table"] for table in catalog}
    columns_by_table = {
        table["table"]: list(table["columns"])
        for table in catalog
    }
    columns = {column for values in columns_by_table.values() for column in values}

    def resolve_column(value: Any) -> Any:
        if not isinstance(value, str) or value in columns:
            return value
        if "." not in value:
            return value
        table_id, name = value.split(".", 1)
        if table_id not in columns_by_table:
            return value
        candidates = columns_by_table[table_id]
        # Match only a unique safe-name variant after case/diacritic/punctuation
        # normalization. This catches harmless header spelling differences.
        from .io import _slug
        key = _slug(name).removesuffix("_1")
        matches = [column for column in candidates
                   if _slug(column.split(".", 1)[1]).removesuffix("_1") == key]
        return matches[0] if len(matches) == 1 else value

    joins = []
    for join in result.get("joins", []):
        if not isinstance(join, dict):
            joins.append(join)
            continue
        item = dict(join)
        if "left" not in item and "left_table" in item and "left_column" in item:
            table_id, column = item.pop("left_table"), item.pop("left_column")
            item["left"] = column if isinstance(column, str) and "." in column else f"{table_id}.{column}"
        if "right" not in item and "right_table" in item and "right_column" in item:
            table_id, column = item.pop("right_table"), item.pop("right_column")
            item["right"] = column if isinstance(column, str) and "." in column else f"{table_id}.{column}"
        # This executor implements inner joins only. Drop redundant inner markers;
        # preserve unsupported semantics so strict validation can reject them.
        if isinstance(item.get("type", "inner"), str) and item.get("type", "inner").lower() == "inner":
            item.pop("type", None)
        if isinstance(item.get("op", "inner"), str) and item.get("op", "inner").lower() == "inner":
            item.pop("op", None)
        if "left" in item:
            item["left"] = resolve_column(item["left"])
        if "right" in item:
            item["right"] = resolve_column(item["right"])
        joins.append(item)
    result["joins"] = joins

    # Join endpoints identify tables omitted from the model's tables list.
    used = list(result.get("tables", [])) if isinstance(result.get("tables", []), list) else []
    for join in joins:
        if isinstance(join, dict):
            for key in ("left", "right"):
                ref = join.get(key)
                if isinstance(ref, str) and "." in ref and ref.split(".", 1)[0] in tables:
                    table_id = ref.split(".", 1)[0]
                    if table_id not in used:
                        used.append(table_id)
    result["tables"] = used

    for condition in result.get("filters", []):
        if isinstance(condition, dict) and "column" in condition:
            condition["column"] = resolve_column(condition["column"])
    for key in ("group_by", "select"):
        if isinstance(result.get(key), list):
            result[key] = [resolve_column(value) for value in result[key]]
    for metric in result.get("metrics", []):
        if isinstance(metric, dict) and "column" in metric:
            metric["column"] = resolve_column(metric["column"])
    for order in result.get("sort", []):
        if isinstance(order, dict) and "column" in order:
            order["column"] = resolve_column(order["column"])

    for derived in result.get("derive", []):
        if isinstance(derived, dict):
            op = derived.get("op")
            if op in {"percent", "percentage", "pct_change", "percentage_change"}:
                derived["op"] = "percent_change"
    return result


def validate_plan(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the constrained relational plan before execution."""
    _strict_keys(plan, PLAN_KEYS, "plan")
    if type(plan.get("version")) is not int or plan.get("version") != PLAN_VERSION:
        raise ValueError(f"Plan version must be {PLAN_VERSION}")
    tables = plan.get("tables")
    if not isinstance(tables, list) or not tables or not all(isinstance(x, str) for x in tables):
        raise ValueError("tables must be a non-empty list of table IDs")
    known_tables = {table["table"] for table in catalog}
    if len(set(tables)) != len(tables) or set(tables) - known_tables:
        raise ValueError(f"Unknown or duplicate table IDs: {sorted(set(tables) - known_tables)}")
    known_columns = {column for table in catalog for column in table["columns"]}
    if not isinstance(plan.get("joins", []), list) or not isinstance(plan.get("filters", []), list):
        raise ValueError("joins and filters must be lists")
    if not isinstance(plan.get("group_by", []), list) or not isinstance(plan.get("metrics", []), list):
        raise ValueError("group_by and metrics must be lists")
    if not isinstance(plan.get("derive", []), list):
        raise ValueError("derive must be a list")
    if not isinstance(plan.get("select", []), list) or not isinstance(plan.get("sort", []), list):
        raise ValueError("select and sort must be lists")
    if not all(isinstance(value, str) for value in plan.get("group_by", []) + plan.get("select", [])):
        raise ValueError("group_by and select entries must be column-name strings")
    if len(set(plan.get("group_by", []))) != len(plan.get("group_by", [])):
        raise ValueError("group_by columns must be unique")
    if len(set(plan.get("select", []))) != len(plan.get("select", [])):
        raise ValueError("select columns must be unique")
    used_tables = set(tables)

    def check_column(column: Any, label: str) -> None:
        if not isinstance(column, str) or column not in known_columns:
            raise ValueError(f"{label} references unknown column {column!r}")
        table_id = column.split(".", 1)[0]
        if table_id not in used_tables:
            raise ValueError(f"{label} references {table_id}, which is not in tables")

    for join in plan.get("joins", []):
        if not isinstance(join, dict):
            raise ValueError("Each join must be an object")
        _strict_keys(join, {"left", "right"}, "join")
        check_column(join.get("left"), "join.left")
        check_column(join.get("right"), "join.right")
        if join["left"].split(".", 1)[0] == join["right"].split(".", 1)[0]:
            raise ValueError("Join columns must come from different tables")
    for condition in plan.get("filters", []):
        if not isinstance(condition, dict):
            raise ValueError("Each filter must be an object")
        _strict_keys(condition, {"column", "op", "value"}, "filter")
        check_column(condition.get("column"), "filter.column")
        if condition.get("op") not in FILTER_OPS:
            raise ValueError(f"Unsupported filter op {condition.get('op')!r}")
        if condition["op"] not in {"is_null", "not_null"} and "value" not in condition:
            raise ValueError(f"filter {condition['op']} requires value")
        if condition["op"] in {"in", "not_in"} and not isinstance(condition.get("value"), list):
            raise ValueError(f"filter {condition['op']} requires a list value")
        if condition["op"] in {"between"}:
            value = condition.get("value")
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("between filter value must contain exactly two values")
    for column in plan.get("group_by", []):
        check_column(column, "group_by/select")
    for metric in plan.get("metrics", []):
        if not isinstance(metric, dict):
            raise ValueError("Each metric must be an object")
        _strict_keys(metric, {"op", "column", "as"}, "metric")
        op = metric.get("op")
        if op not in METRIC_OPS:
            raise ValueError(f"Unsupported metric op {op!r}")
        column = metric.get("column")
        if op == "count_rows":
            if column is not None:
                raise ValueError("count_rows does not take a column")
        else:
            check_column(column, "metric.column")
        alias = metric.get("as")
        if not isinstance(alias, str) or not alias or "." in alias:
            raise ValueError("metric.as must be a non-empty output name without dots")
    metric_aliases = {metric["as"] for metric in plan.get("metrics", [])}
    if len(metric_aliases) != len(plan.get("metrics", [])):
        raise ValueError("Metric output names must be unique")
    if metric_aliases & set(plan.get("group_by", [])):
        raise ValueError("Metric output names cannot collide with group_by columns")
    available_after_aggregate = set(plan.get("group_by", [])) | metric_aliases
    derived_aliases = set()
    for derived in plan.get("derive", []):
        if not isinstance(derived, dict):
            raise ValueError("Each derived metric must be an object")
        _strict_keys(derived, {"op", "left", "right", "as"}, "derive")
        if derived.get("op") not in {"add", "subtract", "multiply", "divide", "percent_change"}:
            raise ValueError(f"Unsupported derive op {derived.get('op')!r}")
        if derived.get("left") not in available_after_aggregate | derived_aliases:
            raise ValueError(f"Unknown derived input {derived.get('left')!r}")
        if derived.get("right") not in available_after_aggregate | derived_aliases:
            raise ValueError(f"Unknown derived input {derived.get('right')!r}")
        alias = derived.get("as")
        if not isinstance(alias, str) or not alias or alias in available_after_aggregate | derived_aliases:
            raise ValueError("derive.as must be a new non-empty output name")
        derived_aliases.add(alias)
    select_candidates = available_after_aggregate | derived_aliases if plan.get("metrics") else known_columns
    for column in plan.get("select", []):
        if column not in select_candidates:
            raise ValueError(f"select references unavailable output column {column!r}")
    for order in plan.get("sort", []):
        if not isinstance(order, dict):
            raise ValueError("Each sort item must be an object")
        _strict_keys(order, {"column", "direction"}, "sort")
        column = order.get("column")
        if not isinstance(column, str):
            raise ValueError("sort.column must be a string")
        if column not in select_candidates:
            raise ValueError(f"sort references unknown column {column!r}")
        if column in known_columns:
            check_column(column, "sort.column")
        if order.get("direction") not in SORT_DIRECTIONS:
            raise ValueError("sort.direction must be asc or desc")
    limit = plan.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 1000):
        raise ValueError("limit must be an integer from 1 to 1000")
    output = plan.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    _strict_keys(output, {"type", "column", "columns", "compare", "reason", "decimals"}, "output")
    if output.get("type") not in OUTPUT_TYPES:
        raise ValueError(f"Unsupported output type {output.get('type')!r}")
    if output.get("type") in {"number", "percent", "boolean", "string", "date", "list"}:
        if not isinstance(output.get("column"), str):
            raise ValueError("Scalar/list output requires output.column")
        if output["column"] not in select_candidates:
            raise ValueError(f"output references unavailable column {output['column']!r}")
    if output.get("type") == "table":
        columns = output.get("columns")
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
            raise ValueError("table output requires a non-empty columns list")
        if len(set(columns)) != len(columns):
            raise ValueError("table output columns must be unique")
        if set(columns) - select_candidates:
            raise ValueError(f"table output references unavailable columns: {sorted(set(columns) - select_candidates)}")
    if "decimals" in output and (not isinstance(output["decimals"], int) or isinstance(output["decimals"], bool) or not 0 <= output["decimals"] <= 6):
        raise ValueError("output.decimals must be an integer from 0 to 6")
    compare = output.get("compare")
    if compare is not None:
        if output.get("type") != "boolean" or not isinstance(compare, dict):
            raise ValueError("compare is only allowed for boolean output")
        _strict_keys(compare, {"op", "value"}, "output.compare")
        if compare.get("op") not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
            raise ValueError("Unsupported output comparison")
        if "value" not in compare:
            raise ValueError("output.compare.value is required")
    return plan


def _tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    stop = {"la", "va", "cho", "cua", "trong", "bao", "nhieu", "gi", "nao", "co", "the", "mot", "cac", "nhung", "voi", "shop", "hang", "so", "nao"}
    return {token for token in tokens if len(token) > 1 and token not in stop}


def _select_catalog(catalog: list[dict[str, Any]], question: str, budget: int) -> str:
    query_tokens = _tokens(question)
    ranked = []
    for index, table in enumerate(catalog):
        text = " ".join([table["table"], table["file"], table["sheet"], *table["original_columns"]])
        table_tokens = _tokens(text)
        score = sum(2 if token in table_tokens else 1 for token in query_tokens if token in table_tokens)
        ranked.append((score, index, table))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    summary_budget = min(1200, budget // 4)
    summaries = []
    summary_chars = 0
    for _, _, table in ranked:
        item = f"{table['table']}={table['file']}/{table['sheet']}"
        if summary_chars + len(item) > summary_budget:
            break
        summaries.append(item)
        summary_chars += len(item)
    details = []
    chars = 0
    detail_count = min(12, len(ranked))
    for _, _, table in ranked[:detail_count]:
        block = f"TABLE {table['table']} | {table['file']} | {table['sheet']} | rows≈{table['rows']}\n"
        block += ", ".join(f"{name} ({original})" for name, original in zip(table["columns"], table["original_columns"]))
        if summary_chars + chars + len(block) > budget:
            break
        details.append(block)
        chars += len(block)
    summary = "TABLE INDEX (ranked candidates): " + "; ".join(summaries)
    return summary + "\n\nCANDIDATE TABLE DETAILS:\n" + "\n\n".join(details)


def _relevant_rules(rulebook: str, question: str, budget: int = 4500) -> str:
    sections = re.split(r"(?=^## )", rulebook, flags=re.MULTILINE)
    query_tokens = _tokens(question)
    required = ("## 2.", "## 7.", "## 8.", "## 9.")
    ranked = []
    for index, section in enumerate(sections):
        if not section.strip():
            continue
        title = section.splitlines()[0]
        score = sum(token in _tokens(section) for token in query_tokens)
        ranked.append((score, index, title, section))
    chosen = [item for item in ranked if any(item[2].startswith(prefix) for prefix in required)]
    chosen_ids = {item[1] for item in chosen}
    for item in sorted(ranked, key=lambda value: (-value[0], value[1])):
        if item[1] not in chosen_ids and len(chosen) < 6:
            chosen.append(item)
            chosen_ids.add(item[1])
    chosen.sort(key=lambda item: item[1])
    output, chars = [], 0
    for _, _, _, section in chosen:
        if chars + len(section) <= budget:
            output.append(section.strip())
            chars += len(section)
    return "\n\n".join(output)


def make_prompt(question: dict[str, Any], package: Any, max_catalog_chars: int = 6000) -> list[dict[str, str]]:
    catalog = _select_catalog(package.catalog, question["question"], max_catalog_chars)
    rules = _relevant_rules(package.rulebook, question["question"])
    user = (
        f"AS_OF_DATE: {question['as_of_date']}\nBUSINESS_ID: {package.business_id}\n"
        f"QUESTION: {question['question']}\n\nSHOP README:\n{package.readme[:1800]}\n\n"
        f"RELEVANT RULEBOOK SECTIONS:\n{rules}\n\nTABLE CATALOG:\n{catalog}\n\n"
        "A join has ONLY the keys left and right, each a fully qualified column; for example {\"left\":\"table_a.key\",\"right\":\"table_b.key\"}. Do not use type, op, left_table, right_table, left_column, or right_column in a join.\n"
        'For ordinary filters, value is one scalar or a list for in/not_in. For between, value MUST be a two-item list [lower, upper], for example {"column":"orders.date","op":"between","value":["2025-01-01","2025-01-31"]}.\n'
        "Return one JSON object only, following this schema exactly:\n"
        '{"version":1,"tables":["table_id"],"joins":[{"left":"table_id.column","right":"other_table.column"}],"filters":[{"column":"table_id.column","op":"eq|ne|contains|not_contains|in|not_in|gt|gte|lt|lte|between|is_null|not_null","value":"scalar or list; between requires [lower,upper]"}],'
        '"group_by":[],"metrics":[{"op":"count_rows|count|sum|mean|min|max|nunique","column":"table_id.column or null for count_rows","as":"value"}],'
        '"derive":[{"op":"add|subtract|multiply|divide|percent_change","left":"metric alias","right":"metric alias","as":"result"}],'
        '"select":[],"sort":[{"column":"qualified column or metric alias","direction":"asc|desc"}],"limit":null,'
        '"output":{"type":"number|percent|boolean|string|list|table|date|unanswerable|abstain","column":"output column or metric alias","decimals":0}}. '
        "Only use listed IDs and columns. Never write SQL or Python. If the released files cannot answer the question, use type unanswerable. "
        "Represent date boundaries as explicit ISO dates in filters. For percent change use (left-right)/right*100. "
        "Round money to 0 decimals, percentages and non-money averages to 2 decimals. "
        "For ranked ties, add an ascending ID sort; if there is no ID, sort by the exact name. "
        "For strings, lists and table names, use the exact spelling from the data or a valid alias."
    )
    system = "You are a deterministic spreadsheet question planner. Output valid JSON only. Do not include explanations or chain of thought."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
