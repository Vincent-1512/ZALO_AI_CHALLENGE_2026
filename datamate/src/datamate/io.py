from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BusinessPackage:
    business_id: str
    readme: str
    rulebook: str
    calendar_path: Path
    tables: dict[str, Any]
    catalog: list[dict[str, Any]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unnamed"


def _unique_columns(values: list[Any]) -> tuple[list[str], list[str]]:
    original = [str(value).strip() if value is not None else "" for value in values]
    counts: dict[str, int] = {}
    safe = []
    for index, value in enumerate(original):
        base = _slug(value) if value else f"column_{index + 1}"
        counts[base] = counts.get(base, 0) + 1
        safe.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return original, safe


def _header_score(row: list[Any]) -> float:
    cells = [value for value in row if value is not None and str(value).strip()]
    if not cells:
        return -1.0
    strings = sum(isinstance(value, str) for value in cells)
    unique = len({_slug(value) for value in cells})
    return len(cells) + 1.5 * strings + unique / len(cells)


def _find_header_row(frame: Any) -> int:
    candidates = min(20, len(frame))
    scores = [_header_score(frame.iloc[index].tolist()) for index in range(candidates)]
    return max(range(candidates), key=scores.__getitem__) if candidates else 0


def _read_csv(path: Path):
    import pandas as pd

    last_error = None
    for encoding in ("utf-8-sig", "cp1258", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=object, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise UnicodeError(f"Could not decode {path}") from last_error


def _read_excel_tables(path: Path, readme: str):
    import pandas as pd

    tables = []
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            raw = workbook.parse(sheet_name=sheet_name, header=None, dtype=object)
            if raw.empty:
                continue
            header_row = _find_header_row(raw)
            merge_header = "header gộp 2 dòng" in readme.lower() or "header hai dòng" in readme.lower()
            header_count = 2 if merge_header and header_row + 1 < len(raw) else 1
            header_values = []
            for column_index in range(raw.shape[1]):
                parts = []
                for offset in range(header_count):
                    value = raw.iat[header_row + offset, column_index]
                    if value is not None and str(value).strip() and str(value).strip() not in parts:
                        parts.append(str(value).strip())
                header_values.append(" / ".join(parts))
            original, safe = _unique_columns(header_values)
            frame = raw.iloc[header_row + header_count:].copy()
            frame.columns = safe
            frame = frame.dropna(how="all").reset_index(drop=True)
            if frame.empty:
                continue
            tables.append((sheet_name, frame, original, safe, header_row + header_count + 1))
    return tables


def _read_sqlite_tables(path: Path):
    import pandas as pd

    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    tables = []
    with sqlite3.connect(uri, uri=True) as connection:
        names = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        for name in names:
            escaped = name.replace('"', '""')
            frame = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
            original, safe = _unique_columns(list(frame.columns))
            frame.columns = safe
            tables.append((name, frame, original, safe, 1))
    return tables


def _qualify_frame(frame: Any, table_id: str, source_file: str, source_sheet: str, first_row: int):
    qualified = frame.copy()
    qualified.columns = [f"{table_id}.{column}" for column in qualified.columns]
    qualified["__source_file"] = source_file
    qualified["__source_sheet"] = source_sheet
    qualified["__source_row"] = range(first_row, first_row + len(qualified))
    return qualified


def _business_files(business_dir: Path) -> list[Path]:
    supported = {".csv", ".xlsx", ".sqlite", ".db"}
    return sorted(path for path in business_dir.iterdir()
                  if path.is_file() and path.suffix.lower() in supported)


def load_business_package(split_root: Path, business_id: str):
    business_dir = split_root / "data" / business_id
    if not business_dir.is_dir():
        raise FileNotFoundError(f"Business directory not found: {business_dir}")
    readme_path = business_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    rulebook_path = split_root / "rulebook.md"
    rulebook = rulebook_path.read_text(encoding="utf-8") if rulebook_path.exists() else ""
    calendar_path = split_root / "calendar.csv"
    tables: dict[str, Any] = {}
    catalog: list[dict[str, Any]] = []
    for file_index, path in enumerate(_business_files(business_dir), start=1):
        extracted = _extract_tables(path, readme)
        for table_index, (sheet, frame, original, safe, first_row) in enumerate(extracted, start=1):
            table_id = f"t{file_index:02d}_{table_index:02d}_{_slug(path.stem)[:30]}"
            tables[table_id] = _qualify_frame(frame, table_id, path.name, sheet, first_row)
            catalog.append({
                "table": table_id, "file": path.name, "sheet": sheet,
                "columns": [f"{table_id}.{column}" for column in safe],
                "original_columns": original, "rows": int(len(frame)),
            })
    if not catalog:
        raise ValueError(f"No readable tables found for {business_id}")
    return BusinessPackage(business_id, readme, rulebook, calendar_path, tables, catalog)


def _extract_tables(path: Path, readme: str):
    if path.suffix.lower() == ".csv":
        frame = _read_csv(path)
        original, safe = _unique_columns(list(frame.columns))
        frame.columns = safe
        return [(path.stem, frame, original, safe, 2)]
    if path.suffix.lower() == ".xlsx":
        return _read_excel_tables(path, readme)
    return _read_sqlite_tables(path)


def catalog_prompt(catalog: list[dict[str, Any]]) -> str:
    blocks = []
    for table in catalog:
        columns = list(zip(table["columns"], table["original_columns"]))
        rendered = ", ".join(f"{qualified} ({original})" for qualified, original in columns)
        blocks.append(f"TABLE {table['table']} | {table['file']} | {table['sheet']} | rows≈{table['rows']}\n{rendered}")
    return "\n\n".join(blocks)
