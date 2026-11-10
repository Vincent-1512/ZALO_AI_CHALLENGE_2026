from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class SplitPaths:
    split_name: str
    split_dir: Path
    env_root: Path
    temporary_root: Path | None = None


def resolve_split(data_path: str | Path, split_name: str) -> SplitPaths:
    root = Path(data_path).resolve()
    nested = root / split_name
    split_dir = nested if (nested / f"{split_name}.jsonl").is_file() else root
    if not (split_dir / f"{split_name}.jsonl").is_file():
        raise FileNotFoundError(f"{split_name}.jsonl not found under {root}")
    if split_dir.name == split_name:
        return SplitPaths(split_name, split_dir, split_dir.parent)
    temp = Path(tempfile.mkdtemp(prefix="lifemate-data-root-"))
    (temp / split_name).symlink_to(split_dir, target_is_directory=True)
    return SplitPaths(split_name, split_dir, temp, temp)


def import_tool_env(split_dir: Path):
    split_path = str(split_dir.resolve())
    if split_path not in sys.path:
        sys.path.insert(0, split_path)
    from zaic_tools import ToolEnv

    return ToolEnv


def load_catalog(split_dir: Path) -> dict[str, Any]:
    path = split_dir / f"{split_dir.name}_tools_catalog.json"
    if not path.is_file():
        catalogs = sorted(split_dir.glob("*_tools_catalog.json"))
        if len(catalogs) != 1:
            raise FileNotFoundError(f"Could not identify one tools catalog in {split_dir}")
        path = catalogs[0]
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def resolve_audio_path(audio: str, split_dir: Path) -> Path:
    path = Path(audio)
    candidate = path if path.is_absolute() else split_dir / path
    if not candidate.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    return candidate
