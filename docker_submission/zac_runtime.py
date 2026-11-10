"""Shared runtime for predict.py and predict_notebook.ipynb.

Everything here is offline and deterministic. Hyperparameters live in this file
(the Docker Guide requires defaults inside the code, not on the command line).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# ---------------------------------------------------------------- defaults
SEED = 2026
MAX_CONTEXT_TOKENS = 8192      # prompt+answer budget used in training (hard limit is 16,384)
MAX_NEW_TOKENS = 192
MAX_TOOL_STEPS = 8
ASR_BATCH_SIZE = 4
AUDIO_SAMPLE_RATE = 16000

DATA_ROOT = Path(os.environ.get("ZAC_DATA", "/data"))
RESULT_DIR = Path(os.environ.get("ZAC_RESULT", "/result"))
MODELS_DIR = Path(os.environ.get("ZAC_MODELS", str(CODE_DIR / "saved_models")))
ASR_DIRNAME = "asr"   # PhoWhisper-large fine-tuned (merged), safetensors
LLM_DIRNAME = "llm"   # Qwen3-4B-Instruct-2507 with the LoRA adapter merged, safetensors


def seed_everything(seed: int = SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_split(root: Path) -> str:
    """The split directory is the one that holds <name>_tools_catalog.json (as in the Docker Guide)."""
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and (child / f"{child.name}_tools_catalog.json").exists():
            return child.name
    raise FileNotFoundError(f"No split with *_tools_catalog.json under {root}")


def read_items(root: Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    items = []
    with open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                items.append(json.loads(line))
                if limit and len(items) >= limit:
                    break
    return items


@dataclass
class Runtime:
    root: Path
    split: str
    paths: Any
    tool_env: Any
    catalog: dict
    asr: Any
    agent: Any
    max_steps: int = MAX_TOOL_STEPS
    infer_fn: Callable | None = None   # tests inject a fake; None -> lifemate.runner.infer_one
    extra: dict = field(default_factory=dict)


def load_models(root: Path = DATA_ROOT, models_dir: Path = MODELS_DIR, allow_cpu: bool = False) -> Runtime:
    """Load ASR + LLM from local disk only. Raises if CUDA is missing (30 s/sample needs a GPU)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Wall-clock guard (grader limit 30 s/sample): no new agent step after 22 s of the policy loop. Measured on a 3090: p99 total 21.5 s.
    os.environ.setdefault("LIFEMATE_LOOP_BUDGET_S", "18")
    import torch

    if not torch.cuda.is_available() and not (allow_cpu or os.environ.get("ZAC_ALLOW_CPU") == "1"):
        raise RuntimeError("CUDA is required; set ZAC_ALLOW_CPU=1 only for local debugging.")
    from lifemate.data import import_tool_env, load_catalog, resolve_split
    from lifemate.models import SpeechRecognizer, ToolAgent

    split = detect_split(root)
    paths = resolve_split(root, split)
    tool_env = import_tool_env(paths.split_dir)       # zaic_tools comes from /data, not from the image
    catalog = load_catalog(paths.split_dir)
    asr = SpeechRecognizer(Path(models_dir) / ASR_DIRNAME, batch_size=ASR_BATCH_SIZE, sample_rate=AUDIO_SAMPLE_RATE)
    agent = ToolAgent(Path(models_dir) / LLM_DIRNAME, None, MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS)
    return Runtime(Path(root), split, paths, tool_env, catalog, asr, agent)


def _fallback_hash(runtime: Runtime, sample_id: str):
    try:
        env = runtime.tool_env(runtime.split, sample_id, root=runtime.paths.env_root)
        return env.get_state_hash()
    except Exception:   # noqa: BLE001 - never let one sample stop the whole run
        return None


def run_sample(item: dict[str, Any], runtime: Runtime) -> dict[str, Any]:
    """One sample -> {"id","tool_calls","state_hash","time"} ; time in ms, measured before audio is read."""
    infer = runtime.infer_fn
    if infer is None:
        from lifemate.runner import infer_one as infer
    started = time.perf_counter()
    try:
        tool_calls, state_hash = infer(
            item, runtime.tool_env, runtime.paths, runtime.catalog,
            runtime.asr, runtime.agent, runtime.max_steps,
        )
    except Exception as error:   # noqa: BLE001
        print(f"sample={item.get('id')} fatal={type(error).__name__}: {error}", file=sys.stderr)
        tool_calls, state_hash = [], _fallback_hash(runtime, item["id"])
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash, "time": elapsed_ms}


def predict_all(runtime: Runtime, items: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(output_path, "w", encoding="utf-8") as stream:
        for item in items:
            record = run_sample(item, runtime)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            results.append(record)
            print(f"processed={record['id']} ms={record['time']}", file=sys.stderr, flush=True)
    return results
