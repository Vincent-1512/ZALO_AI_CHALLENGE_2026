"""Cheap, offline sanity checks before predict.py. Never blocks the run (prints warnings only)."""
import json
import os
import sys
from pathlib import Path

import zac_runtime as rt


def main() -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rt.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    warnings = []
    for name in (rt.ASR_DIRNAME, rt.LLM_DIRNAME):
        folder = rt.MODELS_DIR / name
        if not any(folder.glob("*.safetensors")):
            warnings.append(f"missing weights: {folder}")
    try:
        split = rt.detect_split(rt.DATA_ROOT)
    except Exception as error:   # noqa: BLE001
        print(f"WARNING: {error}", file=sys.stderr)
        return 0
    items = rt.read_items(rt.DATA_ROOT, split)
    missing_audio = 0
    history_audio = 0
    for item in items:
        for ref in [item.get("audio")] + [t.get("audio") for t in item.get("history", []) if t.get("role") == "user"]:
            if ref:
                history_audio += 1
                if not (rt.DATA_ROOT / split / ref).is_file():
                    missing_audio += 1
    if missing_audio:
        warnings.append(f"{missing_audio} audio files referenced but not found")
    print(json.dumps({"split": split, "samples": len(items), "audio_refs": history_audio,
                      "warnings": warnings}, ensure_ascii=False))
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
