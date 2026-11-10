from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import string
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import load_catalog, read_jsonl, resolve_split
from lifemate.prompts import action_json, build_messages


def normalized_request(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\W+", " ", text, flags=re.UNICODE).strip()


def _generate_random_name(prefix: str, length: int, rng: random.Random) -> str:
    letters = string.ascii_lowercase + string.digits
    rand_part = "".join(rng.choices(letters, k=length))
    return f"{prefix}_{rand_part}"


def create_masked_variant(
    row: dict,
    catalog: dict,
    rng: random.Random,
) -> tuple[dict, dict]:
    """Hammer ICLR 2025 Function & Parameter Masking augmentation:
    Replaces function and parameter names with randomized identifiers
    while preserving descriptions and values to eliminate name-memorization bias.
    """
    masked_row = copy.deepcopy(row)
    masked_catalog = copy.deepcopy(catalog)

    fn_map: dict[str, str] = {}
    param_map: dict[str, dict[str, str]] = {}

    for app in masked_catalog.get("apps", []):
        for tool in app.get("tools", []):
            old_name = tool.get("name", "")
            if old_name and old_name != "sys.ask_missing_info":
                prefix = old_name.split(".", 1)[0] if "." in old_name else "fn"
                new_fn = f"{prefix}.{_generate_random_name('act', 5, rng)}"
                fn_map[old_name] = new_fn
                tool["name"] = new_fn

                p_dict = param_map.setdefault(old_name, {})
                props = tool.get("parameters", {}).get("properties", {})
                if isinstance(props, dict):
                    new_props = {}
                    for p_old, p_spec in props.items():
                        new_p = _generate_random_name("p", 4, rng)
                        p_dict[p_old] = new_p
                        new_props[new_p] = p_spec
                    tool["parameters"]["properties"] = new_props
                    if "required" in tool.get("parameters", {}):
                        tool["parameters"]["required"] = [
                            p_dict.get(r, r) for r in tool["parameters"]["required"]
                        ]

    context = masked_row.get("context", {})
    if "installed_apps" in context and isinstance(context["installed_apps"], dict):
        for app_data in context["installed_apps"].values():
            if isinstance(app_data, dict) and "tools" in app_data and isinstance(app_data["tools"], list):
                app_data["tools"] = [fn_map.get(t, t) for t in app_data["tools"]]

    labels = masked_row.get("label", {})
    for call in labels.get("tool_calls", []):
        old_fn = call.get("name", "")
        if old_fn in fn_map:
            call["name"] = fn_map[old_fn]
            p_dict = param_map.get(old_fn, {})
            old_args = call.get("arguments", {})
            if isinstance(old_args, dict):
                call["arguments"] = {p_dict.get(k, k): v for k, v in old_args.items()}

    masked_row["id"] = f"{row['id']}_hammer_masked"
    return masked_row, masked_catalog


def split_rows(rows: list[dict], seed: int, ratios: tuple[float, float, float]):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[normalized_request(row.get("audio_text", ""))].append(row)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    total = len(rows)
    train_goal = round(total * ratios[0])
    dev_goal = round(total * ratios[1])
    partitions = {"train": [], "dev": [], "holdout": []}
    for group in shuffled:
        target = "train" if len(partitions["train"]) < train_goal else None
        if target is None:
            target = "dev" if len(partitions["dev"]) < dev_goal else "holdout"
        partitions[target].extend(group)
    return partitions


def make_records(rows: list[dict], catalog: dict) -> list[dict]:
    records = []
    for row in rows:
        calls = row["label"].get("tool_calls", [])
        results = row["label"].get("tool_results", [])
        if len(calls) != len(results):
            raise ValueError(f"Gold calls/results count differs for sample {row['id']}")
        for step in range(len(calls) + 1):
            messages = build_messages(
                row,
                row.get("audio_text", ""),
                catalog,
                prior_calls=calls[:step],
                prior_results=results[:step],
            )
            target = {"action": "stop"} if step == len(calls) else {
                "action": "call",
                "name": calls[step]["name"],
                "arguments": calls[step].get("arguments", {}),
            }
            records.append({"sample_id": row["id"], "messages": messages, "target": action_json(target)})
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_partitions(
    output_dir: Path,
    partitions: dict,
    catalog: dict,
    seed: int,
    ratios: tuple,
    augment_masking: bool = False,
    mask_ratio: float = 0.5,
):
    manifest = {
        "seed": seed,
        "ratios": ratios,
        "augment_masking": augment_masking,
        "samples": {},
        "records": {},
    }
    rng = random.Random(seed)
    for name, subset in partitions.items():
        records = make_records(subset, catalog)
        if augment_masking and name == "train":
            augmented_records = []
            for row in subset:
                if rng.random() < mask_ratio:
                    masked_row, masked_catalog = create_masked_variant(row, catalog, rng)
                    augmented_records.extend(make_records([masked_row], masked_catalog))
            records.extend(augmented_records)
            manifest["augmented_records"] = len(augmented_records)

        write_jsonl(output_dir / f"{name}.jsonl", records)
        manifest["samples"][name] = len(subset)
        manifest["records"][name] = len(records)
        ids = "\n".join(sorted(row["id"] for row in subset)).encode("utf-8")
        manifest[f"{name}_id_sha256"] = hashlib.sha256(ids).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train-only LifeMate next-action SFT records.")
    parser.add_argument("--data-root", default="data/raw/train+public_test")
    parser.add_argument("--output-dir", default="data/processed/sft")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument(
        "--augment-masking",
        action="store_true",
        default=False,
        help="Enable Hammer ICLR 2025 Function & Parameter Masking augmentation for train partition",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Ratio of train samples to augment with randomized masking (default 0.5)",
    )
    args = parser.parse_args()
    ratios = (args.train_ratio, args.dev_ratio, args.holdout_ratio)
    if abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) <= 0:
        raise ValueError("Split ratios must be positive and sum to 1")

    split = resolve_split(args.data_root, "train")
    catalog = load_catalog(split.split_dir)
    rows = list(read_jsonl(split.split_dir / "train.jsonl"))
    partitions = split_rows(rows, args.seed, ratios)
    output_dir = Path(args.output_dir)
    manifest = write_partitions(
        output_dir,
        partitions,
        catalog,
        args.seed,
        ratios,
        augment_masking=args.augment_masking,
        mask_ratio=args.mask_ratio,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
