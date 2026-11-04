from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import import_tool_env, load_catalog, read_jsonl, resolve_split
from lifemate.models import SpeechRecognizer, ToolAgent
from lifemate.runner import infer_one, set_deterministic_seed


def tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", value).lower()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def token_f1(predicted: str, gold: str) -> float:
    left, right = tokens(predicted), tokens(gold)
    if not left or not right:
        return float(left == right)
    overlap = sum(min(left.count(token), right.count(token)) for token in set(left))
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def calls_match(predicted: dict, gold: dict, catalog_index: dict[str, dict]) -> bool:
    if predicted.get("name") != gold.get("name"):
        return False
    predicted_args, gold_args = predicted.get("arguments", {}), gold.get("arguments", {})
    if set(predicted_args) != set(gold_args):
        return False
    properties = catalog_index.get(gold["name"], {}).get("parameters", {}).get("properties", {})
    for key, expected in gold_args.items():
        actual = predicted_args[key]
        if properties.get(key, {}).get("x-match") == "fuzzy":
            if not isinstance(actual, str) or not isinstance(expected, str) or token_f1(actual, expected) <= 0.6:
                return False
        elif actual != expected:
            return False
    return True


def maximum_matching(predicted: list[dict], gold: list[dict], catalog_index: dict) -> int:
    edges = [[i for i, target in enumerate(gold) if calls_match(call, target, catalog_index)] for call in predicted]
    matched = [-1] * len(gold)

    def assign(prediction_index: int, visited: set[int]) -> bool:
        for gold_index in edges[prediction_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if matched[gold_index] == -1 or assign(matched[gold_index], visited):
                matched[gold_index] = prediction_index
                return True
        return False

    return sum(assign(index, set()) for index in range(len(predicted)))


def changed_apps_match(predicted_state: dict, gold_state: dict, state_hash) -> bool:
    return all(state_hash({app: predicted_state.get(app)}) == state_hash({app: state}) for app, state in gold_state.items())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one model on the fixed train-dev split.")
    parser.add_argument("--data-root", default="data/raw/train+public_test")
    parser.add_argument("--sft-dev", default="data/processed/sft/dev.jsonl")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--asr-model")
    parser.add_argument("--llm-model")
    parser.add_argument("--adapter")
    parser.add_argument("--report", default="reports/runs/dev_metrics.json")
    return parser.parse_args()


def load_evaluation_data(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_deterministic_seed(config["generation"]["seed"])
    split = resolve_split(args.data_root, "train")
    tool_env = import_tool_env(split.split_dir)
    from zaic_tools import state_hash

    catalog = load_catalog(split.split_dir)
    index = {tool["name"]: tool for app in catalog["apps"] for tool in app["tools"]}
    dev_ids = {row["sample_id"] for row in read_jsonl(Path(args.sft_dev))}
    rows = [row for row in read_jsonl(split.split_dir / "train.jsonl") if row["id"] in dev_ids]
    if not rows or {row["id"] for row in rows} != dev_ids:
        raise ValueError("The SFT dev IDs do not match the available train samples")
    return config, split, tool_env, catalog, index, state_hash, rows


def load_agent(config: dict, args):
    asr_config, llm_config = config["models"]["asr"], config["models"]["llm"]
    asr = SpeechRecognizer(
        args.asr_model or asr_config["local_path"],
        batch_size=config["limits"]["asr_batch_size"],
        sample_rate=config["limits"]["audio_sample_rate"],
    )
    agent = ToolAgent(
        args.llm_model or llm_config["local_path"],
        args.adapter or config["models"].get("adapter_path"),
        min(config["limits"]["max_context_tokens"], config["limits"]["hard_context_limit"]),
        config["limits"]["max_new_tokens"],
    )
    return asr, agent


def evaluate_sample(sample, runtime):
    tool_env, split, catalog, index, state_hash, asr, agent, max_steps = runtime
    profile = {}
    calls, _, env = infer_one(
        sample, tool_env, split, catalog, asr, agent, max_steps,
        return_env_for_train_eval=True, profile=profile,
    )
    expected = sample["label"]["tool_calls"]
    return {
        "state_match": changed_apps_match(env.get_state(), sample["label"]["final_state"], state_hash),
        "matched_calls": maximum_matching(calls, expected, index),
        "predicted_calls": len(calls),
        "gold_calls": len(expected),
        "oos_exact": not calls if not expected else None,
        "profile": profile,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return round(ordered[index], 4)


def latency_summary(profiles: list[dict]) -> dict:
    latency = {}
    for phase in ("asr_seconds", "policy_seconds", "tool_seconds", "total_seconds"):
        values = [row.get(phase, 0.0) for row in profiles]
        latency[phase] = {
            "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99), "max": round(max(values), 4),
        }
    return latency


def build_report(results: list[dict]) -> dict:
    samples = len(results)
    matches = sum(row["matched_calls"] for row in results)
    predictions = sum(row["predicted_calls"] for row in results)
    gold = sum(row["gold_calls"] for row in results)
    precision = matches / predictions if predictions else 0.0
    recall = matches / gold if gold else 0.0
    oos = [row["oos_exact"] for row in results if row["oos_exact"] is not None]
    profiles = [row["profile"] for row in results]
    return {
        "split": "train-dev",
        "samples": samples,
        "changed_app_state_match_rate": sum(row["state_match"] for row in results) / samples,
        "matched_calls": matches,
        "predicted_calls": predictions,
        "gold_calls": gold,
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "tool_call_f1_diagnostic": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "oos_samples": len(oos),
        "oos_exact_rate": sum(oos) / len(oos) if oos else None,
        "latency_seconds": latency_summary(profiles),
        "note": "Call F1 is a local diagnostic; the full official combined-score formula is not available in text form.",
    }


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Train-dev rollout requires a CUDA GPU.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config, split, tool_env, catalog, index, state_hash, rows = load_evaluation_data(args)
    asr, agent = load_agent(config, args)
    runtime = (tool_env, split, catalog, index, state_hash, asr, agent, config["limits"]["max_tool_steps"])
    report = build_report([evaluate_sample(row, runtime) for row in rows])
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
