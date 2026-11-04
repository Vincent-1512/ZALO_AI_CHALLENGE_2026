from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from .data import import_tool_env, load_catalog, read_jsonl, resolve_audio_path, resolve_split
from .models import SpeechRecognizer, ToolAgent
from .prompts import build_messages


def set_deterministic_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _audio_inputs(sample: dict[str, Any], split_dir: Path):
    current = sample["audio"]
    history = [
        turn["audio"]
        for turn in sample.get("history", [])
        if turn.get("role") == "user" and turn.get("audio")
    ]
    refs = [current] + list(reversed(history))
    return refs, [resolve_audio_path(ref, split_dir) for ref in refs]


def _transcribe_sample(sample: dict, split_dir: Path, asr: SpeechRecognizer):
    refs, paths = _audio_inputs(sample, split_dir)
    recognized = asr.transcribe_many(paths)
    transcript_by_ref = dict(zip(refs, recognized))
    return transcript_by_ref[sample["audio"]], transcript_by_ref


def _valid_action(action: dict[str, Any] | None) -> bool:
    return bool(
        action
        and action.get("action") == "call"
        and isinstance(action.get("name"), str)
        and isinstance(action.get("arguments"), dict)
    )


def _next_action(agent, env, sample, transcript, catalog, transcripts, results, feedback=None):
    messages = build_messages(
        sample,
        transcript,
        catalog,
        transcripts=transcripts,
        prior_calls=env.get_tool_calls(),
        prior_results=results,
        feedback=feedback,
    )
    return agent.predict(messages)


def _validate_action(env, action, profile):
    started = time.perf_counter()
    validation = env.validate(action["name"], action["arguments"])
    if profile is not None:
        profile["tool_seconds"] += time.perf_counter() - started
    return validation


def _action_for_step(agent, env, sample, transcript, catalog, transcripts, results, profile):
    started = time.perf_counter()
    action = _next_action(agent, env, sample, transcript, catalog, transcripts, results)
    if profile is not None:
        profile["policy_seconds"] += time.perf_counter() - started
    if not _valid_action(action):
        return None
    validation = _validate_action(env, action, profile)
    if validation.get("ok"):
        return action
    feedback = json.dumps({"rejected_call": action, "schema_error": validation.get("error")}, ensure_ascii=False)
    started = time.perf_counter()
    corrected = _next_action(
        agent, env, sample, transcript, catalog, transcripts, results, feedback=feedback
    )
    if profile is not None:
        profile["policy_seconds"] += time.perf_counter() - started
    if not _valid_action(corrected):
        return None
    return corrected if _validate_action(env, corrected, profile).get("ok") else None


def _run_actions(env, sample, transcript, catalog, transcripts, prior_results, agent, max_steps, profile):
    for _ in range(max_steps):
        action = _action_for_step(
            agent, env, sample, transcript, catalog, transcripts, prior_results, profile
        )
        if action is None:
            return
        started = time.perf_counter()
        result = env.call(action["name"], action["arguments"])
        if profile is not None:
            profile["tool_seconds"] += time.perf_counter() - started
        prior_results.append(result)


def infer_one(
    sample,
    tool_env,
    split_paths,
    catalog,
    asr,
    agent,
    max_tool_steps,
    return_env_for_train_eval=False,
    profile=None,
):
    started = time.perf_counter()
    env = tool_env(split_paths.split_name, sample["id"], root=split_paths.env_root)
    if profile is not None:
        profile.update(asr_seconds=0.0, policy_seconds=0.0, tool_seconds=0.0)
    try:
        phase_started = time.perf_counter()
        transcript, transcripts = _transcribe_sample(sample, split_paths.split_dir, asr)
        if profile is not None:
            profile["asr_seconds"] = time.perf_counter() - phase_started
        _run_actions(
            env, sample, transcript, catalog, transcripts, [], agent, max_tool_steps, profile
        )
    except Exception as error:
        print(f"sample={sample['id']} error={type(error).__name__}: {error}", file=sys.stderr)
    result = (env.get_tool_calls(), env.get_state_hash())
    if profile is not None:
        profile["total_seconds"] = time.perf_counter() - started
    return (*result, env) if return_env_for_train_eval else result


def _load_runtime(data_path, split_name, config, asr_model, llm_model, adapter_path):
    paths = resolve_split(data_path, split_name)
    tool_env = import_tool_env(paths.split_dir)
    catalog = load_catalog(paths.split_dir)
    asr_cfg, llm_cfg = config["models"]["asr"], config["models"]["llm"]
    asr = SpeechRecognizer(
        asr_model or asr_cfg["local_path"],
        batch_size=config["limits"]["asr_batch_size"],
        sample_rate=config["limits"]["audio_sample_rate"],
    )
    agent = ToolAgent(
        llm_model or llm_cfg["local_path"],
        adapter_path if adapter_path is not None else config["models"].get("adapter_path"),
        min(config["limits"]["max_context_tokens"], config["limits"]["hard_context_limit"]),
        config["limits"]["max_new_tokens"],
    )
    return paths, tool_env, catalog, asr, agent


def _write_predictions(input_path, output_path, paths, tool_env, catalog, asr, agent, config):
    seen = set()
    expected = sum(1 for _ in read_jsonl(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for sample in read_jsonl(input_path):
            if sample["id"] in seen:
                raise ValueError(f"Duplicate sample id: {sample['id']}")
            seen.add(sample["id"])
            started = time.perf_counter()
            calls, state_hash = infer_one(
                sample, tool_env, paths, catalog, asr, agent,
                config["limits"]["max_tool_steps"],
            )
            record = {"id": sample["id"], "tool_calls": calls, "state_hash": state_hash}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"processed={sample['id']} seconds={time.perf_counter() - started:.3f}", file=sys.stderr)
    if len(seen) != expected:
        raise RuntimeError(f"Wrote {len(seen)} rows for {expected} input samples")


def run_inference(
    data_path: str | Path,
    split_name: str,
    output_path: str | Path,
    config: dict[str, Any],
    asr_model: str | Path | None = None,
    llm_model: str | Path | None = None,
    adapter_path: str | Path | None = None,
    allow_cpu: bool = False,
) -> None:
    import torch

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is required for the 30-second scoring runtime; pass --allow-cpu only for local debugging.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    set_deterministic_seed(config["generation"]["seed"])
    runtime = _load_runtime(data_path, split_name, config, asr_model, llm_model, adapter_path)
    paths, tool_env, catalog, asr, agent = runtime
    input_path = paths.split_dir / f"{split_name}.jsonl"
    _write_predictions(input_path, output_path, paths, tool_env, catalog, asr, agent, config)
