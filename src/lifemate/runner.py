from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .data import import_tool_env, load_catalog, read_jsonl, resolve_audio_path, resolve_split
from .models import SpeechRecognizer, ToolAgent
from .prompts import build_messages
from .asr_correct import correct_transcript


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


def _transcribe_sample(sample: dict, split_dir: Path, asr: SpeechRecognizer, asr_correct=False, catalog=None, profile=None):
    refs, paths = _audio_inputs(sample, split_dir)
    recognized = asr.transcribe_many(paths)
    transcript_by_ref = dict(zip(refs, recognized))
    if asr_correct:
        changes = []
        for ref, text in transcript_by_ref.items():
            corrected, replacements = correct_transcript(text, sample, catalog or {})
            transcript_by_ref[ref] = corrected
            if replacements:
                changes.append({"audio": ref, "replacements": replacements})
        if profile is not None:
            profile["asr_corrections"] = changes
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


def _normalized_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sample_actions(agent, env, sample, transcript, catalog, transcripts, results, count, temperature):
    messages = build_messages(sample, transcript, catalog, transcripts=transcripts,
                              prior_calls=env.get_tool_calls(), prior_results=results)
    if not hasattr(agent, "predict_candidates"):
        raise TypeError("Best-of-k requires an agent with predict_candidates()")
    return agent.predict_candidates(messages, count, temperature)


def _validate_action(env, action, profile):
    started = time.perf_counter()
    validation = env.validate(action["name"], action["arguments"])
    if profile is not None:
        profile["tool_seconds"] += time.perf_counter() - started
    return validation


def _action_for_step(agent, env, sample, transcript, catalog, transcripts, results, profile,
                     num_samples=1, temperature=None):
    started = time.perf_counter()
    selected_validation = None
    if num_samples == 1:
        action = _next_action(agent, env, sample, transcript, catalog, transcripts, results)
    else:
        candidates = _sample_actions(agent, env, sample, transcript, catalog, transcripts, results,
                                     num_samples, temperature or 0.7)
        valid = []
        for candidate in candidates:
            if candidate and candidate.get("action") == "stop":
                valid.append((candidate, None))
            elif _valid_action(candidate):
                validation = _validate_action(env, candidate, profile)
                if validation.get("ok"):
                    valid.append((candidate, validation))
        if profile is not None:
            profile["candidate_valid_count"] = profile.get("candidate_valid_count", 0) + len(valid)
            counts = Counter(_normalized_action(candidate) for candidate, _ in valid)
            profile.setdefault("candidate_agreement", []).append(
                max(counts.values()) / len(valid) if valid else 0.0
            )
        if valid:
            counts = Counter(_normalized_action(candidate) for candidate, _ in valid)
            first_by_key = {}
            for candidate, validation in valid:
                first_by_key.setdefault(_normalized_action(candidate), (candidate, validation))
            action, selected_validation = first_by_key[counts.most_common(1)[0][0]]
        else:
            action = _next_action(agent, env, sample, transcript, catalog, transcripts, results)
    if profile is not None:
        profile["policy_seconds"] += time.perf_counter() - started
    if not _valid_action(action):
        if profile is not None:
            stopped = bool(action) and action.get("action") == "stop"
            profile["end_reason"] = "stop" if stopped else "parse_failure"
        return None
    if action.get("action") == "stop":
        if profile is not None:
            profile["end_reason"] = "stop"
        return None
    validation = selected_validation or _validate_action(env, action, profile)
    if validation.get("ok"):
        return action
    if profile is not None:
        profile["schema_retries"] = profile.get("schema_retries", 0) + 1
    feedback = json.dumps({"rejected_call": action, "schema_error": validation.get("error")}, ensure_ascii=False)
    started = time.perf_counter()
    corrected = _next_action(
        agent, env, sample, transcript, catalog, transcripts, results, feedback=feedback
    )
    if profile is not None:
        profile["policy_seconds"] += time.perf_counter() - started
    if not _valid_action(corrected):
        if profile is not None:
            profile["end_reason"] = "retry_not_a_call"
        return None
    if _validate_action(env, corrected, profile).get("ok"):
        return corrected
    if profile is not None:
        profile["end_reason"] = "schema_rejected_after_retry"
    return None


def _run_actions(env, sample, transcript, catalog, transcripts, prior_results, agent, max_steps,
                 profile, num_samples=1, temperature=None):
    for _ in range(max_steps):
        action = _action_for_step(
            agent, env, sample, transcript, catalog, transcripts, prior_results, profile,
            num_samples, temperature,
        )
        if action is None:
            return
        started = time.perf_counter()
        result = env.call(action["name"], action["arguments"])
        if profile is not None:
            profile["tool_seconds"] += time.perf_counter() - started
        prior_results.append(result)
    if profile is not None:
        profile["end_reason"] = "max_steps"


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
    asr_correct=False,
    num_samples=1,
    temperature=None,
):
    started = time.perf_counter()
    env = tool_env(split_paths.split_name, sample["id"], root=split_paths.env_root)
    if profile is not None:
        profile.update(asr_seconds=0.0, policy_seconds=0.0, tool_seconds=0.0)
    try:
        phase_started = time.perf_counter()
        transcript, transcripts = _transcribe_sample(sample, split_paths.split_dir, asr,
                                                      asr_correct, catalog, profile)
        if profile is not None:
            profile["asr_seconds"] = time.perf_counter() - phase_started
            profile["transcript"] = transcript
            if num_samples > 1:
                profile["num_samples"] = num_samples
                profile["temperature"] = temperature or 0.7
                profile["candidate_valid_count"] = 0
                profile["candidate_agreement"] = []
            if asr_correct:
                profile.setdefault("asr_corrections", [])
        _run_actions(
            env, sample, transcript, catalog, transcripts, [], agent, max_tool_steps, profile,
            num_samples, temperature,
        )
    except Exception as error:
        print(f"sample={sample['id']} error={type(error).__name__}: {error}", file=sys.stderr)
        if profile is not None:
            profile["end_reason"] = f"exception:{type(error).__name__}"
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


def _write_predictions(input_path, output_path, paths, tool_env, catalog, asr, agent, config,
                       asr_correct=False, num_samples=1, temperature=None, per_sample_log=None):
    seen = set()
    expected = sum(1 for _ in read_jsonl(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_stream = Path(per_sample_log).open("w", encoding="utf-8") if per_sample_log else None
    with output.open("w", encoding="utf-8") as stream:
        for sample in read_jsonl(input_path):
            if sample["id"] in seen:
                raise ValueError(f"Duplicate sample id: {sample['id']}")
            seen.add(sample["id"])
            started = time.perf_counter()
            profile = {}
            calls, state_hash = infer_one(
                sample, tool_env, paths, catalog, asr, agent,
                config["limits"]["max_tool_steps"],
                profile=profile, asr_correct=asr_correct,
                num_samples=num_samples, temperature=temperature,
            )
            record = {"id": sample["id"], "tool_calls": calls, "state_hash": state_hash}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if log_stream:
                log_stream.write(json.dumps({"id": sample["id"], "profile": profile}, ensure_ascii=False) + "\n")
                log_stream.flush()
            print(f"processed={sample['id']} seconds={time.perf_counter() - started:.3f}", file=sys.stderr)
    if log_stream:
        log_stream.close()
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
    asr_correct: bool = False,
    num_samples: int = 1,
    temperature: float | None = None,
    per_sample_log: str | Path | None = None,
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
    _write_predictions(input_path, output_path, paths, tool_env, catalog, asr, agent, config,
                       asr_correct, num_samples, temperature, per_sample_log)
