from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .graph_retrieval import BipartiteToolGraph
from .middleware import extract_inter_tool_bindings, verify_step_execution


SYSTEM_PROMPT = """Bạn là LifeMate, agent tiếng Việt dùng các công cụ được cung cấp để hoàn thành yêu cầu cuối của người dùng. Dùng thời gian và timezone trong context. History chỉ để hiểu ngữ cảnh; không thực hiện lại hành động cũ. Dựa vào kết quả tool thật trong steps_this_request để quyết định bước tiếp. Với tham số có x-ref hoặc cần ID thực thể chưa có sẵn, bắt buộc gọi tool tra cứu (kind="read") trước rồi mới gọi tool thao tác (kind="write"); tuyệt đối không tự đoán ID. Nếu có validation_feedback, hãy sửa đúng lỗi được chỉ ra. Nếu ngoài phạm vi, không cần hành động, hoặc đã hoàn tất yêu cầu, trả đúng {\"action\":\"stop\"}. Nếu cần thực hiện, trả đúng một JSON: {\"action\":\"call\",\"name\":\"app.tool\",\"arguments\":{...}}. Chọn tool trong available_tools và tuân thủ parameters. Nếu thiếu dữ liệu bắt buộc mà không thể tra cứu bằng tool, dùng sys.ask_missing_info. Chỉ xuất JSON, không giải thích."""


def _catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for app in catalog.get("apps", []) for tool in app.get("tools", [])}


def _compact_schema(schema: Any) -> Any:
    """Hammer/X-LAM style ultra-compact schema pruner:
    Removes title, $schema, additionalProperties, examples, and empty fields
    to conserve prompt context budget and reduce latency.
    """
    if isinstance(schema, list):
        return [_compact_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    ignored_keys = {"additionalProperties", "title", "$schema", "examples", "$id", "definitions"}
    for k, v in schema.items():
        if k in ignored_keys:
            continue
        if v in ("", [], {}, None) and k not in ("default", "enum"):
            continue
        cleaned[k] = _compact_schema(v)
    return cleaned


def available_tool_specs(
    sample: dict[str, Any],
    catalog: dict[str, Any],
    transcript: str = "",
    prior_calls: list[dict[str, Any]] | None = None,
    prior_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    graph = BipartiteToolGraph.from_sample_and_catalog(sample, catalog, _compact_schema)
    return graph.rank_tools_wrrf(
        transcript=transcript,
        history=sample.get("history", []),
        prior_calls=prior_calls,
        prior_results=prior_results,
        alpha_a=1.5,
        alpha_t=1.0,
        rrf_k=10,
    )


def normalized_history(
    history: list[dict[str, Any]], transcripts: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    transcripts = transcripts or {}
    result = []
    for turn in history:
        role = turn.get("role", "user")
        if role == "user":
            text = transcripts.get(turn.get("audio", ""), turn.get("audio_text", turn.get("text", "")))
            result.append({"role": role, "text": text})
        elif role == "assistant":
            result.append({
                "role": role,
                "text": turn.get("text", ""),
                "tool_calls": turn.get("tool_calls", []),
                "tool_results": turn.get("tool_results", []),
            })
    return result


def _build_step_entry(call: dict[str, Any], result: Any, catalog: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"call": call, "result": result}
    bindings = extract_inter_tool_bindings(result)
    if bindings:
        entry["resolved_refs"] = bindings
    verifier_note = verify_step_execution(call, result, catalog)
    if verifier_note:
        entry["verifier_note"] = verifier_note
    return entry


def build_messages(
    sample: dict[str, Any],
    transcript: str,
    catalog: dict[str, Any],
    transcripts: dict[str, str] | None = None,
    prior_calls: list[dict[str, Any]] | None = None,
    prior_results: list[dict[str, Any]] | None = None,
    feedback: str | None = None,
) -> list[dict[str, str]]:
    context = sample.get("context", {})
    payload = {
        "request": transcript,
        "context": {
            key: context.get(key)
            for key in ("current_time", "timezone", "location", "user_profile")
            if key in context
        },
        "available_tools": available_tool_specs(
            sample, catalog, transcript=transcript, prior_calls=prior_calls, prior_results=prior_results
        ),
        "history": normalized_history(sample.get("history", []), transcripts),
        "steps_this_request": [
            _build_step_entry(call, result, catalog)
            for call, result in zip(prior_calls or [], prior_results or [])
        ],
    }
    if feedback:
        payload["validation_feedback"] = feedback
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _pop_lowest_wrrf_tool(tools: list[dict[str, Any]]) -> bool:
    for idx in range(len(tools) - 1, -1, -1):
        if tools[idx].get("name") != "sys.ask_missing_info":
            if idx >= 4:
                tools.pop(idx)
                return True
            return False
    return False


def fit_messages(messages: list[dict[str, str]], tokenizer, max_input_tokens: int):
    payload = json.loads(messages[1]["content"])
    while True:
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if token_ids.shape[-1] <= max_input_tokens:
            return token_ids
        if payload["history"]:
            payload["history"].pop(0)
            continue
        if _pop_lowest_wrrf_tool(payload.get("available_tools", [])):
            continue
        raise ValueError(
            f"Prompt requires {token_ids.shape[-1]} tokens; budget is {max_input_tokens}"
        )


def action_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))
