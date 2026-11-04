from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """Bạn là LifeMate, agent tiếng Việt dùng các công cụ được cung cấp để hoàn thành yêu cầu cuối của người dùng. Dùng thời gian và timezone trong context. History chỉ để hiểu ngữ cảnh; không thực hiện lại hành động cũ. Dựa vào kết quả tool thật để quyết định bước tiếp. Không đoán ID hoặc dữ liệu không có căn cứ. Nếu ngoài phạm vi hoặc không cần hành động, trả đúng {\"action\":\"stop\"}. Nếu cần thực hiện, trả đúng một JSON: {\"action\":\"call\",\"name\":\"app.tool\",\"arguments\":{...}}. Chọn tool trong available_tools và tuân thủ parameters. Nếu thiếu dữ liệu bắt buộc mà không thể tra cứu bằng tool, dùng sys.ask_missing_info. Chỉ xuất JSON, không giải thích."""


def _catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for app in catalog.get("apps", []) for tool in app.get("tools", [])}


def available_tool_specs(sample: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    index = _catalog_index(catalog)
    names = {
        name
        for app in sample.get("context", {}).get("installed_apps", {}).values()
        for name in app.get("tools", [])
    }
    names.add("sys.ask_missing_info")
    specs = []
    for name in sorted(names):
        tool = index.get(name)
        if tool is None:
            specs.append({"name": name})
            continue
        specs.append({
            "name": name,
            "description": tool.get("description", ""),
            "kind": tool.get("kind", ""),
            "parameters": tool.get("parameters", {}),
        })
    return specs


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
        "available_tools": available_tool_specs(sample, catalog),
        "history": normalized_history(sample.get("history", []), transcripts),
        "steps_this_request": [
            {"call": call, "result": result}
            for call, result in zip(prior_calls or [], prior_results or [])
        ],
    }
    if feedback:
        payload["validation_feedback"] = feedback
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def fit_messages(messages: list[dict[str, str]], tokenizer, max_input_tokens: int):
    payload = json.loads(messages[1]["content"])
    while True:
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if token_ids.shape[-1] <= max_input_tokens:
            return token_ids
        if not payload["history"]:
            raise ValueError(
                f"Prompt requires {token_ids.shape[-1]} tokens; budget is {max_input_tokens}"
            )
        payload["history"].pop(0)


def action_json(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))
