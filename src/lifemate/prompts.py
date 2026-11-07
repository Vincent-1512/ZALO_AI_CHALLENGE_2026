from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from typing import Any


SYSTEM_PROMPT = """Bạn là LifeMate, agent tiếng Việt dùng các công cụ được cung cấp để hoàn thành yêu cầu cuối của người dùng. Dùng thời gian và timezone trong context. History chỉ để hiểu ngữ cảnh; không thực hiện lại hành động cũ. Dựa vào kết quả tool thật để quyết định bước tiếp. Không đoán ID hoặc dữ liệu không có căn cứ. Nếu ngoài phạm vi hoặc không cần hành động, trả đúng {\"action\":\"stop\"}. Nếu cần thực hiện, trả đúng một JSON: {\"action\":\"call\",\"name\":\"app.tool\",\"arguments\":{...}}. Chọn tool trong available_tools và tuân thủ parameters. Nếu thiếu dữ liệu bắt buộc mà không thể tra cứu bằng tool, dùng sys.ask_missing_info. Chỉ xuất JSON, không giải thích."""


def _catalog_index(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for app in catalog.get("apps", []) for tool in app.get("tools", [])}


_QUERY_STOPWORDS = {
    "cho", "giup", "giúp", "minh", "mình", "toi", "tôi", "ban", "bạn",
    "hay", "hãy", "voi", "với", "duoc", "được", "nhe", "nhé", "la", "là",
    "cua", "của", "mot", "một", "cac", "các", "nhung", "những", "nay", "này",
    "do", "đó", "theo", "va", "và", "thi", "thì", "co", "có", "khong", "không",
}


def _search_terms(value: Any) -> set[str]:
    text = unicodedata.normalize("NFC", str(value)).lower().replace("_", " ").replace(".", " ")
    return {
        term for term in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if len(term) > 1 and term not in _QUERY_STOPWORDS
    }


def _tool_search_text(tool: dict[str, Any], app_names: dict[str, str]) -> str:
    parameters = tool.get("parameters", {})
    pieces = [
        tool.get("name", ""), tool.get("description", ""), tool.get("kind", ""),
        tool.get("returns", ""), app_names.get(tool.get("app", ""), ""),
    ]
    for name, spec in parameters.get("properties", {}).items():
        pieces.extend((name, spec.get("description", "")))
        pieces.extend(spec.get("enum", []))
    return " ".join(map(str, pieces))


def available_tool_specs(
    sample: dict[str, Any], catalog: dict[str, Any], request_text: str = ""
) -> list[dict[str, Any]]:
    index = _catalog_index(catalog)
    names = {
        name
        for app in sample.get("context", {}).get("installed_apps", {}).values()
        for name in app.get("tools", [])
    }
    names.add("sys.ask_missing_info")
    app_names = {
        app.get("id", ""): f"{app.get('name_vi', '')} {app.get('description', '')}"
        for app in catalog.get("apps", [])
    }
    query_terms = _search_terms(request_text)
    candidate_terms = {
        name: _search_terms(_tool_search_text(index[name], app_names))
        for name in names if name in index
    }
    document_frequency = Counter(
        term for terms in candidate_terms.values() for term in terms
    )
    ranked = []
    for name in sorted(names):
        tool = index.get(name)
        if tool is None:
            ranked.append((0.0, name, {"name": name}))
            continue
        terms = candidate_terms[name]
        # Prefer query words that distinguish a tool from the other installed tools.
        score = sum(1.0 / (1 + document_frequency[term]) for term in query_terms & terms)
        spec = {
            "name": name,
            "description": tool.get("description", ""),
            "kind": tool.get("kind", ""),
            "parameters": tool.get("parameters", {}),
        }
        ranked.append((score, name, spec))
    # Keep every installed tool to protect recall; optionally compare relevance order
    # against the original alphabetical order using the same model and dev split.
    if os.environ.get("LIFEMATE_TOOL_RETRIEVAL", "off").lower() in {"0", "off", "false"}:
        ranked.sort(key=lambda row: row[1])
    else:
        ranked.sort(key=lambda row: (-row[0], row[1]))
    specs = [row[2] for row in ranked]
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
        "available_tools": available_tool_specs(sample, catalog, transcript),
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
