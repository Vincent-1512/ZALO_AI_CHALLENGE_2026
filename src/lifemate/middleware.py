from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _normalize_str(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _normalize_vietnamese_datetime_str(text: str) -> str:
    stripped = text.strip()
    if re.fullmatch(r"\d:\d{2}(?::\d{2})?", stripped):
        return "0" + stripped
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", stripped):
        return stripped.replace("/", "-")
    m_hm = re.fullmatch(r"(\d{1,2})\s*(?:h|giờ|:)\s*(\d{1,2})(?:\s*phút)?", stripped, flags=re.IGNORECASE)
    if m_hm:
        h, m = int(m_hm.group(1)), int(m_hm.group(2))
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    m_half = re.fullmatch(r"(\d{1,2})\s*(?:h|giờ)\s*rưỡi", stripped, flags=re.IGNORECASE)
    if m_half:
        h = int(m_half.group(1))
        if 0 <= h <= 23:
            return f"{h:02d}:30"
    return text


def _schema_types(prop_schema: dict[str, Any]) -> list[str]:
    raw_type = prop_schema.get("type")
    if isinstance(raw_type, str):
        return [raw_type]
    if isinstance(raw_type, list):
        return [t for t in raw_type if isinstance(t, str)]
    forbranch = prop_schema.get("anyOf") or prop_schema.get("oneOf") or []
    types: list[str] = []
    for sub in forbranch:
        if isinstance(sub, dict):
            types.extend(_schema_types(sub))
    return types


def _try_unwrap_json(value: Any, expected_types: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if ("object" in expected_types and stripped.startswith("{") and stripped.endswith("}")) or (
        "array" in expected_types and stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _snap_enum(value: Any, enum_values: list[Any]) -> Any:
    if value in enum_values:
        return value
    if isinstance(value, str):
        norm_val = _normalize_str(value).lower()
        matches = [
            candidate
            for candidate in enum_values
            if isinstance(candidate, str) and _normalize_str(candidate).lower() == norm_val
        ]
        if len(matches) == 1:
            return matches[0]
    return value


def _coerce_scalar(value: Any, target_type: str) -> Any:
    if target_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                return int(text)
            if re.fullmatch(r"[+-]?\d+\.0+", text):
                return int(float(text))
    elif target_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", text):
                try:
                    num = float(text)
                    return int(num) if num.is_integer() and "." not in text else num
                except ValueError:
                    return value
    elif target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes"):
                return True
            if lowered in ("false", "0", "no"):
                return False
    elif target_type == "string":
        if isinstance(value, str):
            norm_str = unicodedata.normalize("NFC", value)
            return _normalize_vietnamese_datetime_str(norm_str)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return value


def _remap_argument_keys(
    arguments: dict[str, Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    if not properties or not arguments:
        return dict(arguments)

    remapped: dict[str, Any] = {}
    unknown_items: list[tuple[str, Any]] = []

    for key, val in arguments.items():
        if key in properties:
            remapped[key] = val
        else:
            unknown_items.append((key, val))

    if not unknown_items:
        return remapped

    norm_to_prop: dict[str, str] = {}
    for prop_name in properties:
        if prop_name not in remapped:
            norm_to_prop[_normalize_key(prop_name)] = prop_name

    for raw_key, val in unknown_items:
        norm_key = _normalize_key(raw_key)
        target_prop = norm_to_prop.get(norm_key)

        if target_prop is None and norm_key == "id":
            id_candidates = [
                p for p in properties if p not in remapped and p.endswith("_id")
            ]
            if len(id_candidates) == 1:
                target_prop = id_candidates[0]
        elif target_prop is None and norm_key.endswith("_id") and "id" in properties and "id" not in remapped:
            target_prop = "id"

        if target_prop is not None and target_prop not in remapped:
            remapped[target_prop] = val
            norm_to_prop.pop(_normalize_key(target_prop), None)
        else:
            remapped[raw_key] = val

    return remapped


def coerce_value_by_schema(value: Any, prop_schema: dict[str, Any]) -> Any:
    if not isinstance(prop_schema, dict):
        return value

    expected_types = _schema_types(prop_schema)
    value = _try_unwrap_json(value, expected_types)

    for target_type in expected_types:
        coerced = _coerce_scalar(value, target_type)
        if coerced is not value or type(coerced) is not type(value):
            value = coerced
            break

    if "array" in expected_types:
        items_schema = prop_schema.get("items", {})
        if not isinstance(value, list) and value is not None:
            item_types = _schema_types(items_schema) if isinstance(items_schema, dict) else []
            if not item_types or any(
                isinstance(value, t)
                for t in (
                    (str,) if "string" in item_types else ()
                ) + (
                    (int, float) if ("integer" in item_types or "number" in item_types) else ()
                ) + (
                    (dict,) if "object" in item_types else ()
                )
            ):
                value = [value]
        if isinstance(value, list) and isinstance(items_schema, dict):
            value = [coerce_value_by_schema(item, items_schema) for item in value]

    if "object" in expected_types and isinstance(value, dict):
        value = coerce_arguments_by_schema(value, prop_schema)

    if "enum" in prop_schema and isinstance(prop_schema["enum"], list):
        value = _snap_enum(value, prop_schema["enum"])

    return value


def coerce_arguments_by_schema(
    arguments: dict[str, Any],
    parameters_schema: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict) or not isinstance(parameters_schema, dict):
        return arguments

    properties = parameters_schema.get("properties")
    if not isinstance(properties, dict):
        return arguments

    required = set(parameters_schema.get("required") or [])
    remapped = _remap_argument_keys(arguments, properties)

    cleaned: dict[str, Any] = {}
    for key, val in remapped.items():
        prop_schema = properties.get(key)
        if val is None and key not in required:
            if not isinstance(prop_schema, dict) or "null" not in _schema_types(prop_schema):
                continue
        if isinstance(prop_schema, dict):
            val = coerce_value_by_schema(val, prop_schema)
        elif isinstance(val, str):
            val = unicodedata.normalize("NFC", val)
        cleaned[key] = val

    return cleaned


def _allowed_tool_names(sample: dict[str, Any]) -> set[str]:
    names = {
        name
        for app in sample.get("context", {}).get("installed_apps", {}).values()
        for name in app.get("tools", [])
    }
    names.add("sys.ask_missing_info")
    return names


def _catalog_tool_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tool["name"]: tool
        for app in catalog.get("apps", [])
        for tool in app.get("tools", [])
    }


def silent_correct_action(
    action: dict[str, Any] | None,
    sample: dict[str, Any],
    catalog: dict[str, Any],
    prior_results: list[Any] | None = None,
) -> dict[str, Any] | None:
    """PRISM Silent Correction + HEART Inter-Tool Router at the tool boundary."""
    if not isinstance(action, dict):
        return None
    if action.get("action") == "stop":
        return {"action": "stop"}
    if action.get("action") != "call":
        return None

    raw_name = action.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    name = raw_name.strip()

    allowed = _allowed_tool_names(sample)
    if name not in allowed and "." not in name:
        suffix = f".{name}"
        candidates = [cand for cand in allowed if cand.endswith(suffix)]
        if len(candidates) == 1:
            name = candidates[0]

    raw_args = action.get("arguments", {})
    if isinstance(raw_args, str):
        raw_args = _try_unwrap_json(raw_args, ["object"])
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return None

    tool_map = _catalog_tool_map(catalog)
    tool_spec = tool_map.get(name)
    if tool_spec and isinstance(tool_spec.get("parameters"), dict):
        corrected_args = coerce_arguments_by_schema(raw_args, tool_spec["parameters"])
        corrected_args = _resolve_xref_labels_to_ids(
            corrected_args, tool_spec["parameters"], sample, prior_results or []
        )
    else:
        corrected_args = dict(raw_args)

    return {"action": "call", "name": name, "arguments": corrected_args}


def _collect_scalars(obj: Any, out: set[str]) -> None:
    if isinstance(obj, str):
        norm = _normalize_str(obj)
        if norm:
            out.add(norm)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(str(obj))
    elif isinstance(obj, list):
        for item in obj:
            _collect_scalars(item, out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                out.add(_normalize_str(k))
            _collect_scalars(v, out)


def _extract_observed_scalars(
    sample: dict[str, Any],
    transcript: str,
    transcripts: dict[str, str] | None,
    prior_results: list[Any],
) -> set[str]:
    observed: set[str] = set()
    _collect_scalars(sample.get("context", {}), observed)
    _collect_scalars(transcript, observed)
    if transcripts:
        _collect_scalars(list(transcripts.values()), observed)
    for turn in sample.get("history", []):
        if isinstance(turn, dict):
            _collect_scalars(turn.get("text", ""), observed)
            _collect_scalars(turn.get("audio_text", ""), observed)
            _collect_scalars(turn.get("tool_calls", []), observed)
            _collect_scalars(turn.get("tool_results", []), observed)
    _collect_scalars(prior_results, observed)
    return observed


def _value_in_observed(val: Any, observed: set[str], full_text_blob: str) -> bool:
    if isinstance(val, list):
        return all(_value_in_observed(item, observed, full_text_blob) for item in val)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        s = str(val)
        return s in observed or s in full_text_blob
    if isinstance(val, str):
        norm = _normalize_str(val)
        if not norm:
            return True
        return norm in observed or norm.lower() in full_text_blob
    return True


def _iter_entity_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_entity_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_entity_dicts(item)


def _record_id_for_key(record: dict[str, Any], key: str) -> str | None:
    for candidate_key in (key, "id", key.split("_")[-1]):
        val = record.get(candidate_key)
        if isinstance(val, (str, int)) and not isinstance(val, bool):
            return _normalize_str(str(val))
    return None


def _check_cross_entity_consistency(
    arguments: dict[str, Any],
    xref_keys: list[str],
    sample: dict[str, Any],
    prior_results: list[Any],
) -> str | None:
    if len(xref_keys) < 2:
        return None

    sources: list[Any] = list(prior_results)
    for turn in sample.get("history", []):
        if isinstance(turn, dict):
            sources.extend(turn.get("tool_results", []))

    records = [d for src in sources for d in _iter_entity_dicts(src)]
    if not records:
        return None

    for i, parent_key in enumerate(xref_keys):
        parent_val = arguments.get(parent_key)
        if not isinstance(parent_val, (str, int)) or isinstance(parent_val, bool):
            continue
        parent_val_str = _normalize_str(str(parent_val))

        matching_parent_records = [
            rec for rec in records if _record_id_for_key(rec, parent_key) == parent_val_str
        ]
        if not matching_parent_records:
            continue

        parent_scalars: set[str] = set()
        for rec in matching_parent_records:
            _collect_scalars(rec, parent_scalars)

        for j, child_key in enumerate(xref_keys):
            if i == j:
                continue
            child_val = arguments.get(child_key)
            if not isinstance(child_val, (str, int)) or isinstance(child_val, bool):
                continue
            child_val_str = _normalize_str(str(child_val))
            if child_val_str in parent_scalars:
                continue

            # Check if child_val belongs to a different sibling record of the same parent type
            for rec in records:
                other_parent_id = _record_id_for_key(rec, parent_key)
                if not other_parent_id or other_parent_id == parent_val_str:
                    continue
                # Ensure rec looks like the same entity type (shares keys with matching_parent_records[0])
                if len(set(rec.keys()) & set(matching_parent_records[0].keys())) < 2:
                    continue
                rec_scalars: set[str] = set()
                _collect_scalars(rec, rec_scalars)
                if child_val_str in rec_scalars:
                    return (
                        f"Lỗi bất nhất trạng thái dữ liệu: Giá trị '{child_key}={child_val!r}' "
                        f"không thuộc về '{parent_key}={parent_val!r}' mà thuộc về "
                        f"'{parent_key}={other_parent_id!r}' trong dữ liệu đã tra cứu. "
                        "Hãy kiểm tra lại và truyền đúng cặp ID."
                    )
    return None


def check_local_invariants(
    action: dict[str, Any],
    sample: dict[str, Any],
    catalog: dict[str, Any],
    transcript: str,
    transcripts: dict[str, str] | None,
    prior_calls: list[dict[str, Any]],
    prior_results: list[Any],
) -> str | None:
    """PRISM Prerequisite & State Tracking middleware before tool execution.

    Returns an explicit Vietnamese error string if a local invariant is violated,
    or None if the call is safe to validate and execute.
    """
    name = action.get("name", "")
    arguments = action.get("arguments", {})

    allowed = _allowed_tool_names(sample)
    if name not in allowed:
        app_prefix = name.split(".", 1)[0] if "." in name else ""
        same_app = sorted(t for t in allowed if t.startswith(f"{app_prefix}.")) if app_prefix else []
        hint = f" Các tool hợp lệ của '{app_prefix}': {same_app}." if same_app else ""
        return (
            f"Tool '{name}' không nằm trong danh sách available_tools của mẫu này.{hint} "
            "Chỉ chọn tool trong available_tools hoặc trả {\"action\":\"stop\"} nếu yêu cầu ngoài phạm vi."
        )

    for prev in prior_calls:
        if prev.get("name") == name and prev.get("arguments", {}) == arguments:
            return (
                f"Lệnh gọi '{name}' với cùng arguments đã được thực thi ở bước trước trong lượt này. "
                "Không gọi lặp lại; hãy dùng kết quả đã có trong steps_this_request để thực hiện bước tiếp theo "
                "hoặc trả {\"action\":\"stop\"} nếu đã hoàn thành yêu cầu."
            )

    tool_map = _catalog_tool_map(catalog)
    tool_spec = tool_map.get(name)
    if not tool_spec:
        return None

    parameters = tool_spec.get("properties") if "properties" in tool_spec else tool_spec.get("parameters", {})
    if not isinstance(parameters, dict):
        return None

    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        return None

    # Check x-ref prerequisite invariant: prevent hallucinated entity IDs
    xref_keys = [
        key
        for key, prop in properties.items()
        if isinstance(prop, dict) and prop.get("x-ref") and key in arguments and arguments[key] is not None
    ]
    if xref_keys:
        observed = _extract_observed_scalars(sample, transcript, transcripts, prior_results)
        full_text_blob = "\n".join(s.lower() for s in observed)
        for key in xref_keys:
            val = arguments[key]
            if not _value_in_observed(val, observed, full_text_blob):
                app_prefix = name.split(".", 1)[0] if "." in name else ""
                read_tools = sorted(
                    t_name
                    for t_name in allowed
                    if t_name.startswith(f"{app_prefix}.")
                    and tool_map.get(t_name, {}).get("kind") == "read"
                )
                read_hint = (
                    f" Hãy gọi công cụ tra cứu (read) trước như {read_tools} để lấy ID thực tế"
                    if read_tools
                    else " Hãy gọi công cụ tra cứu (read) trước để lấy ID thực tế"
                )
                return (
                    f"Lỗi điều kiện tiên quyết (x-ref): Tham số '{key}={val!r}' của '{name}' "
                    f"chưa xuất hiện trong kết quả tra cứu hay lịch sử hội thoại (không được tự đoán ID)."
                    f"{read_hint}, hoặc gọi 'sys.ask_missing_info' nếu người dùng chưa cung cấp đủ thông tin."
                )
        ambig_error = _check_ambiguous_xref_labels(arguments, xref_keys, sample, prior_results)
        if ambig_error is not None:
            return ambig_error
        cross_error = _check_cross_entity_consistency(arguments, xref_keys, sample, prior_results)
        if cross_error is not None:
            return cross_error

    return None


_LABEL_FIELDS = ("name", "title", "label", "summary", "contact_name", "recipient", "subject", "item_name")


def _check_ambiguous_xref_labels(
    arguments: dict[str, Any],
    xref_keys: list[str],
    sample: dict[str, Any],
    prior_results: list[Any],
) -> str | None:
    """Ambiguity Guard (từ promt/entity_resolution.txt):
    Nếu tham số x-ref truyền vào là tên tự nhiên trùng với >= 2 thực thể khác ID
    trong kết quả tra cứu, chặn thao tác và yêu cầu gọi sys.ask_missing_info.
    """
    sources: list[Any] = list(prior_results)
    for turn in sample.get("history", []):
        if isinstance(turn, dict):
            sources.extend(turn.get("tool_results", []))
    records = [d for src in sources for d in _iter_entity_dicts(src)]
    if not records:
        return None

    all_ids = {
        _record_id_for_key(rec, key)
        for rec in records
        for key in xref_keys
    } - {None}

    for key in xref_keys:
        val = arguments.get(key)
        if not isinstance(val, str):
            continue
        val_str = _normalize_str(val)
        if val_str in all_ids:
            continue
        matched_ids: list[str] = []
        for rec in records:
            cand_id = _record_id_for_key(rec, key)
            if not cand_id:
                continue
            for label_field in _LABEL_FIELDS:
                label_val = rec.get(label_field)
                if isinstance(label_val, str) and _normalize_str(label_val).lower() == val_str.lower():
                    if cand_id not in matched_ids:
                        matched_ids.append(cand_id)
        if len(matched_ids) > 1:
            return (
                f"Lỗi nhập nhằng thực thể (Ambiguous Entity): Tên '{val_str}' ở tham số '{key}' "
                f"khớp với {len(matched_ids)} thực thể khác nhau ({matched_ids}). "
                f"Không được tự ý chọn 1 ID; hãy gọi 'sys.ask_missing_info' để hỏi người dùng xác nhận."
            )
    return None


def _resolve_xref_labels_to_ids(
    arguments: dict[str, Any],
    parameters_schema: dict[str, Any],
    sample: dict[str, Any],
    prior_results: list[Any],
) -> dict[str, Any]:
    """HEART Router: If LLM passes an entity's natural language name/title/label into an x-ref ID field,
    automatically resolve it to the exact entity ID from prior tool results or history.
    """
    properties = parameters_schema.get("properties")
    if not isinstance(properties, dict):
        return arguments

    xref_keys = [
        key
        for key, prop in properties.items()
        if isinstance(prop, dict) and prop.get("x-ref") and isinstance(arguments.get(key), str)
    ]
    if not xref_keys:
        return arguments

    sources: list[Any] = list(prior_results)
    for turn in sample.get("history", []):
        if isinstance(turn, dict):
            sources.extend(turn.get("tool_results", []))
    records = [d for src in sources for d in _iter_entity_dicts(src)]
    if not records:
        return arguments

    all_ids = {
        _record_id_for_key(rec, key)
        for rec in records
        for key in xref_keys
    } - {None}

    resolved = dict(arguments)
    for key in xref_keys:
        val_str = _normalize_str(resolved[key])
        if val_str in all_ids:
            continue
        matched_ids: list[str] = []
        for rec in records:
            cand_id = _record_id_for_key(rec, key)
            if not cand_id:
                continue
            for label_field in _LABEL_FIELDS:
                label_val = rec.get(label_field)
                if isinstance(label_val, str) and _normalize_str(label_val).lower() == val_str.lower():
                    if cand_id not in matched_ids:
                        matched_ids.append(cand_id)
        if len(matched_ids) == 1:
            resolved[key] = matched_ids[0]

    return resolved


def extract_inter_tool_bindings(result: Any) -> dict[str, list[Any]]:
    """HEART Tool Primitive: Extract uniform ID/reference bindings from heterogeneous tool outputs."""
    bindings: dict[str, list[Any]] = {}
    for rec in _iter_entity_dicts(result):
        for k, v in rec.items():
            if not isinstance(k, str):
                continue
            if (k == "id" or k.endswith("_id")) and isinstance(v, (str, int)) and not isinstance(v, bool):
                bucket = bindings.setdefault(k, [])
                if v not in bucket:
                    bucket.append(v)
    return bindings


def evaluate_4_criteria_verifier(
    call: dict[str, Any],
    result: Any,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """HEART Verifier (Appendix D.4): Score step outcome across 4 independent criteria:
    1. task_completion
    2. argument_consistency
    3. execution_validity
    4. constraint_satisfaction
    """
    name = call.get("name", "")
    tool_map = _catalog_tool_map(catalog)
    tool_spec = tool_map.get(name, {})
    kind = tool_spec.get("kind", "")

    execution_validity = True
    constraint_satisfaction = True
    argument_consistency = True
    task_completion = True
    feedback_phi: str | None = None

    if isinstance(result, dict):
        status = str(result.get("status", "")).lower()
        if status in ("error", "failed", "fail") or result.get("ok") is False or "error" in result:
            execution_validity = False
            task_completion = False
            err_msg = result.get("error") or result.get("message") or "Lỗi thực thi"
            feedback_phi = (
                f"Verifier [execution_validity=FAIL]: Lệnh '{name}' thất bại ({err_msg}). "
                "Hãy kiểm tra lại đối số, gọi tool read khác hoặc dùng sys.ask_missing_info; không gọi lặp lại y hệt."
            )
        elif kind == "read":
            list_fields = [v for v in result.values() if isinstance(v, list)]
            if list_fields and all(len(lst) == 0 for lst in list_fields):
                task_completion = False
                feedback_phi = (
                    f"Verifier [task_completion=INCOMPLETE]: Công cụ tra cứu '{name}' trả về danh sách rỗng. "
                    "Tuyệt đối không tự bịa ID cho bước tiếp theo; hãy thử mở rộng điều kiện tìm kiếm hoặc gọi sys.ask_missing_info."
                )
    elif isinstance(result, list) and len(result) == 0 and kind == "read":
        task_completion = False
        feedback_phi = (
            f"Verifier [task_completion=INCOMPLETE]: Công cụ tra cứu '{name}' trả về danh sách rỗng. "
            "Tuyệt đối không tự bịa ID cho bước tiếp theo; hãy thử mở rộng điều kiện tìm kiếm hoặc gọi sys.ask_missing_info."
        )

    passed = all((task_completion, argument_consistency, execution_validity, constraint_satisfaction))
    return {
        "passed": passed,
        "criteria": {
            "task_completion": task_completion,
            "argument_consistency": argument_consistency,
            "execution_validity": execution_validity,
            "constraint_satisfaction": constraint_satisfaction,
        },
        "feedback_phi": feedback_phi,
    }


def verify_step_execution(
    call: dict[str, Any],
    result: Any,
    catalog: dict[str, Any],
) -> str | None:
    """HEART Post-Execution Verifier returning structured feedback phi_k if any of the 4 criteria fails."""
    verdict = evaluate_4_criteria_verifier(call, result, catalog)
    return verdict["feedback_phi"]
