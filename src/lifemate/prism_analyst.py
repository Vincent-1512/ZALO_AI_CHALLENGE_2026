from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


def compute_truncated_state_diff(
    predicted_state: dict[str, Any],
    gold_state: dict[str, Any],
    state_hash_fn,
    max_chars: int = 400,
) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for app, expected_app_state in gold_state.items():
        actual_app_state = predicted_state.get(app)
        if state_hash_fn({app: actual_app_state}) != state_hash_fn({app: expected_app_state}):
            serialized_actual = json.dumps(actual_app_state, ensure_ascii=False)
            serialized_gold = json.dumps(expected_app_state, ensure_ascii=False)
            diff[app] = {
                "actual": serialized_actual[:max_chars] + ("..." if len(serialized_actual) > max_chars else ""),
                "expected": serialized_gold[:max_chars] + ("..." if len(serialized_gold) > max_chars else ""),
            }
    return diff


def detect_stuck_type(
    calls: list[dict[str, Any]],
    profile: dict[str, Any],
    max_steps: int,
    timeout_seconds: float = 30.0,
) -> str:
    if profile.get("error"):
        return "crash"
    if profile.get("total_seconds", 0.0) > timeout_seconds:
        return "timeout"
    if len(calls) >= max_steps:
        return "max_steps_exceeded"
    seen: list[tuple[str, str]] = []
    for call in calls:
        sig = (call.get("name", ""), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if sig in seen:
            return "infinite_tool_loop"
        seen.append(sig)
    return "none"


def build_case_diagnostic(
    sample: dict[str, Any],
    calls: list[dict[str, Any]],
    predicted_state: dict[str, Any],
    state_match: bool,
    matched_calls: int,
    profile: dict[str, Any],
    max_steps: int,
    state_hash_fn,
) -> dict[str, Any]:
    gold_calls = sample["label"].get("tool_calls", [])
    gold_state = sample["label"].get("final_state", {})
    passed = bool(state_match and matched_calls == len(gold_calls) and len(calls) == len(gold_calls))
    stuck_type = detect_stuck_type(calls, profile, max_steps)
    state_diff = (
        {}
        if state_match
        else compute_truncated_state_diff(predicted_state, gold_state, state_hash_fn)
    )
    return {
        "case_id": sample["id"],
        "passed": passed,
        "state_match": state_match,
        "matched_calls": matched_calls,
        "predicted_calls": len(calls),
        "gold_calls": len(gold_calls),
        "stuck_type": stuck_type,
        "tool_call_sequence": calls,
        "gold_tool_call_sequence": gold_calls,
        "truncated_state_diff": state_diff,
    }


def _classify_failure_root_cause(case: dict[str, Any]) -> tuple[str, str, str]:
    pred_calls = case.get("tool_call_sequence", [])
    gold_calls = case.get("gold_tool_call_sequence", [])
    stuck = case.get("stuck_type", "none")

    if stuck in ("infinite_tool_loop", "max_steps_exceeded"):
        return (
            "Vòng lặp gọi lại tool hoặc vượt quá số bước tối đa do không dừng sau khi đã có kết quả",
            "middleware_only",
            "Chặn lệnh gọi lặp lại (duplicate call blocking) xác định 100% tại biên công cụ.",
        )

    if not gold_calls and pred_calls:
        return (
            "Gọi tool dư thừa cho yêu cầu ngoài phạm vi (Out-of-Scope) thay vì trả về action stop",
            "prompt_only",
            "Quy tắc nhận diện yêu cầu ngoài phạm vi thuộc chính sách suy luận toàn cục trong System Prompt.",
        )

    gold_names = [c.get("name") for c in gold_calls]
    pred_names = [c.get("name") for c in pred_calls]

    if "sys.ask_missing_info" in gold_names and "sys.ask_missing_info" not in pred_names:
        return (
            "Tự đoán tham số hoặc gọi thẳng tool thao tác thay vì hỏi thông tin còn thiếu qua sys.ask_missing_info",
            "prompt_middleware_both",
            "Cần kết hợp Prompt hướng dẫn hỏi thông tin thiếu và Middleware chặn tham số x-ref tự đoán chưa được tra cứu.",
        )

    if set(pred_names) == set(gold_names) and len(pred_calls) == len(gold_calls):
        return (
            "Gọi đúng tên công cụ nhưng sai định dạng/kiểu dữ liệu/giá trị đối số hoặc nhầm ID thực thể",
            "middleware_only",
            "Chuẩn hóa kiểu dữ liệu, ánh xạ tên tham số và đối chiếu ID thực thể (local invariants) tại Middleware.",
        )

    if len(pred_calls) < len(gold_calls) and set(pred_names).issubset(set(gold_names)):
        return (
            "Thiếu bước tra cứu (proactive lookup) trước khi thao tác hoặc dừng sớm trước khi hoàn tất chuỗi hành động",
            "prompt_middleware_both",
            "Cần kết hợp Prompt định hướng chuỗi bước read-before-write và Middleware kiểm tra điều kiện tiên quyết x-ref.",
        )

    return (
        "Chọn sai công cụ hoặc gọi dư công cụ không cần thiết so với yêu cầu cuối của người dùng",
        "prompt_only",
        "Chính sách chọn đúng công cụ theo ngữ cảnh và không gọi lại lịch sử thuộc về System Prompt.",
    )


def cluster_failures_for_prism(cases: list[dict[str, Any]], max_clusters: int = 4) -> list[dict[str, Any]]:
    """PRISM Analyst: Group failed cases by root cause and route to repair surfaces."""
    failed = [c for c in cases if not c.get("passed") and c.get("stuck_type") != "crash"]
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for case in failed:
        key = _classify_failure_root_cause(case)
        grouped[key].append(case["case_id"])

    sorted_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    clusters: list[dict[str, Any]] = []
    for (root_cause, surface, reason), case_ids in sorted_groups[:max_clusters]:
        clusters.append({
            "root_cause": root_cause,
            "repair_surface": surface,
            "case_ids": case_ids,
            "reason": reason,
        })
    return clusters


def compute_cross_generation_matrix(
    current_cases: list[dict[str, Any]],
    previous_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute PRISM Cross-Generation Failure Matrix (FIXED, NEW, RECURRING, PERSISTENT)."""
    if not previous_reports:
        return {}

    history_by_case: dict[str, list[bool]] = defaultdict(list)
    for rep in previous_reports:
        for case in rep.get("cases", []):
            history_by_case[case["case_id"]].append(bool(case.get("passed")))

    matrix: dict[str, list[str]] = {
        "FIXED": [],
        "NEW": [],
        "RECURRING": [],
        "PERSISTENT": [],
        "STABLE_PASS": [],
    }

    for case in current_cases:
        cid = case["case_id"]
        passed_now = bool(case.get("passed"))
        past = history_by_case.get(cid, [])
        if not past:
            continue
        if passed_now:
            if not past[-1] or not all(past):
                matrix["FIXED"].append(cid)
            else:
                matrix["STABLE_PASS"].append(cid)
        else:
            if all(past):
                matrix["NEW"].append(cid)
            elif not any(past) and len(past) >= 2:
                matrix["PERSISTENT"].append(cid)
            else:
                matrix["RECURRING"].append(cid)

    return {
        "counts": {k: len(v) for k, v in matrix.items()},
        "cases": matrix,
        "regression_safe": len(matrix["NEW"]) == 0,
    }



def _count_sentences(text: str) -> int:
    segments = [seg.strip() for seg in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if seg.strip()]
    return len(segments)


def validate_p1_prompt_mutation(
    parent_prompt: str,
    candidate_prompt: str,
    max_added_sentences: int = 2,
) -> dict[str, Any]:
    """Enforce PRISM Slot p_1 safety constraint: prevent prompt bloat (at most 1-2 sentences added per generation)."""
    parent_count = _count_sentences(parent_prompt)
    candidate_count = _count_sentences(candidate_prompt)
    added_sentences = candidate_count - parent_count
    char_delta = len(candidate_prompt.strip()) - len(parent_prompt.strip())
    ok = added_sentences <= max_added_sentences
    return {
        "ok": ok,
        "parent_sentences": parent_count,
        "candidate_sentences": candidate_count,
        "added_sentences": added_sentences,
        "max_allowed_added_sentences": max_added_sentences,
        "char_delta": char_delta,
        "reason": (
            "Hợp lệ: tuân thủ giới hạn chống phình ngữ cảnh (anti-prompt-bloat) của khe p_1."
            if ok
            else f"Vi phạm ràng buộc khe p_1: đã thêm {added_sentences} câu (tối đa cho phép {max_added_sentences} câu mỗi thế hệ)."
        ),
    }


def evaluate_crossover_opportunity(
    p1_prompt_report: dict[str, Any],
    p2_middleware_report: dict[str, Any],
) -> dict[str, Any]:
    """Check if conditional Crossover between Slot p_1 (Prompt) and Slot p_2 (Middleware) should trigger."""
    p1_passed = {c["case_id"] for c in p1_prompt_report.get("cases", []) if c.get("passed")}
    p2_passed = {c["case_id"] for c in p2_middleware_report.get("cases", []) if c.get("passed")}

    uniquely_fixed_by_p1 = sorted(p1_passed - p2_passed)
    uniquely_fixed_by_p2 = sorted(p2_passed - p1_passed)
    union_passed = sorted(p1_passed | p2_passed)

    should_crossover = bool(uniquely_fixed_by_p1 and uniquely_fixed_by_p2)
    return {
        "should_trigger_crossover": should_crossover,
        "p1_passed_count": len(p1_passed),
        "p2_passed_count": len(p2_passed),
        "potential_crossover_passed_count": len(union_passed),
        "uniquely_solved_by_p1_prompt": uniquely_fixed_by_p1,
        "uniquely_solved_by_p2_middleware": uniquely_fixed_by_p2,
    }
