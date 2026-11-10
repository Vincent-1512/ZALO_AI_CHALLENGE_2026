import pytest
from lifemate.reasoning_rewriter import ReasoningRewriter


def test_reasoning_rewriter_heuristic():
    rewriter = ReasoningRewriter(model_path=None)
    
    # Test duplicate removal and homophone correction
    raw = "bật bật đen phòng khách"
    cleaned = rewriter.rewrite(raw)
    assert "bật đèn phòng khách" in cleaned
    assert "bật bật" not in cleaned

    # Test "nhắn tinh" -> "nhắn tin"
    raw2 = "nhắn tinh cho mẹ bảo con về muộn"
    cleaned2 = rewriter.rewrite(raw2)
    assert "nhắn tin cho mẹ" in cleaned2

    # Test empty string
    assert rewriter.rewrite("") == ""


def test_reasoning_rewriter_with_history():
    rewriter = ReasoningRewriter(model_path=None)
    history = [
        {"role": "user", "text": "Kiểm tra danh bạ của mẹ"},
        {"role": "assistant", "tool_calls": [{"name": "contacts_search", "arguments": {"query": "mẹ"}}]},
    ]
    context = {"current_time": "2026-03-01 08:30:00"}
    raw = "gọi điện thoại thoại cho bà ấy"
    cleaned = rewriter.rewrite(raw, context=context, history=history)
    assert "gọi điện thoại cho bà ấy" in cleaned

