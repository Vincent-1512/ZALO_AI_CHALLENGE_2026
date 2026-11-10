"""Unit test for ToolNet (Bottom-up Graph Adaptation) + Reflexion (Top-down Verbal Feedback)."""
from __future__ import annotations

import json
from lifemate.graph_retrieval import BipartiteToolGraph
from lifemate.middleware import verify_step_execution


def test_toolnet_bottom_up_adaptation():
    catalog = {
        "apps": [
            {
                "name": "alarm",
                "description": "Quản lý báo thức và chuông báo",
                "tools": [
                    {"name": "alarm.create", "description": "Tạo báo thức mới", "kind": "write", "parameters": {}},
                    {"name": "alarm.list", "description": "Liệt kê danh sách báo thức", "kind": "read", "parameters": {}},
                    {"name": "alarm.delete", "description": "Xóa báo thức", "kind": "write", "parameters": {}},
                ],
            },
            {
                "name": "clock",
                "description": "Xem giờ thế giới và đồng hồ bấm giờ",
                "tools": [
                    {"name": "clock.world_clock", "description": "Xem giờ thế giới", "kind": "read", "parameters": {}},
                ],
            },
        ]
    }
    sample = {
        "context": {
            "installed_apps": {
                "alarm": {"tools": ["alarm.create", "alarm.list", "alarm.delete"]},
                "clock": {"tools": ["clock.world_clock"]},
            }
        },
        "history": [],
    }

    graph = BipartiteToolGraph.from_sample_and_catalog(sample, catalog, lambda x: x)

    # 1. Round 1: No prior failure -> alarm.list and alarm.create have normal ranking
    ranked_r1 = graph.rank_tools_wrrf(
        transcript="Báo thức 6h sáng mai",
        history=[],
        prior_calls=[],
        prior_results=[],
    )
    r1_names = [t["name"] for t in ranked_r1]
    assert "alarm.create" in r1_names[:3]

    # 2. Round 2: ToolNet Bottom-Up Graph Adaptation:
    # Suppose alarm.create failed with error
    ranked_r2 = graph.rank_tools_wrrf(
        transcript="Báo thức 6h sáng mai",
        history=[],
        prior_calls=[{"name": "alarm.create", "arguments": {}}],
        prior_results=[{"status": "failed", "error": "Missing required field 'time'"}],
    )
    r2_names = [t["name"] for t in ranked_r2]
    # alarm.create MUST be penalized and dropped down in priority
    assert r2_names.index("alarm.create") > r2_names.index("alarm.list")

    # 3. Round 3: Successful read -> boosts write tools
    ranked_r3 = graph.rank_tools_wrrf(
        transcript="Báo thức 6h sáng mai",
        history=[],
        prior_calls=[{"name": "alarm.list", "arguments": {}}],
        prior_results=[{"alarms": [{"id": "alm_01", "time": "06:00"}]}],
    )
    r3_names = [t["name"] for t in ranked_r3]
    assert "alarm" in r3_names[0]
    print("[TEST 1 PASSED] ToolNet Bottom-Up Graph Adaptation verified successfully!")


def test_top_down_reflexion():
    catalog = {
        "apps": [
            {
                "name": "calendar",
                "tools": [
                    {"name": "calendar.search", "kind": "read"},
                    {"name": "calendar.create_event", "kind": "write"},
                ],
            }
        ]
    }

    # Test error reflexion
    fail_call = {"name": "calendar.create_event", "arguments": {}}
    fail_res = {"status": "error", "error": "Invalid timestamp format"}
    feedback = verify_step_execution(fail_call, fail_res, catalog)
    assert feedback is not None
    assert "[Reflexion - Phản chiếu lỗi]" in feedback
    assert "ToolNet" in feedback

    # Test empty read reflexion
    empty_call = {"name": "calendar.search", "arguments": {"query": "Họp nhóm"}}
    empty_res = {"events": []}
    feedback_empty = verify_step_execution(empty_call, empty_res, catalog)
    assert feedback_empty is not None
    assert "[Reflexion - Phản chiếu tra cứu]" in feedback_empty
    print("[TEST 2 PASSED] Top-down Verbal Reflexion verified successfully!")


if __name__ == "__main__":
    test_toolnet_bottom_up_adaptation()
    test_top_down_reflexion()
    print("ALL TESTS PASSED 100%!")

