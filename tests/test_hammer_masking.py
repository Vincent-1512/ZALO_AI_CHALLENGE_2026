"""Unit test for Hammer ICLR 2025 Function & Parameter Masking Data Augmentation."""
from __future__ import annotations

import random
from training.prepare_sft import create_masked_variant


def test_hammer_masking_augmentation():
    catalog = {
        "apps": [
            {
                "id": "alarm",
                "name": "alarm",
                "tools": [
                    {
                        "name": "alarm.create",
                        "description": "Tạo báo thức mới",
                        "kind": "write",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string", "description": "Thời gian báo thức"},
                                "label": {"type": "string", "description": "Nhãn báo thức"},
                            },
                            "required": ["time"],
                        },
                    }
                ],
            }
        ]
    }

    row = {
        "id": "sample_001",
        "audio_text": "Đặt báo thức 6h sáng mai",
        "context": {
            "installed_apps": {
                "alarm": {"tools": ["alarm.create"]}
            }
        },
        "label": {
            "tool_calls": [
                {"name": "alarm.create", "arguments": {"time": "2026-11-21T06:00:00+07:00", "label": "Dậy đi học"}}
            ],
            "tool_results": [{"status": "success"}],
        },
    }

    rng = random.Random(42)
    masked_row, masked_catalog = create_masked_variant(row, catalog, rng)

    # 1. Tool name must be randomized, NOT 'alarm.create'
    masked_tool = masked_catalog["apps"][0]["tools"][0]
    new_fn_name = masked_tool["name"]
    assert new_fn_name != "alarm.create"
    assert new_fn_name.startswith("alarm.act_")

    # 2. Descriptions MUST be preserved 100%
    assert masked_tool["description"] == "Tạo báo thức mới"
    assert "properties" in masked_tool["parameters"]

    # 3. Parameter names must be randomized
    masked_props = masked_tool["parameters"]["properties"]
    assert "time" not in masked_props
    assert len(masked_props) == 2
    for p_name, p_schema in masked_props.items():
        assert p_name.startswith("p_")
        assert "description" in p_schema

    # 4. Row labels must be synchronized 1:1 with masked catalog
    masked_call = masked_row["label"]["tool_calls"][0]
    assert masked_call["name"] == new_fn_name
    assert "time" not in masked_call["arguments"]
    assert "2026-11-21T06:00:00+07:00" in masked_call["arguments"].values()
    assert "Dậy đi học" in masked_call["arguments"].values()

    # 5. Installed apps must be updated
    assert new_fn_name in masked_row["context"]["installed_apps"]["alarm"]["tools"]

    print("[HAMMER TEST PASSED] Function and parameter masking generated successfully!")


if __name__ == "__main__":
    test_hammer_masking_augmentation()
