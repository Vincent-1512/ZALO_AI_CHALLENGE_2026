"""
LifeMate Prompt & Dialogue Graph Signal Manager (`promt/prompts.py`).

Đã được tinh chỉnh cho bài toán LifeMate (Zalo AI Challenge 2026):
- Xóa bỏ các prompt Neo4j Cypher / Meeting QA không phù hợp.
- Giữ lại và chuẩn hóa 3 prompt cốt lõi hỗ trợ Multi-Turn Voice History & Tool Calling:
  1. `lifemate_system.txt`: System prompt tích hợp Temporal Intent Shift (SUPERSEDES),
     Pronoun/Ellipsis Resolution (DUR), và Null-Safety.
  2. `dur_restoration.txt`: Khôi phục câu nói tỉnh lược (Incomplete Utterance Restoration)
     và sửa lỗi chính tả ASR tiếng Việt (PhoWhisper) cho data augmentation / SFT.
  3. `entity_resolution.txt`: Liên kết thực thể tiếng Việt sang `x-ref` ID và phát hiện
     trường hợp trùng tên (Ambiguity Guard -> `sys.ask_missing_info`).
- Cung cấp hàm deterministic `extract_dialogue_graph_signals()` trích xuất tín hiệu
  Dialogue-Tool Graph (Anaphora, SUPERSEDES, Entity Mentions) với độ trễ ~0ms.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

PROMT_DIR = Path(__file__).resolve().parent

# Từ khóa tiếng Việt báo hiệu đại từ chỉ định / tỉnh lược (cần tra ngược history)
_VI_ANAPHORA_PATTERNS = re.compile(
    r"\b(vừa nãy|hồi nãy|lúc nãy|ở trên|cái đó|đơn đó|lịch đó|báo thức đó|"
    r"nhắc nhở đó|ghi chú đó|người đó|anh ấy|chị ấy|bạn ấy|họ|tiếp theo|"
    r"như cũ|như vậy|của tôi vừa tạo|vừa đặt)\b",
    re.IGNORECASE,
)

# Từ khóa tiếng Việt báo hiệu thay đổi/cập nhật/hủy ý định cũ (SUPERSEDES)
_VI_SUPERSEDES_PATTERNS = re.compile(
    r"\b(đổi lại|dời sang|dời lại|chuyển sang|sửa lại|cập nhật lại|"
    r"thay đổi|hủy giúp|hủy đơn|xóa lịch|xóa báo thức|thôi không|"
    r"khoan đã|à nhầm|sửa thành)\b",
    re.IGNORECASE,
)


class PromptManager:
    """Quản lý các prompt template trong thư mục `promt/` cho dự án LifeMate."""

    @staticmethod
    def _load_template(filename: str) -> str:
        filepath = PROMT_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy file prompt: {filepath}")
        return filepath.read_text(encoding="utf-8")

    @classmethod
    def get_lifemate_system_prompt(cls) -> str:
        """Lấy System Prompt chuẩn của LifeMate (tích hợp DUR + SUPERSEDES + Null-Safety)."""
        return cls._load_template("lifemate_system.txt")

    @classmethod
    def get_dur_restoration_prompt(
        cls,
        history_json: str,
        raw_transcript: str,
        installed_apps: str,
    ) -> str:
        """
        Lấy prompt khôi phục câu nói tỉnh lược (DUR) & sửa lỗi ASR tiếng Việt.
        Dùng cho offline SFT data enrichment hoặc phân tích lỗi multi-turn.
        """
        template = cls._load_template("dur_restoration.txt")
        return (
            template.replace("{history_json}", history_json)
            .replace("{raw_transcript}", raw_transcript)
            .replace("{installed_apps}", installed_apps)
        )

    @classmethod
    def get_entity_resolution_prompt(
        cls,
        transcript: str,
        resolved_refs_json: str,
        target_tool_schema: str,
    ) -> str:
        """
        Lấy prompt phân giải thực thể (Entity Resolution) sang `x-ref` ID
        và kiểm tra Ambiguity Guard (`sys.ask_missing_info`).
        """
        template = cls._load_template("entity_resolution.txt")
        return (
            template.replace("{transcript}", transcript)
            .replace("{resolved_refs_json}", resolved_refs_json)
            .replace("{target_tool_schema}", target_tool_schema)
        )


def extract_dialogue_graph_signals(
    transcript: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Trích xuất nhanh các cạnh quan hệ trên Đồ thị Hội thoại (Dialogue-Tool Graph)
    bằng luật từ vựng tiếng Việt (0ms overhead, không tốn thêm lượt gọi LLM):
    - `has_anaphora`: Câu nói hiện tại có chứa đại từ chỉ định/tỉnh lược (`COREFERS_TO`) không?
    - `has_supersedes`: Câu nói hiện tại có yêu cầu sửa/dời/hủy hành động trước (`SUPERSEDES`) không?
    - `prior_tool_apps`: Tập các app đã được gọi trong `history` (cạnh `FOLLOWS_TOOL`).
    - `history_entity_ids`: Các ID thực thể đã xuất hiện trong `history` để nối cạnh `BINDS_ENTITY`.
    """
    text = (transcript or "").strip()
    anaphora_matches = [m.group(0).lower() for m in _VI_ANAPHORA_PATTERNS.finditer(text)]
    supersedes_matches = [m.group(0).lower() for m in _VI_SUPERSEDES_PATTERNS.finditer(text)]

    prior_tool_apps: list[str] = []
    prior_tool_names: list[str] = []
    history_entity_ids: dict[str, list[str]] = {}

    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        for tc in turn.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tname = str(tc.get("name") or "")
            if tname:
                prior_tool_names.append(tname)
                if "." in tname:
                    app = tname.split(".", 1)[0]
                    if app not in prior_tool_apps:
                        prior_tool_apps.append(app)
            args = tc.get("arguments")
            if isinstance(args, dict):
                for k, v in args.items():
                    if (k == "id" or k.endswith("_id")) and isinstance(v, (str, int)) and str(v).strip():
                        bucket = history_entity_ids.setdefault(k, [])
                        sval = str(v).strip()
                        if sval not in bucket:
                            bucket.append(sval)

    return {
        "has_anaphora": bool(anaphora_matches),
        "anaphora_cues": anaphora_matches,
        "has_supersedes": bool(supersedes_matches),
        "supersedes_cues": supersedes_matches,
        "prior_tool_apps": prior_tool_apps,
        "prior_tool_names": prior_tool_names,
        "history_entity_ids": history_entity_ids,
    }
