from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

REWRITER_PROMPT_TEMPLATE = """Bạn là chuyên gia hiệu đính tiếng Việt thông minh cho trợ lý ảo LifeMate.
Nhiệm vụ của bạn là xem xét các giả thuyết nhận dạng giọng nói từ mô hình ASR (kèm độ tự tin), đối chiếu với lịch sử hội thoại trước đó của người dùng và thời gian hiện tại để khôi phục lại câu lệnh gốc chuẩn xác nhất.

Ngữ cảnh:
- Thời gian hiện tại: {current_time}
- Lịch sử hội thoại trước đó:
{history_str}

Danh sách các giả thuyết ASR nhận diện được:
{hypotheses_str}

Hãy suy nghĩ trong thẻ <think>...</think>:
1. Phân tích các từ bị mâu thuẫn giữa các giả thuyết (đặc biệt là số tiền, tên người, từ đồng âm, từ địa phương).
2. Dựa vào lịch sử để tìm manh mối (ví dụ tên người đã từng liên lạc, tác vụ đang dở).
3. Chuẩn hóa lại ngày giờ nếu cần thiết.

Sau thẻ </think>, xuất đúng một dòng duy nhất theo định dạng:
[CORRECTED]: <câu_lệnh_tiếng_Việt_chuẩn_xác>"""


class ReasoningRewriter:
    """Intelligent Vietnamese speech rewriter and contextual reasoner powered by DeepSeek-R1-Distill-Qwen-1.5B

    (with automatic fallback to heuristics when weights are uninitialized or running in lightweight test mode).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.dtype_str = dtype
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self._is_ready = False

        if self.model_path and (self.model_path.exists() or not str(self.model_path).startswith("/")):
            self._init_model()

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _init_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.bfloat16 if self.dtype_str == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
            if self.device == "cpu":
                torch_dtype = torch.float32

            logger.info("Loading ReasoningRewriter from %s on %s...", self.model_path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch_dtype,
                device_map=self.device if self.device != "cpu" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
            self._is_ready = True
            logger.info("ReasoningRewriter loaded successfully!")
        except Exception as exc:
            logger.warning("Could not initialize 1.5B LLM weights (%s). Running in heuristic fallback mode.", exc)
            self.tokenizer = None
            self.model = None
            self._is_ready = False

    def _format_history(self, history: Optional[List[Dict[str, Any]]]) -> str:
        if not history:
            return "(Không có lịch sử trước đó)"
        lines = []
        for idx, turn in enumerate(history[-4:], 1):
            role = turn.get("role", "user")
            text = turn.get("text", "") or turn.get("audio_text", "")
            calls = turn.get("tool_calls", [])
            call_str = f" -> Gọi tool: {[c.get('name') for c in calls]}" if calls else ""
            lines.append(f"- Lượt {idx} ({role}): {text}{call_str}")
        return "\n".join(lines)

    def _heuristic_rewrite(self, raw_transcript: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Lightweight heuristic cleanup when 1.5B LLM is unavailable or for instant low-latency processing."""
        if not raw_transcript:
            return ""
        text = unicodedata.normalize("NFC", raw_transcript).strip()
        # Remove stutter / duplicate consecutive words (e.g., "bật bật", "gọi gọi")
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        # Fix common ASR phonetic misrecognitions in Vietnamese
        replacements = [
            (r"\bbật đen\b", "bật đèn"),
            (r"\btắt đen\b", "tắt đèn"),
            (r"\bnhắn tinh\b", "nhắn tin"),
            (r"\bgọi điện thoại thoại\b", "gọi điện thoại"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text.strip()

    def rewrite(
        self,
        raw_transcript: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        n_best: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Rewrite raw transcript using 1.5B reasoning with context & dialogue history."""
        if not raw_transcript and not n_best:
            return ""

        context = context or {}
        history = history or context.get("history", [])
        current_time = context.get("current_time", context.get("time", "2026-03-01 08:00:00"))

        if not self._is_ready or self.model is None or self.tokenizer is None:
            return self._heuristic_rewrite(raw_transcript, context)

        try:
            import torch

            if n_best and len(n_best) > 0:
                hyp_lines = [
                    f"{i+1}. \"{h.get('text', '')}\" (Độ tự tin: {h.get('confidence', 90.0)}%)"
                    for i, h in enumerate(n_best[:4])
                ]
                hypotheses_str = "\n".join(hyp_lines)
            else:
                hypotheses_str = f"1. \"{raw_transcript}\" (Độ tự tin: 95.0%)"

            history_str = self._format_history(history)
            prompt = REWRITER_PROMPT_TEMPLATE.format(
                current_time=current_time,
                history_str=history_str,
                hypotheses_str=hypotheses_str,
            )

            messages = [{"role": "user", "content": prompt}]
            text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            generated_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Extract output after [CORRECTED]:
            if "[CORRECTED]:" in response:
                result = response.split("[CORRECTED]:")[-1].strip().split("\n")[0].strip()
                result = result.strip('"').strip("'")
                return unicodedata.normalize("NFC", result)

            if "</think>" in response:
                result = response.split("</think>")[-1].strip().split("\n")[0].strip()
                result = result.strip('"').strip("'")
                return unicodedata.normalize("NFC", result)

            return self._heuristic_rewrite(raw_transcript, context)

        except Exception as error:
            logger.warning("ReasoningRewriter inference error: %s. Using heuristic fallback.", error)
            return self._heuristic_rewrite(raw_transcript, context)

