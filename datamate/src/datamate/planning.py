from __future__ import annotations

from typing import Any

from .executor import execute_plan
from .modeling import generate_text
from .query_plan import make_prompt, normalize_plan, parse_plan, validate_plan


class PlanningError(ValueError):
    def __init__(self, error: Exception, raw_response: str, attempts: int):
        super().__init__(f"{type(error).__name__}: {error}")
        self.raw_response = raw_response
        self.attempts = attempts


def plan_and_execute(model: Any, tokenizer: Any, question: dict[str, Any], package: Any,
                     max_new_tokens: int = 384, max_context_tokens: int = 8192,
                     repair_attempts: int = 0) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    """Generate, normalize, validate, and execute a plan; optionally repair schema errors.

    Repair prompts contain only the question, available business schema/rules, model output,
    and validation error. They never contain the gold answer.
    """
    messages = make_prompt(question, package)
    raw = ""
    last_error: Exception | None = None
    for attempt in range(repair_attempts + 1):
        if attempt:
            # Rebuild with a smaller catalog so the repair context stays below the model limit.
            messages = make_prompt(question, package, max_catalog_chars=3000)
            messages.append({
                "role": "user",
                "content": (
                    "Your previous JSON plan did not validate or execute. Return a corrected JSON plan only. "
                    "Keep the same question and use only the listed schema. Fix this error exactly:\n"
                    f"{type(last_error).__name__}: {last_error}\n\n"
                    f"Previous response (truncated):\n{raw[:1800]}"
                ),
            })
        try:
            raw = generate_text(model, tokenizer, messages, max_new_tokens, max_context_tokens)
            plan = normalize_plan(parse_plan(raw), package.catalog)
            validate_plan(plan, package.catalog)
            prediction = execute_plan(plan, package)
            return plan, prediction, raw, attempt + 1
        except Exception as error:
            last_error = error
            if attempt >= repair_attempts:
                raise PlanningError(error, raw, attempt + 1) from error
    raise RuntimeError("Plan generation ended without a result")
