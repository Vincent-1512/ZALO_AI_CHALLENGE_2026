"""Optional transcript spelling repair using only the current sample's context."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _strings(nested)


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value).lower().replace("đ", "d")
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def context_vocabulary(sample: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    """Return 1–3 word phrases from user/app context and tool schemas."""
    sources = [sample.get("context", {}).get("user_profile", {})]
    apps = sample.get("context", {}).get("installed_apps", {})
    sources.extend([list(apps.keys()), apps])
    schema_values = []
    for app in catalog.get("apps", []):
        schema_values.extend((app.get("id", ""), app.get("name_vi", ""), app.get("description", "")))
        for tool in app.get("tools", []):
            schema_values.extend((tool.get("name", ""), tool.get("description", ""),
                                  tool.get("kind", ""), tool.get("returns", "")))
            for name, spec in tool.get("parameters", {}).get("properties", {}).items():
                schema_values.extend((name, spec.get("description", ""), spec.get("enum", [])))
    sources.append(schema_values)
    phrases: set[str] = set()
    for source in sources:
        for value in _strings(source):
            raw_words = _WORD.findall(value)
            if 1 <= len(raw_words) <= 3 and len(_fold(" ".join(raw_words)).replace(" ", "")) >= 3:
                phrases.add(value.strip())
            words = _WORD.findall(value.replace("_", " ").replace(".", " "))
            for start in range(len(words)):
                for size in range(1, min(3, len(words) - start) + 1):
                    phrase = " ".join(words[start : start + size])
                    if len(_fold(phrase).replace(" ", "")) >= 3:
                        phrases.add(phrase)
    return sorted(phrases, key=lambda phrase: (len(_WORD.findall(phrase)), phrase.casefold(), phrase))


def correct_transcript(
    transcript: str,
    sample: dict[str, Any],
    catalog: dict[str, Any],
    threshold: float = 0.88,
) -> tuple[str, list[dict[str, Any]]]:
    """Repair non-exact spans against context phrases; return text and audit entries."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    candidates: dict[int, list[tuple[str, str]]] = {}
    for phrase in context_vocabulary(sample, catalog):
        words = _WORD.findall(phrase)
        candidates.setdefault(len(words), []).append((phrase, " ".join(_fold(word) for word in words)))
    matches = list(_WORD.finditer(transcript))
    replacements: list[tuple[int, int, str, dict[str, Any]]] = []
    cursor = 0
    while cursor < len(matches):
        best = None
        for size in range(min(3, len(matches) - cursor), 0, -1):
            span = matches[cursor : cursor + size]
            observed = transcript[span[0].start() : span[-1].end()]
            folded = " ".join(_fold(match.group()) for match in span)
            for candidate, candidate_folded in candidates.get(size, []):
                if observed == candidate:
                    continue
                score = difflib.SequenceMatcher(None, folded, candidate_folded, autojunk=False).ratio()
                if score >= threshold and (best is None or (size, score) > (best[0], best[1])):
                    best = (size, score, span[0].start(), span[-1].end(), candidate, observed)
        if best is None:
            cursor += 1
            continue
        size, score, start, end, candidate, observed = best
        replacements.append((start, end, candidate, {"before": observed, "after": candidate, "score": round(score, 4)}))
        cursor += size
    corrected = transcript
    for start, end, replacement, _ in reversed(replacements):
        corrected = corrected[:start] + replacement + corrected[end:]
    changes = [entry for _, _, _, entry in replacements]
    return corrected, changes
