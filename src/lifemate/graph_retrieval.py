from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def _extract_tokens(text: str) -> list[str]:
    norm = _normalize_text(text)
    words = [tok for tok in re.findall(r"[^\W_]+", norm, flags=re.UNICODE) if len(tok) > 1]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


@dataclass
class AppNode:
    """Node in the Agent/App set A of the Bipartite Knowledge Graph G = (A, T, E)."""

    app_id: str
    description: str
    tokens: Counter[str] = field(default_factory=Counter)
    tool_ids: list[str] = field(default_factory=list)


@dataclass
class ToolNode:
    """Node in the Tool set T of the Bipartite Knowledge Graph G = (A, T, E)."""

    tool_id: str
    owner_app: str
    kind: str
    spec: dict[str, Any]
    tokens: Counter[str] = field(default_factory=Counter)


class BipartiteToolGraph:
    """Agent-as-a-Graph Bipartite Knowledge Graph G = (A, T, E) with wRRF & Tool->App Traversal."""

    def __init__(
        self,
        app_nodes: dict[str, AppNode],
        tool_nodes: dict[str, ToolNode],
        sys_tools: list[dict[str, Any]],
    ):
        self.app_nodes = app_nodes
        self.tool_nodes = tool_nodes
        self.sys_tools = sys_tools

    @classmethod
    def from_sample_and_catalog(
        cls,
        sample: dict[str, Any],
        catalog: dict[str, Any],
        compact_schema_fn,
    ) -> "BipartiteToolGraph":
        app_catalog_map: dict[str, dict[str, Any]] = {}
        tool_catalog_map: dict[str, dict[str, Any]] = {}
        for app_entry in catalog.get("apps", []):
            app_name = app_entry.get("name", "")
            if app_name:
                app_catalog_map[app_name] = app_entry
            for tool_entry in app_entry.get("tools", []):
                t_name = tool_entry.get("name", "")
                if t_name:
                    tool_catalog_map[t_name] = tool_entry

        installed_names = {
            name
            for app in sample.get("context", {}).get("installed_apps", {}).values()
            for name in app.get("tools", [])
        }
        installed_names.add("sys.ask_missing_info")

        app_nodes: dict[str, AppNode] = {}
        tool_nodes: dict[str, ToolNode] = {}
        sys_tools: list[dict[str, Any]] = []

        for name in sorted(installed_names):
            raw_tool = tool_catalog_map.get(name)
            if raw_tool is None:
                spec = {"name": name}
            else:
                spec = {
                    "name": name,
                    "description": raw_tool.get("description", ""),
                    "kind": raw_tool.get("kind", ""),
                    "parameters": compact_schema_fn(raw_tool.get("parameters", {})),
                }

            if name == "sys.ask_missing_info" or name.startswith("sys."):
                sys_tools.append(spec)
                continue

            owner_app = name.split(".", 1)[0] if "." in name else name
            if owner_app not in app_nodes:
                raw_app = app_catalog_map.get(owner_app, {})
                app_desc = raw_app.get("description", "")
                app_text = f"{owner_app.replace('_', ' ')} {app_desc}"
                app_nodes[owner_app] = AppNode(
                    app_id=owner_app,
                    description=app_desc,
                    tokens=Counter(_extract_tokens(app_text)),
                )

            param_parts: list[str] = []
            props = spec.get("parameters", {}).get("properties", {})
            if isinstance(props, dict):
                for p_name, p_schema in props.items():
                    param_parts.append(p_name.replace("_", " "))
                    if isinstance(p_schema, dict):
                        param_parts.append(str(p_schema.get("description", "")))

            tool_text = (
                f"{name.replace('.', ' ').replace('_', ' ')} "
                f"{spec.get('description', '')} "
                f"{' '.join(param_parts)}"
            )
            t_tokens = Counter(_extract_tokens(tool_text))
            tool_nodes[name] = ToolNode(
                tool_id=name,
                owner_app=owner_app,
                kind=spec.get("kind", ""),
                spec=spec,
                tokens=t_tokens,
            )
            # Enrich parent App node representation via bipartite edge (T -> A)
            app_nodes[owner_app].tool_ids.append(name)
            app_nodes[owner_app].tokens.update(t_tokens)

        return cls(app_nodes=app_nodes, tool_nodes=tool_nodes, sys_tools=sys_tools)

    def _cosine_lexical_similarity(self, query_counter: Counter[str], doc_counter: Counter[str]) -> float:
        if not query_counter or not doc_counter:
            return 0.0
        common = set(query_counter.keys()) & set(doc_counter.keys())
        if not common:
            return 0.0
        dot = sum(query_counter[k] * doc_counter[k] for k in common)
        norm_q = math.sqrt(sum(v * v for v in query_counter.values()))
        norm_d = math.sqrt(sum(v * v for v in doc_counter.values()))
        if norm_q == 0.0 or norm_d == 0.0:
            return 0.0
        return dot / (norm_q * norm_d)

    def rank_tools_wrrf(
        self,
        transcript: str,
        history: list[dict[str, Any]],
        prior_calls: list[dict[str, Any]] | None = None,
        alpha_a: float = 1.5,
        alpha_t: float = 1.0,
        rrf_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rank tools using Type-Specific Weighted Reciprocal Rank Fusion (wRRF)
        and Tool -> App Graph Traversal:
            s_new(e) = alpha_T / (k + r_T(e))  for e in C_T
            s_new(e) = alpha_A / (k + r_A(e))  for e in C_A
        """
        if not self.tool_nodes:
            return list(self.sys_tools)

        # Build query representation from current request + recent user/assistant history turns
        weighted_query: Counter[str] = Counter()
        for tok in _extract_tokens(transcript):
            weighted_query[tok] += 3
        for turn in history[-4:]:
            if isinstance(turn, dict):
                turn_text = str(turn.get("text", "") or turn.get("audio_text", ""))
                for tok in _extract_tokens(turn_text):
                    weighted_query[tok] += 1

        active_apps = {
            call.get("name", "").split(".", 1)[0]
            for call in (prior_calls or [])
            if isinstance(call, dict) and "." in call.get("name", "")
        }
        history_apps = {
            call.get("name", "").split(".", 1)[0]
            for turn in history[-2:]
            if isinstance(turn, dict)
            for call in turn.get("tool_calls", [])
            if isinstance(call, dict) and "." in call.get("name", "")
        }

        # 1. Score and rank Agent/App Corpus C_A
        app_raw_scores: dict[str, float] = {}
        for app_id, app_node in self.app_nodes.items():
            sim = self._cosine_lexical_similarity(weighted_query, app_node.tokens)
            if app_id in active_apps:
                sim += 0.50
            elif app_id in history_apps:
                sim += 0.20
            app_raw_scores[app_id] = sim

        sorted_apps = sorted(
            self.app_nodes.keys(),
            key=lambda a: (-app_raw_scores[a], a),
        )
        rank_a: dict[str, int] = {app_id: idx + 1 for idx, app_id in enumerate(sorted_apps)}

        # 2. Score and rank Tool Corpus C_T
        tool_raw_scores: dict[str, float] = {}
        for tool_id, tool_node in self.tool_nodes.items():
            sim = self._cosine_lexical_similarity(weighted_query, tool_node.tokens)
            if tool_node.owner_app in active_apps:
                sim += 0.25
            tool_raw_scores[tool_id] = sim

        sorted_tools = sorted(
            self.tool_nodes.keys(),
            key=lambda t: (-tool_raw_scores[t], t),
        )
        rank_t: dict[str, int] = {tool_id: idx + 1 for idx, tool_id in enumerate(sorted_tools)}

        # 3. Compute wRRF node scores: s_wRRF(e)
        app_wrrf: dict[str, float] = {
            app_id: alpha_a / (rrf_k + r) for app_id, r in rank_a.items()
        }
        tool_wrrf: dict[str, float] = {
            tool_id: alpha_t / (rrf_k + r) for tool_id, r in rank_t.items()
        }

        # 4. Graph Traversal (Tool -> App along edges E):
        # Aggregate top tool signals into their owning App A* = owner(T)
        app_traversed_score: dict[str, float] = {}
        for app_id, app_node in self.app_nodes.items():
            member_scores = sorted(
                (tool_wrrf[t_id] for t_id in app_node.tool_ids),
                reverse=True,
            )
            # Combine App wRRF score + top-2 owned Tool wRRF scores via traversal
            traversed = app_wrrf[app_id] + sum(member_scores[:2])
            app_traversed_score[app_id] = traversed

        # 5. Order Apps by traversed graph score, and within each App keep Read tools before Write tools
        ordered_specs: list[dict[str, Any]] = []
        final_apps = sorted(
            self.app_nodes.keys(),
            key=lambda a: (-app_traversed_score[a], a),
        )
        for app_id in final_apps:
            app_tool_ids = self.app_nodes[app_id].tool_ids
            sorted_app_tools = sorted(
                app_tool_ids,
                key=lambda t_id: (
                    0 if self.tool_nodes[t_id].kind == "read" else 1,
                    -tool_wrrf[t_id],
                    t_id,
                ),
            )
            for t_id in sorted_app_tools:
                ordered_specs.append(self.tool_nodes[t_id].spec)

        ordered_specs.extend(self.sys_tools)
        return ordered_specs
