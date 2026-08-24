"""Context Engine — Fase 2 of the research-first-runtime plan.

Replaces "context window as simple truncation" with a ranked working set:
every candidate item (goal, constraints, decisions, unresolved questions,
evidence, conversation) gets a priority/relevance/freshness score, and
compaction drops low-value conversational filler under budget pressure
while never silently dropping decisions, constraints, or cited evidence
(plan.md section 6, "Fase 2 — Context Engine": "Ao comprimir: preserve
decisões; preserve constraints; preserve unresolved questions; preserve
source references; remova ruído").

Consumer: pi_runtime.loop.Executor calls render_working_set_note() before
each AgentSession.prompt() call and, when there's anything worth
surfacing, queues it as steering context (AgentSession.
queue_steer_message) — a real, tested execution path, not a standalone
unused module (Regra 1.2). When a run has no decisions/constraints/
evidence/unresolved-questions yet (the common single-turn case), this
renders nothing and behavior is identical to Fase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pi_ai import AssistantMessage, TextContent, ToolResultMessage, UserMessage
from pi_runtime.state import AgentState

# Sources that must survive compaction regardless of budget — never
# silently dropped, per plan.md's explicit compaction rules.
_PROTECTED_SOURCES = {"goal", "constraint", "decision", "unresolved_question", "evidence"}


@dataclass
class ContextItem:
    content: str
    source: str  # goal | constraint | decision | unresolved_question | evidence | conversation
    priority: int = 0
    relevance: float = 1.0
    freshness: float = 1.0
    token_estimate: int = 0

    def score(self) -> float:
        return self.priority * 2 + self.relevance + self.freshness


def _message_text(message: Any) -> str:
    if isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, str):
            return content
        return " ".join(b.text for b in content if isinstance(b, TextContent))
    if isinstance(message, AssistantMessage):
        return " ".join(b.text for b in message.content if isinstance(b, TextContent))
    if isinstance(message, ToolResultMessage):
        return " ".join(b.text for b in message.content if isinstance(b, TextContent))
    return str(message)


@dataclass
class ContextEngine:
    """Assembles a token-budgeted working set from AgentState + the raw
    conversation. Only builds items for sources that actually have a
    producer today (goal/constraints from Goal, decisions/evidence/
    unresolved_questions from AgentState, conversation from messages) —
    plan.md's full source list also names skills/plan/tools/filesystem/
    subagents, which don't have a real producer yet (no Skills system
    until Fase 9, no filesystem-context tracking); adding items for
    sources with nothing to feed them would violate Regra 1.2 (no real
    consumer)."""

    chars_per_token: int = 4
    budget_tokens: int = 8000

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def collect_items(self, state: AgentState, messages: list[Any]) -> list[ContextItem]:
        items: list[ContextItem] = []

        if state.goal is not None:
            items.append(
                ContextItem(content=f"GOAL: {state.goal.objective}", source="goal", priority=100, freshness=1.0)
            )
            for constraint in state.goal.constraints:
                items.append(
                    ContextItem(content=f"CONSTRAINT: {constraint}", source="constraint", priority=90, freshness=1.0)
                )

        for decision in state.decisions:
            items.append(ContextItem(content=f"DECISION: {decision}", source="decision", priority=80, freshness=1.0))
        for question in state.unresolved_questions:
            items.append(
                ContextItem(content=f"UNRESOLVED: {question}", source="unresolved_question", priority=70, freshness=1.0)
            )
        for evidence in state.evidence:
            items.append(ContextItem(content=f"EVIDENCE: {evidence}", source="evidence", priority=60, freshness=1.0))

        total = len(messages)
        for index, message in enumerate(messages):
            text = _message_text(message)
            if not text.strip():
                continue
            freshness = (index + 1) / total if total else 1.0
            items.append(ContextItem(content=text, source="conversation", priority=10, freshness=freshness))

        for item in items:
            if not item.token_estimate:
                item.token_estimate = self.estimate_tokens(item.content)
        return items

    def rank(self, items: list[ContextItem]) -> list[ContextItem]:
        return sorted(items, key=lambda item: item.score(), reverse=True)

    def assemble_working_set(self, state: AgentState, messages: list[Any]) -> list[ContextItem]:
        """Protected items (goal/constraint/decision/unresolved_question/
        evidence) always make it in, regardless of budget — only
        conversational filler gets dropped once the budget is tight, and
        it's dropped lowest-ranked (oldest/least-relevant) first."""
        items = self.collect_items(state, messages)
        protected = [item for item in items if item.source in _PROTECTED_SOURCES]
        compactable = self.rank([item for item in items if item.source not in _PROTECTED_SOURCES])

        working_set = list(protected)
        used = sum(item.token_estimate for item in protected)
        for item in compactable:
            if used + item.token_estimate > self.budget_tokens:
                continue
            working_set.append(item)
            used += item.token_estimate
        return working_set

    def render_working_set_note(self, state: AgentState, messages: list[Any]) -> str | None:
        """A short, labeled block surfacing whatever protected context
        (decisions/constraints/evidence/unresolved questions — not the
        goal itself, that's already in the prompt) exists so far. Returns
        None when there's nothing beyond the bare goal, so a fresh run
        with no accumulated state renders nothing and behaves exactly
        like Fase 1 (the "reproduz comportamento atual para casos
        básicos" acceptance criterion)."""
        working_set = self.assemble_working_set(state, messages)
        surfaced = [item for item in working_set if item.source in _PROTECTED_SOURCES and item.source != "goal"]
        if not surfaced:
            return None
        lines = ["[context carried forward from this run — not a new instruction, background only]"]
        lines.extend(f"- {item.content}" for item in surfaced)
        return "\n".join(lines)


__all__ = ["ContextEngine", "ContextItem"]
