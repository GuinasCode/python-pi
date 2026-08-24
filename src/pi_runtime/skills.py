"""Skills System — Fase 9 of the research-first-runtime plan.

Builds on Fase 8's SkillRegistry (versioning/approval/rollback) and the
existing pi_coding_agent.resource_loader.Skill/load_skills (SKILL.md
discovery — already a working form of progressive disclosure at the
metadata level: system_prompt._format_skills_for_prompt only ever puts
name+description in the prompt, never the full SKILL.md body; the model
reads the rest via the `read` tool on skill.path only if it decides to).
Neither is replaced (Regra 1.1) — the actual gap this closes: every
discovered skill's name+description was *always* injected into the
prompt regardless of relevance to the current query ("skills irrelevantes
não entram no contexto" was not enforced anywhere), with no record of
which skills were considered, selected, or how they performed.

Real consumer: pi_runtime.context.ContextEngine (Fase 2) — "skills" was
named there as one of the sources a ContextItem should be able to
represent, explicitly deferred because no selection mechanism existed
yet. SkillSelector closes that gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillSelection:
    """One candidate's outcome from a selection round — every candidate
    gets one of these, not just the winners, so "selection é observável"
    means the full ranking is visible, not a silent top-N slice."""

    name: str
    description: str
    score: float
    selected: bool


@dataclass
class SkillUsageRecord:
    name: str
    query: str
    timestamp: float = field(default_factory=time.time)
    succeeded: bool | None = None  # filled in later via SkillUsageTracker.record_outcome()


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "it",
        "do",
        "does",
        "how",
        "and",
        "or",
        "with",
    }
)


def _keywords(text: str) -> set[str]:
    words = text.lower().replace("-", " ").replace("_", " ").split()
    return {w for w in words if w not in _STOPWORDS}


class SkillSelector:
    """Deterministic keyword-overlap relevance scoring — the same
    lightweight approach already used for Soul overlap detection
    (pi_memory.store.find_overlapping_soul's difflib-based check), not a
    new embedding-based subsystem (Regra: no second search system).
    Common stopwords are stripped before comparing — without that, short
    filler words shared by any two skill descriptions ("a", "to", "how")
    swamp the real signal."""

    def __init__(self, *, top_k: int = 3, min_score: float = 0.0) -> None:
        self._top_k = top_k
        self._min_score = min_score

    def score(self, query: str, skill: Any) -> float:
        query_words = _keywords(query)
        if not query_words:
            return 0.0
        haystack = f"{getattr(skill, 'name', '')} {getattr(skill, 'description', '')}"
        haystack_words = _keywords(haystack)
        if not haystack_words:
            return 0.0
        overlap = query_words & haystack_words
        return len(overlap) / len(query_words)

    def select(self, query: str, skills: list[Any]) -> list[SkillSelection]:
        """Fase 9 acceptance criteria 2+3: "skills irrelevantes não
        entram no contexto" (only the top `top_k` candidates above
        `min_score` are marked selected) and "selection é observável"
        (every candidate's score is returned, not just the winners)."""
        scored = [(skill, self.score(query, skill)) for skill in skills]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        selections: list[SkillSelection] = []
        for index, (skill, score) in enumerate(scored):
            selected = index < self._top_k and score > self._min_score
            selections.append(
                SkillSelection(
                    name=getattr(skill, "name", str(skill)),
                    description=getattr(skill, "description", ""),
                    score=score,
                    selected=selected,
                )
            )
        return selections


class SkillUsageTracker:
    """Fase 9 acceptance criterion 4: "usage e success rate são
    registrados." A plain in-memory log for this vertical slice —
    persisting it (e.g. as procedural/episodic memory via Fase 7) is a
    natural next step, not required by this phase's own acceptance
    criteria, so it isn't built speculatively here (Regra 1.2)."""

    def __init__(self) -> None:
        self._records: list[SkillUsageRecord] = []

    def record_selection(self, query: str, selections: list[SkillSelection]) -> None:
        for selection in selections:
            if selection.selected:
                self._records.append(SkillUsageRecord(name=selection.name, query=query))

    def record_outcome(self, name: str, *, succeeded: bool) -> bool:
        """Attaches an outcome to the most recent still-open usage record
        for this skill. Returns False (rather than raising) if there's no
        open record to attach to — an outcome reported for a skill that
        was never selected is a caller bug, but not one this should crash
        over."""
        for record in reversed(self._records):
            if record.name == name and record.succeeded is None:
                record.succeeded = succeeded
                return True
        return False

    def success_rate(self, name: str) -> float | None:
        outcomes = [r.succeeded for r in self._records if r.name == name and r.succeeded is not None]
        if not outcomes:
            return None
        return sum(1 for outcome in outcomes if outcome) / len(outcomes)

    def usage_count(self, name: str) -> int:
        return sum(1 for r in self._records if r.name == name)

    def all_records(self) -> list[SkillUsageRecord]:
        return list(self._records)


__all__ = ["SkillSelection", "SkillSelector", "SkillUsageRecord", "SkillUsageTracker"]
