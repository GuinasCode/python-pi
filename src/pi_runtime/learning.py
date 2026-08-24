"""Learning Loop — Fase 8 of the research-first-runtime plan.

trajectory -> outcome -> failures -> successful patterns -> memory
candidate -> skill candidate -> evaluation -> approved update
(plan.md section 12).

Reuses pi_runtime.state.AgentState as the trajectory record (Fase 1) and
pi_runtime.memory's cognitive-type vocabulary for memory candidates
(Fase 7) rather than inventing new shapes for either (Regra 1.1). Skill
storage here is a small, explicit, versioned in-memory registry
(SkillRegistry) sufficient for this phase's own acceptance criteria — a
candidate can be generated, approved, rejected, and rolled back; the
full Skills System (loader, selector, progressive disclosure, quality
gates, on-disk persistence) is Fase 9's job, building on this contract
rather than replacing it.

Regra: "nunca alterar skill automaticamente sem diff/avaliação/rollback/
provenance" is enforced structurally, not by convention: SkillCandidate
always carries a diff against what it's replacing and provenance
(source_run_id); SkillRegistry.apply() only ever appends a new version,
never overwrites in place; rollback() appends a copy of an older version
as the new current one rather than deleting history.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from pi_runtime.state import AgentState, RunStatus, VerificationResult


@dataclass
class TrajectoryAnalysis:
    run_id: str
    succeeded: bool
    failures: list[str]
    repaired_steps: int
    total_attempts: int
    decisions: list[str]


class TrajectoryAnalyzer:
    """Deterministic — structural analysis of the AgentState a run
    already produced (Fase 1), no LLM call."""

    def analyze(self, state: AgentState) -> TrajectoryAnalysis:
        failures: list[str] = []
        if state.error_message:
            failures.append(state.error_message)
        if state.verification is not None and not state.verification.passed:
            failures.extend(state.verification.failures)

        repaired_steps = 0
        total_attempts = 0
        if state.plan is not None:
            for step in state.plan.steps:
                total_attempts += step.attempts
                if step.attempts > 1:
                    repaired_steps += 1

        return TrajectoryAnalysis(
            run_id=state.run_id,
            succeeded=state.status == RunStatus.DONE,
            failures=failures,
            repaired_steps=repaired_steps,
            total_attempts=total_attempts,
            decisions=list(state.decisions),
        )


@dataclass
class MemoryCandidate:
    """Uses pi_runtime.memory's cognitive-type vocabulary (Fase 7) — kept
    as a plain str here (not importing CognitiveMemoryType) purely to
    avoid a hard import cycle between learning.py and memory.py; callers
    pass this straight through to CognitiveMemoryType(candidate.
    cognitive_type) when actually writing it."""

    title: str
    content: str
    cognitive_type: str
    confidence: float
    source_run_id: str


def generate_memory_candidates(analysis: TrajectoryAnalysis) -> list[MemoryCandidate]:
    """Fase 8 acceptance criterion 1: "uma execução pode gerar candidato
    de memória." Only a *successful* run's decisions become candidates —
    a failed run's decisions aren't confidently reusable knowledge. This
    only proposes candidates; it never writes them — a caller decides
    whether to pass one through pi_runtime.memory.write_with_policy,
    which still applies its own confidence/dedupe/secret checks."""
    if not analysis.succeeded:
        return []
    return [
        MemoryCandidate(
            title=decision[:80],
            content=decision,
            cognitive_type="episodic",
            confidence=0.6,
            source_run_id=analysis.run_id,
        )
        for decision in analysis.decisions
    ]


@dataclass
class SkillCandidate:
    name: str
    content: str
    diff: str
    version: int
    source_run_id: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending | approved | rejected


def generate_skill_candidate(analysis: TrajectoryAnalysis, *, name: str, repair_text: str) -> SkillCandidate | None:
    """Fase 8 acceptance criterion 2: "pode gerar candidato de skill."
    Only proposed when a run actually needed a repair — a run that
    succeeded on the first attempt learned nothing new worth proposing
    as a procedure to reuse."""
    if analysis.repaired_steps == 0:
        return None
    return SkillCandidate(
        name=name, content=repair_text, diff=f"+ {repair_text}", version=1, source_run_id=analysis.run_id
    )


@dataclass
class SkillReviewResult:
    candidate: SkillCandidate
    approved: bool
    reason: str = ""


class SkillRegistry:
    """Minimal versioned store — apply() only ever appends, never
    overwrites in place. Fase 9 builds the real loader/selector on top
    of this contract rather than replacing it."""

    def __init__(self) -> None:
        self._versions: dict[str, list[SkillCandidate]] = {}

    def review(self, candidate: SkillCandidate, *, approve: bool, reason: str = "") -> SkillReviewResult:
        candidate.status = "approved" if approve else "rejected"
        return SkillReviewResult(candidate=candidate, approved=approve, reason=reason)

    def apply(self, result: SkillReviewResult) -> bool:
        """Fase 8 acceptance criterion 3: "atualização pode ser
        rejeitada" — a rejected result is never applied; this returns
        False without touching the registry at all."""
        if not result.approved:
            return False
        history = self._versions.setdefault(result.candidate.name, [])
        result.candidate.version = len(history) + 1
        history.append(result.candidate)
        return True

    def current(self, name: str) -> SkillCandidate | None:
        history = self._versions.get(name)
        return history[-1] if history else None

    def history(self, name: str) -> list[SkillCandidate]:
        return list(self._versions.get(name, []))

    def rollback(self, name: str, *, to_version: int) -> bool:
        """Fase 8 acceptance criterion 4: "atualização pode ser
        revertida." Never deletes history — appends a copy of the target
        version as the new current one, so the full audit trail (every
        version ever applied, in order) stays intact rather than being
        rewritten."""
        history = self._versions.get(name)
        if not history:
            return False
        target = next((c for c in history if c.version == to_version), None)
        if target is None:
            return False
        restored = dataclasses.replace(target, version=len(history) + 1, created_at=time.time())
        history.append(restored)
        return True


def check_regression(*, previous_score: float | None, new_score: float) -> VerificationResult:
    """Fase 8 acceptance criterion 5: "regressão de skill gera falha de
    eval." Reuses pi_runtime.state.VerificationResult (Fase 1) rather
    than a second verification concept — same pattern as Fase 3's
    verify_tool_result and Fase 4's ResearchVerifier. No previous score
    (first version of a skill) can't regress by definition."""
    if previous_score is None:
        return VerificationResult(passed=True, score=new_score)
    if new_score < previous_score:
        return VerificationResult(
            passed=False,
            score=new_score,
            failures=[f"regression: score dropped from {previous_score:.2f} to {new_score:.2f}"],
            recommended_repair="roll back to the previous version",
        )
    return VerificationResult(passed=True, score=new_score)


__all__ = [
    "MemoryCandidate",
    "SkillCandidate",
    "SkillRegistry",
    "SkillReviewResult",
    "TrajectoryAnalysis",
    "TrajectoryAnalyzer",
    "check_regression",
    "generate_memory_candidates",
    "generate_skill_candidate",
]
