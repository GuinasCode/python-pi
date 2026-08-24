"""Eval Platform — Fase 16 of the research-first-runtime plan.

Does not replace pi_evals (Regra 1.1) — pi_evals already has a real
harness (pi_harness.py wraps AgentSession), LLM-as-judge scoring
(judges.py), and comparative baseline/candidate tables
(harness_table.py) for evaluating chat-style agent *output quality*.
What's missing, and what plan.md's Fase 16 actually asks for: mechanical
behavioral metrics computed directly from pi_runtime's own structured
outputs (AgentState, ResearchResult, RankedMemory, DelegationOutcome,
SkillSelection) — efficiency/coverage/recovery/evidence-quality/
delegation-efficiency/skill-regression, not "output contains X"
(plan.md's own explicit rule against measuring only that). These don't
need an LLM judge: they're deterministic functions over data every
earlier phase (1-15) already produces.

Each suite below (Agent/Research/Memory/Delegation/Skills) matches
plan.md section 20's exact list of capability sub-metrics for that
suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_runtime.delegation import DelegationOutcome
from pi_runtime.learning import check_regression
from pi_runtime.memory import RankedMemory
from pi_runtime.research import ResearchResult
from pi_runtime.skills import SkillSelection
from pi_runtime.state import AgentState, RunStatus, StepStatus


@dataclass
class EvalResult:
    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


# --- Agent suite: success, tool efficiency, recovery, replanning ----------


def agent_success(state: AgentState) -> EvalResult:
    return EvalResult(
        name="agent.success",
        passed=state.status == RunStatus.DONE,
        metrics={"success": float(state.status == RunStatus.DONE)},
    )


def agent_tool_efficiency(state: AgentState) -> EvalResult:
    """Lower attempts-per-step is more efficient — 1.0 (every step
    solved first try) is the ceiling."""
    if state.plan is None or not state.plan.steps:
        return EvalResult(name="agent.tool_efficiency", passed=False, metrics={"efficiency": 0.0})
    total_attempts = sum(step.attempts for step in state.plan.steps)
    efficiency = len(state.plan.steps) / total_attempts if total_attempts else 0.0
    return EvalResult(name="agent.tool_efficiency", passed=efficiency > 0, metrics={"efficiency": efficiency})


def agent_recovery(state: AgentState) -> EvalResult:
    """Did the run recover from at least one failed step and still
    finish? Only meaningful when a repair actually happened — a run with
    no repairs at all has nothing to "recover" from."""
    if state.plan is None:
        return EvalResult(
            name="agent.recovery", passed=False, metrics={"recovered": 0.0}, details={"applicable": False}
        )
    repaired_steps = [s for s in state.plan.steps if s.attempts > 1]
    if not repaired_steps:
        return EvalResult(name="agent.recovery", passed=True, metrics={"recovered": 1.0}, details={"applicable": False})
    recovered = state.status == RunStatus.DONE
    return EvalResult(name="agent.recovery", passed=recovered, metrics={"recovered": float(recovered)})


def agent_replanning(state: AgentState) -> EvalResult:
    """How many repair steps a plan actually needed — plan.md doesn't
    define a target direction (fewer replans is efficient, but zero
    replans on a genuinely hard task may mean under-verification), so
    this reports the count without a pass/fail judgment call."""
    if state.plan is None:
        return EvalResult(name="agent.replanning", passed=True, metrics={"replan_count": 0.0})
    replans = sum(1 for s in state.plan.steps if s.status in (StepStatus.DONE, StepStatus.FAILED) and s.attempts > 1)
    return EvalResult(name="agent.replanning", passed=True, metrics={"replan_count": float(replans)})


# --- Research suite: source quality, citation precision/completeness, -----
# --- unsupported claims, contradiction detection, coverage ----------------


def research_source_quality(result: ResearchResult) -> EvalResult:
    if not result.evidence:
        return EvalResult(name="research.source_quality", passed=False, metrics={"avg_reliability": 0.0})
    avg = sum(e.reliability for e in result.evidence) / len(result.evidence)
    return EvalResult(name="research.source_quality", passed=avg >= 0.5, metrics={"avg_reliability": avg})


def research_citation_precision(result: ResearchResult) -> EvalResult:
    """Of the claims marked supported, what fraction actually cite
    evidence? plan.md's own rule (section 8): a supported claim with no
    evidence_refs is exactly what this must catch."""
    supported = [c for c in result.claims if c.supported]
    if not supported:
        return EvalResult(name="research.citation_precision", passed=True, metrics={"precision": 1.0})
    cited = sum(1 for c in supported if c.evidence_refs)
    precision = cited / len(supported)
    return EvalResult(name="research.citation_precision", passed=precision == 1.0, metrics={"precision": precision})


def research_citation_completeness(result: ResearchResult) -> EvalResult:
    """Of the evidence gathered, what fraction is actually cited by some
    claim? Evidence gathered but never used is a coverage gap on the
    synthesis side, not the retrieval side."""
    if not result.evidence:
        return EvalResult(name="research.citation_completeness", passed=False, metrics={"completeness": 0.0})
    cited_ids = {ref for claim in result.claims for ref in claim.evidence_refs}
    used = sum(1 for e in result.evidence if e.source_id in cited_ids)
    completeness = used / len(result.evidence)
    return EvalResult(
        name="research.citation_completeness", passed=completeness > 0, metrics={"completeness": completeness}
    )


def research_unsupported_claims_rate(result: ResearchResult) -> EvalResult:
    if not result.claims:
        return EvalResult(name="research.unsupported_claims_rate", passed=True, metrics={"rate": 0.0})
    unsupported = sum(1 for c in result.claims if c.supported and not c.evidence_refs)
    rate = unsupported / len(result.claims)
    return EvalResult(name="research.unsupported_claims_rate", passed=rate == 0.0, metrics={"rate": rate})


def research_contradiction_detection(result: ResearchResult) -> EvalResult:
    contradicted = sum(1 for c in result.claims if c.contradicted)
    return EvalResult(
        name="research.contradiction_detection", passed=True, metrics={"contradictions_found": float(contradicted)}
    )


def research_coverage(result: ResearchResult, *, requested_urls: int) -> EvalResult:
    if requested_urls == 0:
        return EvalResult(name="research.coverage", passed=False, metrics={"coverage": 0.0})
    coverage = len(result.evidence) / requested_urls
    return EvalResult(name="research.coverage", passed=coverage > 0, metrics={"coverage": coverage})


# --- Memory suite: retrieval precision/recall, contamination, ------------
# --- stale-memory rate -----------------------------------------------------


def memory_retrieval_precision_recall(ranked: list[RankedMemory], *, relevant_ids: set[int]) -> EvalResult:
    retrieved_ids = {r.record.id for r in ranked}
    true_positives = retrieved_ids & relevant_ids
    precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(true_positives) / len(relevant_ids) if relevant_ids else 1.0
    return EvalResult(
        name="memory.retrieval",
        passed=precision > 0 or not relevant_ids,
        metrics={"precision": precision, "recall": recall},
    )


def memory_contamination(ranked: list[RankedMemory], *, relevant_ids: set[int]) -> EvalResult:
    """What fraction of retrieved memories are irrelevant noise? The
    inverse of precision, reported separately because "contamination"
    is plan.md's own named metric, not just 1-precision restated."""
    if not ranked:
        return EvalResult(name="memory.contamination", passed=True, metrics={"contamination": 0.0})
    irrelevant = sum(1 for r in ranked if r.record.id not in relevant_ids)
    contamination = irrelevant / len(ranked)
    return EvalResult(name="memory.contamination", passed=contamination < 0.5, metrics={"contamination": contamination})


def memory_stale_rate(ranked: list[RankedMemory], *, freshness_threshold: float = 0.1) -> EvalResult:
    if not ranked:
        return EvalResult(name="memory.stale_rate", passed=True, metrics={"stale_rate": 0.0})
    stale = sum(1 for r in ranked if r.freshness < freshness_threshold)
    rate = stale / len(ranked)
    return EvalResult(name="memory.stale_rate", passed=rate < 0.5, metrics={"stale_rate": rate})


# --- Delegation suite: task success, parallel speedup, token savings, -----
# --- failure isolation ------------------------------------------------------


def delegation_task_success(outcomes: list[DelegationOutcome]) -> EvalResult:
    if not outcomes:
        return EvalResult(name="delegation.task_success", passed=False, metrics={"success_rate": 0.0})
    rate = sum(1 for o in outcomes if o.succeeded) / len(outcomes)
    return EvalResult(name="delegation.task_success", passed=rate > 0, metrics={"success_rate": rate})


def delegation_parallel_speedup(outcomes: list[DelegationOutcome], *, wall_clock_seconds: float) -> EvalResult:
    """Sum of each delegation's own elapsed time, divided by the actual
    wall-clock time the parallel batch took — >1.0 means real overlap
    happened, not sequential-looking-parallel."""
    if not outcomes or wall_clock_seconds <= 0:
        return EvalResult(name="delegation.parallel_speedup", passed=False, metrics={"speedup": 0.0})
    sequential_total = sum(o.elapsed_seconds for o in outcomes)
    speedup = sequential_total / wall_clock_seconds
    return EvalResult(name="delegation.parallel_speedup", passed=speedup >= 1.0, metrics={"speedup": speedup})


def delegation_failure_isolation(outcomes: list[DelegationOutcome]) -> EvalResult:
    """At least one success surviving alongside at least one failure is
    the actual signal that failure isolation worked — a batch that's
    all-success or all-failure doesn't exercise isolation at all."""
    if not outcomes:
        return EvalResult(name="delegation.failure_isolation", passed=False, metrics={"isolated": 0.0})
    has_failure = any(not o.succeeded for o in outcomes)
    has_success = any(o.succeeded for o in outcomes)
    isolated = has_failure and has_success
    return EvalResult(
        name="delegation.failure_isolation",
        passed=isolated or not has_failure,
        metrics={"isolated": float(isolated)},
        details={"applicable": has_failure},
    )


# --- Skills suite: selection accuracy, success lift, regression rate ------


def skills_selection_accuracy(selections: list[SkillSelection], *, expected_names: set[str]) -> EvalResult:
    selected_names = {s.name for s in selections if s.selected}
    if not expected_names:
        return EvalResult(name="skills.selection_accuracy", passed=True, metrics={"accuracy": 1.0})
    correct = len(selected_names & expected_names)
    accuracy = correct / len(expected_names)
    return EvalResult(name="skills.selection_accuracy", passed=accuracy > 0, metrics={"accuracy": accuracy})


def skills_success_lift(*, success_rate_with_skill: float, baseline_success_rate: float) -> EvalResult:
    lift = success_rate_with_skill - baseline_success_rate
    return EvalResult(name="skills.success_lift", passed=lift >= 0, metrics={"lift": lift})


def skills_regression_rate(*, previous_score: float | None, new_score: float) -> EvalResult:
    """Reuses pi_runtime.learning.check_regression (Fase 8) rather than
    a second regression concept."""
    verification = check_regression(previous_score=previous_score, new_score=new_score)
    return EvalResult(
        name="skills.regression",
        passed=verification.passed,
        metrics={"score": new_score, "regressed": float(not verification.passed)},
        details={"failures": verification.failures},
    )


__all__ = [
    "EvalResult",
    "agent_recovery",
    "agent_replanning",
    "agent_success",
    "agent_tool_efficiency",
    "delegation_failure_isolation",
    "delegation_parallel_speedup",
    "delegation_task_success",
    "memory_contamination",
    "memory_retrieval_precision_recall",
    "memory_stale_rate",
    "research_citation_completeness",
    "research_citation_precision",
    "research_contradiction_detection",
    "research_coverage",
    "research_source_quality",
    "research_unsupported_claims_rate",
    "skills_regression_rate",
    "skills_selection_accuracy",
    "skills_success_lift",
]
