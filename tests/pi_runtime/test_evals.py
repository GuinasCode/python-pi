"""Tests for pi_runtime.evals. Covers Fase 16's acceptance criteria from
plan.md section 20: every named sub-metric across the Agent/Research/
Memory/Delegation/Skills suites, computed mechanically (not "output
contains X") from real pi_runtime data shapes.
"""

from __future__ import annotations

from pi_runtime.delegation import DelegationOutcome, DelegationRequest
from pi_runtime.evals import (
    agent_recovery,
    agent_replanning,
    agent_success,
    agent_tool_efficiency,
    delegation_failure_isolation,
    delegation_parallel_speedup,
    delegation_task_success,
    memory_contamination,
    memory_retrieval_precision_recall,
    memory_stale_rate,
    research_citation_completeness,
    research_citation_precision,
    research_contradiction_detection,
    research_coverage,
    research_source_quality,
    research_unsupported_claims_rate,
    skills_regression_rate,
    skills_selection_accuracy,
    skills_success_lift,
)
from pi_runtime.memory import RankedMemory
from pi_runtime.research import Claim, Evidence, ResearchResult
from pi_runtime.skills import SkillSelection
from pi_runtime.state import AgentState, Goal, Plan, PlanStep, RunStatus, StepStatus


def _record(id_: int) -> object:
    from pi_memory.store import MemoryRecord, MemoryType

    return MemoryRecord(
        id=id_,
        type=MemoryType.USER,
        title="t",
        content="c",
        project_cwd=None,
        created_at=0,
        updated_at=0,
        source="auto",
    )


class TestAgentSuite:
    def test_success_reflects_run_status(self) -> None:
        assert agent_success(AgentState(status=RunStatus.DONE)).passed
        assert not agent_success(AgentState(status=RunStatus.FAILED)).passed

    def test_tool_efficiency_is_one_when_every_step_solved_first_try(self) -> None:
        steps = [PlanStep(objective="a", attempts=1), PlanStep(objective="b", attempts=1)]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps))
        result = agent_tool_efficiency(state)
        assert result.metrics["efficiency"] == 1.0

    def test_tool_efficiency_drops_with_repairs(self) -> None:
        steps = [PlanStep(objective="a", attempts=3)]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps))
        result = agent_tool_efficiency(state)
        assert result.metrics["efficiency"] < 1.0

    def test_recovery_passes_when_no_repair_was_needed(self) -> None:
        steps = [PlanStep(objective="a", attempts=1, status=StepStatus.DONE)]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps), status=RunStatus.DONE)
        result = agent_recovery(state)
        assert result.passed
        assert result.details["applicable"] is False

    def test_recovery_fails_when_repair_happened_but_run_still_failed(self) -> None:
        steps = [PlanStep(objective="a", attempts=2, status=StepStatus.FAILED)]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps), status=RunStatus.FAILED)
        result = agent_recovery(state)
        assert not result.passed

    def test_recovery_passes_when_repair_happened_and_run_succeeded(self) -> None:
        steps = [PlanStep(objective="a", attempts=2, status=StepStatus.DONE)]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps), status=RunStatus.DONE)
        assert agent_recovery(state).passed

    def test_replanning_counts_repaired_steps(self) -> None:
        steps = [
            PlanStep(objective="a", attempts=2, status=StepStatus.DONE),
            PlanStep(objective="b", attempts=1, status=StepStatus.DONE),
        ]
        state = AgentState(plan=Plan(goal=Goal(objective="x"), steps=steps))
        assert agent_replanning(state).metrics["replan_count"] == 1.0


class TestResearchSuite:
    def test_source_quality_averages_reliability(self) -> None:
        result = ResearchResult(
            question="q",
            evidence=[
                Evidence(source_id="a", url="a", title="a", excerpt="", retrieved_at=0, reliability=1.0),
                Evidence(source_id="b", url="b", title="b", excerpt="", retrieved_at=0, reliability=0.0),
            ],
        )
        assert research_source_quality(result).metrics["avg_reliability"] == 0.5

    def test_citation_precision_catches_unsupported_claims(self) -> None:
        result = ResearchResult(question="q", claims=[Claim(text="x", supported=True, evidence_refs=[])])
        eval_result = research_citation_precision(result)
        assert not eval_result.passed
        assert eval_result.metrics["precision"] == 0.0

    def test_citation_precision_passes_when_all_supported_claims_cite(self) -> None:
        result = ResearchResult(question="q", claims=[Claim(text="x", supported=True, evidence_refs=["s1"])])
        assert research_citation_precision(result).passed

    def test_citation_completeness_reflects_unused_evidence(self) -> None:
        result = ResearchResult(
            question="q",
            evidence=[Evidence(source_id="s1", url="u", title="t", excerpt="", retrieved_at=0)],
            claims=[],
        )
        assert research_citation_completeness(result).metrics["completeness"] == 0.0

    def test_unsupported_claims_rate(self) -> None:
        result = ResearchResult(
            question="q",
            claims=[
                Claim(text="a", supported=True, evidence_refs=["s1"]),
                Claim(text="b", supported=True, evidence_refs=[]),
            ],
        )
        assert research_unsupported_claims_rate(result).metrics["rate"] == 0.5

    def test_contradiction_detection_counts_contradicted_claims(self) -> None:
        result = ResearchResult(question="q", claims=[Claim(text="a", contradicted=True)])
        assert research_contradiction_detection(result).metrics["contradictions_found"] == 1.0

    def test_coverage_relative_to_requested_urls(self) -> None:
        result = ResearchResult(
            question="q", evidence=[Evidence(source_id="a", url="a", title="a", excerpt="", retrieved_at=0)]
        )
        assert research_coverage(result, requested_urls=2).metrics["coverage"] == 0.5


class TestMemorySuite:
    def test_retrieval_precision_and_recall(self) -> None:
        ranked = [
            RankedMemory(record=_record(1), freshness=1.0, combined_score=1.0),
            RankedMemory(record=_record(2), freshness=1.0, combined_score=0.5),
        ]  # type: ignore[arg-type]
        result = memory_retrieval_precision_recall(ranked, relevant_ids={1})
        assert result.metrics["precision"] == 0.5
        assert result.metrics["recall"] == 1.0

    def test_contamination_reflects_irrelevant_fraction(self) -> None:
        ranked = [
            RankedMemory(record=_record(1), freshness=1.0, combined_score=1.0),
            RankedMemory(record=_record(2), freshness=1.0, combined_score=0.5),
        ]  # type: ignore[arg-type]
        assert memory_contamination(ranked, relevant_ids={1}).metrics["contamination"] == 0.5

    def test_stale_rate_reflects_low_freshness_items(self) -> None:
        ranked = [RankedMemory(record=_record(1), freshness=0.01, combined_score=1.0)]  # type: ignore[arg-type]
        assert memory_stale_rate(ranked).metrics["stale_rate"] == 1.0

    def test_empty_ranked_list_is_never_stale_or_contaminated(self) -> None:
        assert memory_stale_rate([]).metrics["stale_rate"] == 0.0
        assert memory_contamination([], relevant_ids=set()).metrics["contamination"] == 0.0


class TestDelegationSuite:
    def _outcome(self, *, succeeded: bool, elapsed: float = 1.0) -> DelegationOutcome:
        from pi_coding_agent.subagent.registry import SubagentResult

        request = DelegationRequest(objective="x")
        if succeeded:
            return DelegationOutcome(
                request=request,
                result=SubagentResult(output="ok", exit_code=0, agent_name="x", status="done"),
                elapsed_seconds=elapsed,
            )
        return DelegationOutcome(request=request, error="boom", elapsed_seconds=elapsed)

    def test_task_success_rate(self) -> None:
        outcomes = [self._outcome(succeeded=True), self._outcome(succeeded=False)]
        assert delegation_task_success(outcomes).metrics["success_rate"] == 0.5

    def test_parallel_speedup_above_one_when_overlap_happened(self) -> None:
        outcomes = [self._outcome(succeeded=True, elapsed=2.0), self._outcome(succeeded=True, elapsed=2.0)]
        result = delegation_parallel_speedup(outcomes, wall_clock_seconds=2.0)
        assert result.metrics["speedup"] == 2.0
        assert result.passed

    def test_failure_isolation_requires_both_a_success_and_a_failure(self) -> None:
        outcomes = [self._outcome(succeeded=True), self._outcome(succeeded=False)]
        result = delegation_failure_isolation(outcomes)
        assert result.passed
        assert result.metrics["isolated"] == 1.0

    def test_failure_isolation_not_applicable_with_no_failures(self) -> None:
        outcomes = [self._outcome(succeeded=True), self._outcome(succeeded=True)]
        result = delegation_failure_isolation(outcomes)
        assert result.details["applicable"] is False


class TestSkillsSuite:
    def test_selection_accuracy_against_expected_names(self) -> None:
        selections = [
            SkillSelection(name="a", description="", score=1.0, selected=True),
            SkillSelection(name="b", description="", score=0.1, selected=False),
        ]
        result = skills_selection_accuracy(selections, expected_names={"a"})
        assert result.metrics["accuracy"] == 1.0

    def test_success_lift_can_be_negative(self) -> None:
        result = skills_success_lift(success_rate_with_skill=0.3, baseline_success_rate=0.5)
        assert result.metrics["lift"] == -0.2
        assert not result.passed

    def test_regression_rate_uses_learning_check_regression(self) -> None:
        result = skills_regression_rate(previous_score=0.9, new_score=0.5)
        assert not result.passed
        assert result.metrics["regressed"] == 1.0

    def test_no_regression_when_score_improves(self) -> None:
        result = skills_regression_rate(previous_score=0.5, new_score=0.9)
        assert result.passed
