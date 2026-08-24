"""Tests for pi_runtime.learning. Covers Fase 8's acceptance criteria
from plan.md section 12:

- uma execução pode gerar candidato de memória
- pode gerar candidato de skill
- atualização pode ser rejeitada
- atualização pode ser revertida
- regressão de skill gera falha de eval
"""

from __future__ import annotations

from pi_runtime.learning import (
    SkillRegistry,
    TrajectoryAnalyzer,
    check_regression,
    generate_memory_candidates,
    generate_skill_candidate,
)
from pi_runtime.state import AgentState, Goal, Plan, PlanStep, RunStatus, StopReason, VerificationResult


def _succeeded_state(*, decisions: list[str], repaired: bool = False) -> AgentState:
    step = PlanStep(objective="x")
    step.attempts = 2 if repaired else 1
    state = AgentState(
        plan=Plan(goal=Goal(objective="x"), steps=[step]),
        decisions=decisions,
        status=RunStatus.DONE,
        stop_reason=StopReason.COMPLETED,
    )
    return state


def _failed_state(*, error: str) -> AgentState:
    return AgentState(status=RunStatus.FAILED, stop_reason=StopReason.ERROR, error_message=error)


class TestTrajectoryAnalyzer:
    def test_successful_run_is_recognized(self) -> None:
        state = _succeeded_state(decisions=["chose SQLite"])
        analysis = TrajectoryAnalyzer().analyze(state)
        assert analysis.succeeded
        assert analysis.failures == []

    def test_failed_run_captures_the_error(self) -> None:
        state = _failed_state(error="upstream failure")
        analysis = TrajectoryAnalyzer().analyze(state)
        assert not analysis.succeeded
        assert "upstream failure" in analysis.failures

    def test_verification_failures_are_captured_too(self) -> None:
        state = _failed_state(error="")
        state.verification = VerificationResult(passed=False, failures=["no text content"])
        analysis = TrajectoryAnalyzer().analyze(state)
        assert "no text content" in analysis.failures

    def test_repaired_steps_are_counted(self) -> None:
        state = _succeeded_state(decisions=[], repaired=True)
        analysis = TrajectoryAnalyzer().analyze(state)
        assert analysis.repaired_steps == 1
        assert analysis.total_attempts == 2

    def test_no_repair_means_zero_repaired_steps(self) -> None:
        state = _succeeded_state(decisions=[])
        analysis = TrajectoryAnalyzer().analyze(state)
        assert analysis.repaired_steps == 0


class TestMemoryCandidates:
    def test_successful_run_generates_a_candidate_per_decision(self) -> None:
        state = _succeeded_state(decisions=["chose SQLite over Postgres", "used FTS5 for search"])
        analysis = TrajectoryAnalyzer().analyze(state)
        candidates = generate_memory_candidates(analysis)
        assert len(candidates) == 2
        assert candidates[0].cognitive_type == "episodic"
        assert candidates[0].source_run_id == analysis.run_id

    def test_failed_run_generates_no_candidates(self) -> None:
        state = _failed_state(error="boom")
        analysis = TrajectoryAnalyzer().analyze(state)
        assert generate_memory_candidates(analysis) == []

    def test_no_decisions_generates_no_candidates(self) -> None:
        state = _succeeded_state(decisions=[])
        analysis = TrajectoryAnalyzer().analyze(state)
        assert generate_memory_candidates(analysis) == []


class TestSkillCandidateGeneration:
    def test_a_repaired_run_generates_a_skill_candidate(self) -> None:
        state = _succeeded_state(decisions=[], repaired=True)
        analysis = TrajectoryAnalyzer().analyze(state)
        candidate = generate_skill_candidate(analysis, name="handle-flaky-api", repair_text="retry once on timeout")
        assert candidate is not None
        assert candidate.name == "handle-flaky-api"
        assert "retry once on timeout" in candidate.diff
        assert candidate.status == "pending"

    def test_a_first_try_success_generates_no_skill_candidate(self) -> None:
        state = _succeeded_state(decisions=[])
        analysis = TrajectoryAnalyzer().analyze(state)
        candidate = generate_skill_candidate(analysis, name="x", repair_text="irrelevant")
        assert candidate is None


class TestSkillRegistryApprovalFlow:
    def test_approved_candidate_is_applied(self) -> None:
        registry = SkillRegistry()
        candidate = generate_skill_candidate(
            TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
            name="skill-a",
            repair_text="do the thing differently",
        )
        assert candidate is not None
        result = registry.review(candidate, approve=True)
        applied = registry.apply(result)

        assert applied
        current = registry.current("skill-a")
        assert current is not None
        assert current.content == "do the thing differently"
        assert current.version == 1

    def test_rejected_candidate_is_not_applied(self) -> None:
        """Fase 8 acceptance criterion 3."""
        registry = SkillRegistry()
        candidate = generate_skill_candidate(
            TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
            name="skill-b",
            repair_text="a sketchy workaround",
        )
        assert candidate is not None
        result = registry.review(candidate, approve=False, reason="too narrow a fix")
        applied = registry.apply(result)

        assert not applied
        assert registry.current("skill-b") is None
        assert candidate.status == "rejected"

    def test_second_approved_version_increments(self) -> None:
        registry = SkillRegistry()
        for text in ("v1 text", "v2 text"):
            candidate = generate_skill_candidate(
                TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
                name="skill-c",
                repair_text=text,
            )
            assert candidate is not None
            registry.apply(registry.review(candidate, approve=True))

        assert registry.current("skill-c").version == 2  # type: ignore[union-attr]
        assert len(registry.history("skill-c")) == 2


class TestRollback:
    def test_rollback_restores_an_older_version_as_current(self) -> None:
        """Fase 8 acceptance criterion 4."""
        registry = SkillRegistry()
        for text in ("v1 text", "v2 text (buggy)"):
            candidate = generate_skill_candidate(
                TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
                name="skill-d",
                repair_text=text,
            )
            assert candidate is not None
            registry.apply(registry.review(candidate, approve=True))

        rolled_back = registry.rollback("skill-d", to_version=1)
        assert rolled_back
        current = registry.current("skill-d")
        assert current is not None
        assert current.content == "v1 text"
        assert current.version == 3  # rollback appends, never rewrites history

    def test_rollback_preserves_full_history(self) -> None:
        registry = SkillRegistry()
        candidate = generate_skill_candidate(
            TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
            name="skill-e",
            repair_text="v1",
        )
        assert candidate is not None
        registry.apply(registry.review(candidate, approve=True))
        registry.rollback("skill-e", to_version=1)

        assert len(registry.history("skill-e")) == 2  # original + the rollback copy, nothing deleted

    def test_rollback_to_unknown_version_fails_explicitly(self) -> None:
        registry = SkillRegistry()
        candidate = generate_skill_candidate(
            TrajectoryAnalyzer().analyze(_succeeded_state(decisions=[], repaired=True)),
            name="skill-f",
            repair_text="v1",
        )
        assert candidate is not None
        registry.apply(registry.review(candidate, approve=True))

        assert registry.rollback("skill-f", to_version=99) is False

    def test_rollback_on_unknown_skill_fails_explicitly(self) -> None:
        registry = SkillRegistry()
        assert registry.rollback("never-existed", to_version=1) is False


class TestRegressionCheck:
    def test_first_version_cannot_regress(self) -> None:
        result = check_regression(previous_score=None, new_score=0.5)
        assert result.passed

    def test_lower_score_is_a_regression(self) -> None:
        """Fase 8 acceptance criterion 5."""
        result = check_regression(previous_score=0.9, new_score=0.6)
        assert not result.passed
        assert result.failures

    def test_equal_or_higher_score_passes(self) -> None:
        assert check_regression(previous_score=0.7, new_score=0.7).passed
        assert check_regression(previous_score=0.7, new_score=0.9).passed
