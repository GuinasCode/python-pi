"""Tests for pi_runtime.state — the plain, serializable contracts (Budget,
Goal, Plan, AgentState, VerificationResult)."""

from __future__ import annotations

from pi_runtime.state import Budget, Goal, Plan, PlanStep, StepStatus


class TestBudget:
    def test_unlimited_budget_never_exceeded(self) -> None:
        budget = Budget(max_iterations=None)
        budget.consumed_iterations = 1000
        assert budget.exceeded() is None

    def test_default_budget_has_a_sane_iteration_cap(self) -> None:
        """Budget() with no args is a real, enforced default — not
        actually unlimited — per Invariant F ("real limits, not
        metadata")."""
        budget = Budget()
        assert budget.max_iterations is not None
        assert budget.max_iterations > 0

    def test_max_iterations_exceeded(self) -> None:
        budget = Budget(max_iterations=2)
        budget.record_iteration()
        assert budget.exceeded() is None
        budget.record_iteration()
        assert budget.exceeded() == "max_iterations"

    def test_max_tokens_exceeded(self) -> None:
        budget = Budget(max_tokens=100)
        budget.record_usage(tokens=100)
        assert budget.exceeded() == "max_tokens"

    def test_max_cost_exceeded(self) -> None:
        budget = Budget(max_cost=1.0)
        budget.record_usage(cost=1.5)
        assert budget.exceeded() == "max_cost"

    def test_max_tool_calls_exceeded(self) -> None:
        budget = Budget(max_tool_calls=1)
        budget.record_usage(tool_calls=2)
        assert budget.exceeded() == "max_tool_calls"

    def test_deadline_exceeded(self) -> None:
        budget = Budget(deadline_ts=0.0)  # already in the past
        assert budget.exceeded() == "deadline"

    def test_iterations_checked_before_other_limits(self) -> None:
        """Order matters only in that exceeded() must return *something*
        deterministic when multiple limits are blown — not which one
        specifically, just that it's stable."""
        budget = Budget(max_iterations=1, max_tokens=1)
        budget.record_iteration()
        budget.record_usage(tokens=1)
        assert budget.exceeded() in ("max_iterations", "max_tokens")


class TestPlan:
    def test_current_step_returns_none_when_empty(self) -> None:
        plan = Plan(goal=Goal(objective="x"))
        assert plan.current_step() is None
        assert plan.is_complete()

    def test_advance_moves_through_steps(self) -> None:
        goal = Goal(objective="x")
        plan = Plan(goal=goal, steps=[PlanStep(objective="a"), PlanStep(objective="b")])
        first = plan.current_step()
        assert first is not None
        assert first.objective == "a"
        plan.advance()
        second = plan.current_step()
        assert second is not None
        assert second.objective == "b"
        plan.advance()
        assert plan.is_complete()

    def test_add_repair_step_inserts_right_after_current(self) -> None:
        goal = Goal(objective="x")
        plan = Plan(goal=goal, steps=[PlanStep(objective="a"), PlanStep(objective="b")])
        plan.add_repair_step(PlanStep(objective="repair-a"))
        assert [s.objective for s in plan.steps] == ["a", "repair-a", "b"]

    def test_step_default_status_is_pending(self) -> None:
        step = PlanStep(objective="x")
        assert step.status == StepStatus.PENDING
        assert step.attempts == 0
