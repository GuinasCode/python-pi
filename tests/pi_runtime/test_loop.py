"""Tests for pi_runtime.loop.AgentRuntime against a real AgentSession
(faux provider — deterministic, no network). Covers Fase 1's five
acceptance criteria from plan.md section 5:

1. tarefas simples continuam funcionando
2. uma falha de tool pode provocar repair/replan
3. uma tarefa pode terminar sem uso de tools
4. uma tarefa pode terminar por budget
5. toda execução tem stop reason explícito
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.providers.faux import faux_provider as _faux_provider
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_runtime.loop import AgentRuntime, Executor, Planner, Replanner, Verifier
from pi_runtime.state import Budget, Goal, Plan, PlanStep, RunStatus, StopReason, VerificationResult


def _make_session(responses: list[Any]) -> AgentSession:
    handle = _faux_provider()
    handle.set_responses(responses)
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    return AgentSession(AgentSessionOptions(models=models, model=model, cwd="/tmp", enable_subagents=False))


class TestAgentRuntimeAcceptance:
    def test_simple_task_still_works(self) -> None:
        """Criterion 1 + 3: a plain text response, no tools involved."""
        session = _make_session([faux_assistant_message("The answer is 42.")])
        runtime = AgentRuntime()
        goal = Goal(objective="what is the answer?")

        state = asyncio.run(runtime.run(goal, session))

        assert state.status == RunStatus.DONE
        assert state.stop_reason == StopReason.COMPLETED
        assert state.final_text == "The answer is 42."
        assert state.plan is not None
        assert state.plan.is_complete()

    def test_failed_step_triggers_repair_and_recovers(self) -> None:
        """Criterion 2: an error response provokes a repair step, which
        then succeeds."""
        from pi_ai import StopReason as AiStopReason

        session = _make_session(
            [
                faux_assistant_message("boom", stop_reason=AiStopReason.ERROR, error_message="upstream failure"),
                faux_assistant_message("recovered fine"),
            ]
        )
        runtime = AgentRuntime()
        goal = Goal(objective="do the thing", budget=Budget(max_iterations=5))

        state = asyncio.run(runtime.run(goal, session))

        assert state.status == RunStatus.DONE
        assert state.stop_reason == StopReason.COMPLETED
        assert state.final_text == "recovered fine"
        assert state.plan is not None
        assert len(state.plan.steps) == 2  # original + one repair step
        assert state.iteration == 2

    def test_repeated_failures_give_up_after_max_attempts(self) -> None:
        """Two consecutive failures of the same logical objective give up
        (Replanner.max_attempts_per_step=2) rather than replanning
        forever — attempts carry forward from the original step into its
        repair steps."""
        from pi_ai import StopReason as AiStopReason

        session = _make_session(
            [
                faux_assistant_message("boom1", stop_reason=AiStopReason.ERROR, error_message="e1"),
                faux_assistant_message("boom2", stop_reason=AiStopReason.ERROR, error_message="e2"),
            ]
        )
        runtime = AgentRuntime()
        goal = Goal(objective="do the thing", budget=Budget(max_iterations=10))

        state = asyncio.run(runtime.run(goal, session))

        assert state.status == RunStatus.FAILED
        assert state.stop_reason == StopReason.ERROR
        assert state.error_message is not None
        assert state.iteration == 2  # gave up after exactly 2 attempts, not more

    def test_task_can_stop_on_budget(self) -> None:
        """Criterion 4: a budget that only allows one iteration stops the
        run even though a repair step was queued up for a second one."""
        from pi_ai import StopReason as AiStopReason

        session = _make_session(
            [
                faux_assistant_message("boom", stop_reason=AiStopReason.ERROR, error_message="e"),
                faux_assistant_message("would have recovered"),
            ]
        )
        runtime = AgentRuntime()
        goal = Goal(objective="do the thing", budget=Budget(max_iterations=1))

        state = asyncio.run(runtime.run(goal, session))

        assert state.status == RunStatus.STOPPED
        assert state.stop_reason == StopReason.MAX_ITERATIONS
        assert state.iteration == 1

    def test_every_run_has_an_explicit_stop_reason(self) -> None:
        """Criterion 5, checked across every scenario above plus a bare
        happy path."""
        session = _make_session([faux_assistant_message("done")])
        state = asyncio.run(AgentRuntime().run(Goal(objective="x"), session))
        assert state.stop_reason is not None
        assert isinstance(state.stop_reason, StopReason)

    def test_executor_exception_produces_error_stop_reason(self) -> None:
        """A raised exception (not just a StopReason.ERROR response) must
        also always resolve to an explicit stop reason, never propagate
        out of run() uncaught."""

        class _BoomExecutor(Executor):
            async def execute(self, step: PlanStep, session: AgentSession, state: Any) -> Any:
                raise RuntimeError("executor blew up")

        session = _make_session([faux_assistant_message("unused")])
        runtime = AgentRuntime(executor=_BoomExecutor())
        state = asyncio.run(runtime.run(Goal(objective="x"), session))

        assert state.status == RunStatus.FAILED
        assert state.stop_reason == StopReason.ERROR
        assert "executor blew up" in (state.error_message or "")


class TestPolicyEngineIntegration:
    """PolicyEngine (Fase 3) is opt-in on Executor — these confirm it
    actually blocks a step from running at all when a currently-active
    tool is unregistered/denied, and that Executor without one behaves
    exactly like Fase 1/2 (no policy check)."""

    def test_no_policy_engine_means_no_check_at_all(self) -> None:
        session = _make_session([faux_assistant_message("ok")])
        from pi_runtime.state import AgentState

        state = AgentState(goal=Goal(objective="x"))
        step = PlanStep(objective="x", action="go")
        result = asyncio.run(Executor().execute(step, session, state))
        assert result.content  # ran normally, nothing blocked it

    def test_policy_violation_prevents_the_step_from_running(self) -> None:
        from pi_runtime.state import AgentState
        from pi_runtime.tools import PolicyEngine, ToolRegistry

        # A session built with real builtin tools (bash included) but a
        # registry that only knows about "read" — bash is active on the
        # session yet unregistered in the policy layer.
        session = _make_session([faux_assistant_message("should never be reached")])
        registry = ToolRegistry()
        from pi_runtime.tools import Risk, ToolSpec

        for name in session.get_active_tool_names():
            if name != "bash":
                registry.register(ToolSpec(name=name, risk=Risk.NONE))
        policy_engine = PolicyEngine(registry)

        executor = Executor(policy_engine=policy_engine)
        state = AgentState(goal=Goal(objective="x"))
        step = PlanStep(objective="x", action="go")

        with pytest.raises(Exception, match="not registered"):
            asyncio.run(executor.execute(step, session, state))

        # and via AgentRuntime, this surfaces as an explicit stop reason,
        # never an uncaught crash.
        runtime = AgentRuntime(executor=Executor(policy_engine=policy_engine))
        state2 = asyncio.run(runtime.run(Goal(objective="x"), session))
        assert state2.stop_reason == StopReason.ERROR
        assert state2.status == RunStatus.FAILED


class TestContextEngineIntegration:
    """Executor is the Context Engine's real consumer (plan.md Fase 2) —
    these confirm decisions/constraints/evidence accumulated on AgentState
    actually reach the model as steering context, and that a state with
    nothing accumulated yet behaves exactly like Fase 1 (no injected
    note)."""

    def test_accumulated_decision_is_surfaced_to_the_next_execute_call(self) -> None:
        from pi_ai import UserMessage
        from pi_runtime.state import AgentState

        session = _make_session([faux_assistant_message("ok")])
        state = AgentState(goal=Goal(objective="x"), decisions=["chose SQLite over Postgres"])
        step = PlanStep(objective="x", action="continue the task")

        asyncio.run(Executor().execute(step, session, state))

        messages = session.get_messages()
        decision_messages = [
            m for m in messages if isinstance(m, UserMessage) and "chose SQLite over Postgres" in str(m.content)
        ]
        assert len(decision_messages) == 1

    def test_no_accumulated_state_means_no_injected_note(self) -> None:
        """Fase 1's own tests already assert this indirectly (no test
        there populates decisions/constraints/evidence and they all still
        pass) — this makes the guarantee explicit at the Executor level."""
        from pi_runtime.state import AgentState

        session = _make_session([faux_assistant_message("ok")])
        state = AgentState(goal=Goal(objective="x"))
        step = PlanStep(objective="x", action="do the task")

        asyncio.run(Executor().execute(step, session, state))

        messages = session.get_messages()
        # exactly one user message: the step's own action text, nothing injected
        from pi_ai import UserMessage

        user_messages = [m for m in messages if isinstance(m, UserMessage)]
        assert len(user_messages) == 1
        assert user_messages[0].content == "do the task"


class TestPlannerVerifierReplannerUnits:
    def test_planner_produces_a_single_step_plan(self) -> None:
        plan = Planner().plan(Goal(objective="research X"))
        assert len(plan.steps) == 1
        assert plan.steps[0].action == "research X"

    def test_verifier_rejects_empty_text(self) -> None:
        from pi_ai import AssistantMessage
        from pi_ai import StopReason as AiStopReason

        result = AssistantMessage(content=[], stop_reason=AiStopReason.STOP)
        verification = Verifier().verify(PlanStep(objective="x"), result)
        assert not verification.passed
        assert verification.failures

    def test_replanner_stops_after_max_attempts(self) -> None:
        step = PlanStep(objective="x")
        step.attempts = 2
        plan = Plan(goal=Goal(objective="x"), steps=[step])
        replanner = Replanner()
        added = replanner.replan(plan, step, VerificationResult(passed=False, failures=["nope"]))
        assert added is False
        assert len(plan.steps) == 1

    def test_replanner_inserts_repair_step_within_budget(self) -> None:
        step = PlanStep(objective="x")
        step.attempts = 1
        plan = Plan(goal=Goal(objective="x"), steps=[step])
        replanner = Replanner()
        added = replanner.replan(plan, step, VerificationResult(passed=False, failures=["nope"]))
        assert added is True
        assert len(plan.steps) == 2
        assert plan.steps[1].attempts == 1  # carried forward, not reset
