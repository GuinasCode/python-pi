"""The runtime loop: goal -> plan -> act -> observe -> verify -> replan | finish.

Every actual model/tool call still goes through the existing, unmodified
AgentSession.prompt() (pi_coding_agent.agent_session) — this module adds an
explicit state machine on top of it, it does not reimplement the agent
loop (Invariant A: single runtime, no parallel agent loops with different
rules).
"""

from __future__ import annotations

import time
from typing import Any

from pi_ai import AssistantMessage, TextContent
from pi_ai import StopReason as AiStopReason
from pi_coding_agent.agent_session import AgentSession
from pi_runtime.context import ContextEngine
from pi_runtime.state import (
    AgentState,
    Goal,
    Plan,
    PlanStep,
    RunStatus,
    StepStatus,
    StopReason,
    VerificationResult,
)
from pi_runtime.tools import PolicyEngine


def _final_text(result: AssistantMessage) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


class Planner:
    """Fase 1's planner is deliberately minimal: a single step that hands
    the goal straight to AgentSession's own conversational loop (which
    already does its own internal multi-turn tool calling). This is the
    smallest vertical slice that proves the state machine end to end —
    real multi-step task decomposition belongs to the Research Engine
    (Fase 4); adding it here now would duplicate AgentSession's turn loop
    for no benefit yet (plan.md Passo C: smallest vertical slice first)."""

    def plan(self, goal: Goal) -> Plan:
        step = PlanStep(
            objective=goal.objective,
            action=goal.objective,
            expected_outcome="A response that addresses the objective.",
            verification="stop_reason indicates a normal, non-error, non-aborted completion with real text",
        )
        return Plan(goal=goal, steps=[step])


class Executor:
    """Runs one PlanStep against a live AgentSession. The actual work
    (streaming, tool calling, steering, memory, soul, subagents) happens
    entirely inside AgentSession, unchanged — this only invokes it and
    tracks the step's own attempt count.

    Before each call, the Context Engine (Fase 2) renders whatever
    protected context (decisions/constraints/evidence/unresolved
    questions) has accumulated on `state` so far and, if there's
    anything, queues it as steering context so AgentSession's own
    compaction can't silently drop it — a run with no accumulated state
    yet (the common single-turn case) renders nothing and behaves
    exactly like Fase 1.

    If given a PolicyEngine (Fase 3), every tool currently active on the
    session is validated/evaluated *before* the step runs at all — an
    unregistered tool, a missing required environment, or a DENY
    decision raises PolicyViolation, which propagates out of execute()
    into AgentRuntime's existing exception handling (state.status=FAILED,
    stop_reason=ERROR). Without a PolicyEngine (the default), this check
    is skipped entirely — Fase 1/2 behavior is unchanged."""

    def __init__(
        self, *, context_engine: ContextEngine | None = None, policy_engine: PolicyEngine | None = None
    ) -> None:
        self._context_engine = context_engine or ContextEngine()
        self._policy_engine = policy_engine

    async def execute(self, step: PlanStep, session: AgentSession, state: AgentState) -> AssistantMessage:
        step.attempts += 1
        step.status = StepStatus.RUNNING
        if self._policy_engine is not None:
            self._policy_engine.evaluate_active_tools(session.get_active_tool_names())
        context_note = self._context_engine.render_working_set_note(state, session.get_messages())
        if context_note:
            session.queue_steer_message(context_note)
        result = await session.prompt(step.action)
        step.status = StepStatus.DONE if result.stop_reason != AiStopReason.ERROR else StepStatus.FAILED
        return result


class Verifier:
    """Never accept "parece que funcionou" as proof (plan.md secao 10).
    Fase 1's verifier is honest but minimal: it checks the model's own
    stop_reason and that real text came back, rather than judging content
    quality — content-aware verification (unsupported claims, citation
    checks, code test execution) belongs to the Research/Coding verifiers
    in later phases."""

    def verify(self, step: PlanStep, result: AssistantMessage) -> VerificationResult:
        if result.stop_reason == AiStopReason.ERROR:
            return VerificationResult(
                passed=False,
                score=0.0,
                failures=[result.error_message or "model returned an error"],
                recommended_repair="retry, folding the error message into the next attempt",
            )
        if result.stop_reason == AiStopReason.ABORTED:
            return VerificationResult(passed=False, score=0.0, failures=["run was aborted (request_stop)"])

        text = _final_text(result)
        if not text.strip():
            return VerificationResult(
                passed=False,
                score=0.2,
                failures=["no text content in the final response"],
                recommended_repair="ask the model to directly answer the objective in its final message",
            )
        return VerificationResult(passed=True, score=1.0)


class Replanner:
    """On a failed verification, inserts one concrete repair step right
    after the failed one. Bounded by the step's own attempt count — a
    step already retried once is not retried again automatically; that
    would be silent infinite repair, not resilience."""

    max_attempts_per_step = 2

    def replan(self, plan: Plan, step: PlanStep, verification: VerificationResult) -> bool:
        """Returns True if a repair step was inserted, False if this step
        has already used up its retries and the run should fail instead."""
        if step.attempts >= self.max_attempts_per_step:
            return False
        repair_action = step.action
        if verification.recommended_repair:
            reason = "; ".join(verification.failures) or "verification failed"
            repair_action = f"{step.action}\n\n(Previous attempt failed: {reason}. {verification.recommended_repair}.)"
        repair_step = PlanStep(
            objective=step.objective,
            action=repair_action,
            expected_outcome=step.expected_outcome,
            verification=step.verification,
            owner=step.owner,
            # Carry the attempt count forward: a repair step is a retry of
            # the same logical objective, not a fresh one. Without this, a
            # brand-new PlanStep resets attempts to 0 every time, and
            # max_attempts_per_step never actually triggers — the run
            # would replan forever instead of giving up.
            attempts=step.attempts,
        )
        plan.add_repair_step(repair_step)
        return True


class AgentRuntime:
    """Ties Planner/Executor/Verifier/Replanner together around one
    AgentSession, producing a single AgentState per run() call with an
    explicit stop_reason (Fase 1 acceptance criterion 5) — never returns
    without one, whether it finished, failed, or hit budget."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        executor: Executor | None = None,
        verifier: Verifier | None = None,
        replanner: Replanner | None = None,
    ) -> None:
        self._planner = planner or Planner()
        self._executor = executor or Executor()
        self._verifier = verifier or Verifier()
        self._replanner = replanner or Replanner()

    async def run(self, goal: Goal, session: AgentSession, *, telemetry: Any = None) -> AgentState:
        """*telemetry*, if given, is a pi_runtime.telemetry.
        TelemetryRecorder — kept untyped here (not imported at module
        level) to avoid a loop.py<->telemetry.py import cycle (telemetry
        module doesn't need anything from loop.py either, but keeping
        this a duck-typed optional avoids ever creating one). Fase 1's
        own callers (all of them, across every earlier phase) pass
        nothing and are unaffected."""
        state = AgentState(goal=goal, budget=goal.budget)
        state.status = RunStatus.RUNNING
        state.plan = self._planner.plan(goal)
        plan = state.plan

        while not plan.is_complete():
            exceeded = plan.goal.budget.exceeded()
            if exceeded is not None:
                state.status = RunStatus.STOPPED
                state.stop_reason = (
                    StopReason.MAX_ITERATIONS if exceeded == "max_iterations" else StopReason.BUDGET_EXCEEDED
                )
                state.error_message = f"budget exceeded: {exceeded}"
                break

            step = plan.current_step()
            assert step is not None
            state.iteration += 1
            state.budget.record_iteration()

            try:
                result = await self._executor.execute(step, session, state)
            except Exception as exc:
                state.status = RunStatus.FAILED
                state.stop_reason = StopReason.ERROR
                state.error_message = str(exc)
                break

            verification = self._verifier.verify(step, result)
            state.verification = verification

            if verification.passed:
                state.final_text = _final_text(result)
                state.decisions.append(f"step {step.id} completed")
                plan.advance()
                continue

            repaired = self._replanner.replan(plan, step, verification)
            if not repaired:
                state.status = RunStatus.FAILED
                state.stop_reason = StopReason.ERROR
                state.error_message = "; ".join(verification.failures) or "verification failed"
                break
            plan.advance()  # move past the failed step onto the repair step just inserted after it

        if state.stop_reason is None:
            state.status = RunStatus.DONE
            state.stop_reason = StopReason.COMPLETED

        state.finished_at = time.time()
        if telemetry is not None:
            telemetry.record_run(state)
        return state
