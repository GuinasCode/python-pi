"""Explicit-state agent runtime — Fase 1 of the research-first-runtime plan.

Wraps the existing, working AgentSession/agent loop (pi_coding_agent.
agent_session / pi_agent_core.agent_loop) with an explicit AgentState/Goal/
Plan/Budget/Verifier layer, instead of replacing it. AgentSession.prompt()
keeps doing exactly what it does today (streaming, tool calling, steering,
memory, soul, subagents) — AgentRuntime adds a state machine on top that
gives every run an observable plan, a tracked budget, and an explicit stop
reason, and can repair/replan on a failed step.

Not yet wired into the CLI/TUI (that's Fase 17 — "CLI/UX/APIs, somente
depois do runtime estar estável") — this package is tested standalone
against AgentSession first.
"""

from __future__ import annotations

from pi_runtime.loop import AgentRuntime, Executor, Planner, Replanner, Verifier
from pi_runtime.state import (
    AgentState,
    Budget,
    Goal,
    Plan,
    PlanStep,
    RunStatus,
    StopReason,
    VerificationResult,
)

__all__ = [
    "AgentRuntime",
    "AgentState",
    "Budget",
    "Executor",
    "Goal",
    "Plan",
    "PlanStep",
    "Planner",
    "Replanner",
    "RunStatus",
    "StopReason",
    "VerificationResult",
    "Verifier",
]
