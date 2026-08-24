"""Core runtime contracts: Budget, Goal, Plan, AgentState, VerificationResult.

See plan.md section 3 ("Contratos centrais") — these are dataclasses, not
services: no I/O, no provider calls, fully serializable (every field is a
plain type, an enum, or another dataclass here), so AgentState can be
persisted and rebuilt without depending on any live object (Invariant C:
"estado recuperável sem depender de objetos efêmeros").
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    """Why an AgentRuntime.run() call actually stopped. Always set on
    completion — Fase 1 acceptance criterion 5 ("toda execução tem stop
    reason explícito")."""

    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    USER_STOP = "user_stop"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Budget:
    """Real, enforced limits — not metadata (plan.md Invariant F). Every
    `max_*` field is optional (None = unlimited); `consumed_*` fields are
    mutated in place by AgentRuntime as it runs."""

    max_tokens: int | None = None
    max_cost: float | None = None
    max_iterations: int | None = 10
    max_tool_calls: int | None = None
    max_subagents: int | None = None
    deadline_ts: float | None = None

    consumed_tokens: int = 0
    consumed_cost: float = 0.0
    consumed_iterations: int = 0
    consumed_tool_calls: int = 0
    consumed_subagents: int = 0

    def record_iteration(self) -> None:
        self.consumed_iterations += 1

    def record_usage(self, *, tokens: int = 0, cost: float = 0.0, tool_calls: int = 0) -> None:
        self.consumed_tokens += tokens
        self.consumed_cost += cost
        self.consumed_tool_calls += tool_calls

    def exceeded(self) -> str | None:
        """Return which limit was hit (a short machine-readable reason),
        or None if every tracked limit is still within bounds. Checked at
        each loop iteration boundary in AgentRuntime — never mid-tool-call,
        the same "next boundary" convention as AgentSession's steer/stop
        hooks (pi_coding_agent.agent_session.queue_steer_message/
        request_stop)."""
        if self.max_iterations is not None and self.consumed_iterations >= self.max_iterations:
            return "max_iterations"
        if self.max_tokens is not None and self.consumed_tokens >= self.max_tokens:
            return "max_tokens"
        if self.max_cost is not None and self.consumed_cost >= self.max_cost:
            return "max_cost"
        if self.max_tool_calls is not None and self.consumed_tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.deadline_ts is not None and time.time() >= self.deadline_ts:
            return "deadline"
        return None


@dataclass
class Goal:
    """What the runtime is trying to achieve for one run."""

    objective: str
    context: str = ""
    success_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: int = 0
    deadline_ts: float | None = None
    budget: Budget = field(default_factory=Budget)


@dataclass
class PlanStep:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    objective: str = ""
    preconditions: list[str] = field(default_factory=list)
    action: str = ""
    dependencies: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    verification: str = ""
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    owner: str = "principal"  # "principal" | "subagent"


@dataclass
class Plan:
    """A plan is created, updated, invalidated, partially completed, and
    replanned (plan.md 3.3) — this is the mutable record of that, not a
    fixed, one-shot list."""

    goal: Goal
    steps: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0

    def current_step(self) -> PlanStep | None:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> None:
        self.current_step_index += 1

    def add_repair_step(self, step: PlanStep) -> None:
        """Insert a new step right after the current one — this is what a
        Replanner does on a failed verification: not restart from
        scratch, not silently give up, but extend the plan with a
        concrete repair action."""
        self.steps.insert(self.current_step_index + 1, step)

    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)


@dataclass
class VerificationResult:
    """plan.md 3.6 / section 10 — never accept "parece que funcionou" as
    proof; this is what Verifier.verify() must produce instead."""

    passed: bool
    score: float = 0.0
    failures: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    recommended_repair: str | None = None


@dataclass
class AgentState:
    """Single source of truth for one AgentRuntime.run() call (Invariant
    C). Every field here is a plain/serializable type — no live objects,
    no callbacks — so this can be dumped to JSON and rebuilt later
    (Sessions/Replay, Fase 12, builds on this rather than replacing it)."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: Goal | None = None
    plan: Plan | None = None
    working_memory: list[str] = field(default_factory=list)
    active_tasks: list[str] = field(default_factory=list)
    child_agent_handles: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    pending_actions: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    iteration: int = 0
    status: RunStatus = RunStatus.PENDING
    stop_reason: StopReason | None = None
    verification: VerificationResult | None = None
    final_text: str | None = None
    error_message: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
