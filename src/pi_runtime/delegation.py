"""Delegation Runtime — Fase 6 of the research-first-runtime plan.

Does not reimplement subagent process spawning, parallel execution, or
live orchestration — pi_coding_agent.subagent already has a real, tested
one (SubagentRegistry/spawn_subagent, bounded-concurrency parallel
execution, list/stop/steer via the REPL's /agents command). Reinventing
any of that here would violate Invariant A (single runtime, no parallel
agent loops with different rules) and Regra 1.1 (procure funcionalidade
equivalente antes de criar novo módulo).

What plan.md's Fase 6 asks for that the existing system doesn't have: a
*structured*, auditable request shape — objective/constraints/
curated_context/budget/allowed_tools/success_criteria (plan.md section 7's
exact rule for what a child receives) — instead of a bare task string,
plus per-delegation elapsed-time tracking. This module adds exactly that
on top of the existing subagent system; every actual spawn still goes
through pi_coding_agent.subagent.runner.spawn_subagent, unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from pi_coding_agent.subagent.agent_def import AgentDef
from pi_coding_agent.subagent.registry import SubagentRegistry, SubagentResult
from pi_coding_agent.subagent.runner import spawn_subagent
from pi_runtime.state import Budget

_DEFAULT_TIMEOUT_SECONDS = 300.0


@dataclass
class DelegationRequest:
    """plan.md section 7: exactly what a child receives — never the
    parent's full conversation by default. pi_coding_agent.subagent
    already enforces this structurally (each child is a fresh OS process
    given only a rendered task string, nothing else); this dataclass
    makes *what goes into that string* explicit and auditable instead of
    an unstructured blob assembled ad hoc at each call site."""

    objective: str
    constraints: list[str] = field(default_factory=list)
    curated_context: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    allowed_tools: list[str] | None = None
    agent_type: str = "researcher"
    model: str | None = None

    def render_task(self) -> str:
        """The actual text sent to the child — stored on the outcome
        too (DelegationOutcome.request), so "contexto recebido é
        auditável" (Fase 6 acceptance criterion 5) is literal: what was
        sent can be inspected before *and* after a delegation runs."""
        lines = [self.objective]
        if self.constraints:
            lines.append("Constraints:")
            lines.extend(f"- {c}" for c in self.constraints)
        if self.curated_context:
            lines.append("Context:")
            lines.extend(f"- {c}" for c in self.curated_context)
        if self.success_criteria:
            lines.append("Success criteria:")
            lines.extend(f"- {c}" for c in self.success_criteria)
        return "\n".join(lines)


@dataclass
class DelegationOutcome:
    request: DelegationRequest
    result: SubagentResult | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.result is not None and self.result.succeeded


class DelegationManager:
    """Thin layer over pi_coding_agent.subagent's real spawn/registry:
    the structured DelegationRequest contract, per-delegation elapsed-time
    tracking (Fase 6 acceptance criterion 4: "custo individual é
    rastreado" — wall-clock elapsed time; token/dollar cost tracking
    needs provider usage accounting that doesn't flow back from a child
    process today, registered as a TODO rather than faked), and result
    aggregation with failure isolation."""

    def __init__(self, *, registry: SubagentRegistry | None = None, cwd: str | None = None) -> None:
        self._registry = registry or SubagentRegistry()
        self._cwd = cwd

    async def delegate(self, request: DelegationRequest) -> DelegationOutcome:
        agent = AgentDef(
            name=request.agent_type,
            system_prompt=f"You are a {request.agent_type} subagent.",
            tools=request.allowed_tools or [],
            model=request.model,
        )
        timeout = (
            request.budget.deadline_ts - time.time()
            if request.budget.deadline_ts is not None
            else _DEFAULT_TIMEOUT_SECONDS
        )
        started = time.time()
        try:
            handle = await spawn_subagent(self._registry, agent, request.render_task(), timeout=timeout, cwd=self._cwd)
            result = await handle.wait()
            return DelegationOutcome(request=request, result=result, elapsed_seconds=time.time() - started)
        except Exception as exc:  # a failed delegation must never take down the caller (Fase 6 criterion 2)
            return DelegationOutcome(request=request, error=str(exc), elapsed_seconds=time.time() - started)

    async def delegate_parallel(self, requests: list[DelegationRequest]) -> list[DelegationOutcome]:
        """Fase 6 acceptance criteria 1+2: each delegate() call spawns
        its own OS process immediately (spawn_subagent returns as soon as
        the process starts, not when it finishes — see
        pi_coding_agent.subagent.runner), so gathering N of them together
        is genuinely concurrent, not sequential-looking-parallel; one
        request's exception (already caught inside delegate() itself)
        can never prevent the others from completing or being reported."""
        return list(await asyncio.gather(*(self.delegate(r) for r in requests)))

    def registry(self) -> SubagentRegistry:
        """The live registry backing every delegation this manager has
        made — the same kind of object interactive_mode's /agents command
        already knows how to list/stop/steer."""
        return self._registry


def aggregate_results(outcomes: list[DelegationOutcome]) -> str:
    """Fase 6 acceptance criterion 3: resultados podem ser agregados."""
    parts = []
    for outcome in outcomes:
        if outcome.succeeded:
            assert outcome.result is not None
            parts.append(f"[{outcome.request.agent_type}] {outcome.result.output}")
        else:
            reason = outcome.error or (outcome.result.output if outcome.result else "unknown failure")
            parts.append(f"[{outcome.request.agent_type}] FAILED: {reason}")
    return "\n\n---\n\n".join(parts)


__all__ = ["DelegationManager", "DelegationOutcome", "DelegationRequest", "aggregate_results"]
