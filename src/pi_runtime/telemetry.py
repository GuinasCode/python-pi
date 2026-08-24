"""Telemetry + Observability — Fase 15 of the research-first-runtime plan.

Real consumer: TelemetryRecorder attaches to AgentSession.on_event()
(the existing, already-tested event bus — pi_coding_agent.agent_session's
_emit()/on_event(), unchanged, not a second event system) to record real
tool-call spans as they happen, and pi_runtime.loop.AgentRuntime accepts
an optional `telemetry` parameter so every run() call gets one trace with
stable ids (run_id, turn/tool_call ids from the real events) covering the
whole goal->plan->act->verify loop, not just the raw provider stream.

Cost is computed with pi_runtime.router.estimate_cost (Fase 10, unchanged)
against whatever Model the caller attributes a span to — no second cost
model.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pi_ai import Model
from pi_coding_agent.agent_session import AgentSession
from pi_runtime.router import estimate_cost
from pi_runtime.state import AgentState


@dataclass
class Span:
    """plan.md section 19: stable ids so a run can be reconstructed
    after the fact. `trace_id` ties every span in one run together;
    `span_id` is this span's own id; `parent_id` links it to whatever
    span (if any) it happened inside."""

    span_id: str
    trace_id: str
    name: str
    started_at: float
    parent_id: str | None = None
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def finish(self, *, error: str | None = None, **attributes: Any) -> None:
        self.ended_at = time.time()
        self.error = error
        self.attributes.update(attributes)


@dataclass
class CostRecord:
    trace_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost: float


class Trace:
    """One run's worth of spans + cost records — everything needed to
    answer plan.md's acceptance criterion questions (quanto custou,
    quanto tempo levou, quais tools usou, quais agentes rodaram, onde
    falhou, por que terminou) for this one run_id."""

    def __init__(self, trace_id: str | None = None, *, run_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.run_id = run_id or self.trace_id
        self.spans: list[Span] = []
        self.costs: list[CostRecord] = []
        self.stop_reason: str | None = None

    def start_span(self, name: str, *, parent_id: str | None = None, **attributes: Any) -> Span:
        span = Span(
            span_id=uuid.uuid4().hex[:10],
            trace_id=self.trace_id,
            name=name,
            started_at=time.time(),
            parent_id=parent_id,
            attributes=attributes,
        )
        self.spans.append(span)
        return span

    def record_cost(self, model: Model, *, input_tokens: int, output_tokens: int) -> CostRecord:
        record = CostRecord(
            trace_id=self.trace_id,
            model_id=model.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimate_cost(model, input_tokens=input_tokens, output_tokens=output_tokens),
        )
        self.costs.append(record)
        return record

    def total_cost(self) -> float:
        return sum(record.cost for record in self.costs)

    def total_duration_seconds(self) -> float:
        finished = [s for s in self.spans if s.duration_seconds is not None]
        if not finished:
            return 0.0
        start = min(s.started_at for s in self.spans)
        end = max(s.ended_at for s in finished if s.ended_at is not None)
        return end - start

    def tool_names_used(self) -> list[str]:
        return sorted(
            {s.attributes["tool_name"] for s in self.spans if s.name == "tool_call" and "tool_name" in s.attributes}
        )

    def agent_ids_used(self) -> list[str]:
        return sorted({s.attributes["agent_id"] for s in self.spans if "agent_id" in s.attributes})

    def failures(self) -> list[str]:
        return [f"{s.name}: {s.error}" for s in self.spans if s.error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
            "total_cost": self.total_cost(),
            "total_duration_seconds": self.total_duration_seconds(),
            "tool_names_used": self.tool_names_used(),
            "agent_ids_used": self.agent_ids_used(),
            "failures": self.failures(),
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "name": s.name,
                    "started_at": s.started_at,
                    "ended_at": s.ended_at,
                    "duration_seconds": s.duration_seconds,
                    "attributes": s.attributes,
                    "error": s.error,
                }
                for s in self.spans
            ],
            "costs": [
                {
                    "model_id": c.model_id,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost": c.cost,
                }
                for c in self.costs
            ],
        }

    def to_json(self) -> str:
        """plan.md section 19: "JSON event export"."""
        return json.dumps(self.to_dict())


class TelemetryRecorder:
    """Attaches to a real AgentSession's existing event bus
    (on_event/_emit, unchanged) to record tool-call spans as they
    actually happen — turn_start/turn_end and tool_call_start/
    tool_call_end are the same event types pi_coding_agent.
    interactive_mode._handle_event already consumes for its own
    rendering, not a second event vocabulary invented here."""

    def __init__(self, trace: Trace) -> None:
        self._trace = trace
        self._open_tool_spans: dict[str, Span] = {}

    def attach(self, session: AgentSession) -> Any:
        """Returns the unsubscribe callable AgentSession.on_event()
        already provides — the caller is responsible for detaching when
        done, same as any other on_event listener in this codebase."""
        return session.on_event(self._on_event)

    def _on_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        if event_type == "tool_call_start":
            name = getattr(event, "name", "")
            span = self._trace.start_span("tool_call", tool_name=name)
            self._open_tool_spans[name] = span
        elif event_type == "tool_call_end":
            name = getattr(event, "name", "")
            open_span = self._open_tool_spans.get(name)
            if open_span is not None:
                del self._open_tool_spans[name]
                is_error = getattr(event, "is_error", False)
                open_span.finish(error=(getattr(event, "result_text", "") if is_error else None))
        elif event_type == "subagent_progress":
            # A subagent ran under this session but has no per-call
            # start/end event today (pi_coding_agent.subagent.registry
            # tracks its own lifecycle separately, see Fase 6's
            # DelegationOutcome) — record it as an attributed marker span
            # rather than inventing a start/end pair this event stream
            # doesn't actually provide.
            self._trace.start_span("subagent_progress", agent_id="subagent").finish()

    def record_run(self, state: AgentState) -> Span:
        """Records one span representing an entire AgentRuntime.run()
        call, with the stable run_id as its trace tie-in and the final
        stop_reason attached — the top-level entry a caller reconstructs
        "por que terminou" from."""
        span = self._trace.start_span("agent_run", run_id=state.run_id)
        span.finish(
            error=state.error_message,
            status=state.status.value,
            stop_reason=state.stop_reason.value if state.stop_reason else None,
            iterations=state.iteration,
        )
        self._trace.stop_reason = state.stop_reason.value if state.stop_reason else None
        return span


__all__ = ["CostRecord", "Span", "TelemetryRecorder", "Trace"]
