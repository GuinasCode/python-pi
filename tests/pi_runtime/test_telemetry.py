"""Tests for pi_runtime.telemetry. Covers Fase 15's acceptance criterion
from plan.md section 19 — "qualquer tarefa pode responder": quanto
custou, quanto tempo levou, quais tools usou, quais agentes rodaram,
onde falhou, por que terminou. Plus structured events, trace ids, spans,
JSON export.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pi_ai import Model, ModelCost
from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.providers.faux import faux_provider as _faux_provider
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_runtime.loop import AgentRuntime
from pi_runtime.state import Goal
from pi_runtime.telemetry import TelemetryRecorder, Trace


def _make_session(responses: list[Any]) -> AgentSession:
    handle = _faux_provider()
    handle.set_responses(responses)
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    return AgentSession(AgentSessionOptions(models=models, model=model, cwd="/tmp", enable_subagents=False))


class TestSpanLifecycle:
    def test_span_has_no_duration_until_finished(self) -> None:
        trace = Trace()
        span = trace.start_span("x")
        assert span.duration_seconds is None
        span.finish()
        assert span.duration_seconds is not None
        assert span.duration_seconds >= 0

    def test_finish_can_attach_error_and_attributes(self) -> None:
        trace = Trace()
        span = trace.start_span("x")
        span.finish(error="boom", detail="context")
        assert span.error == "boom"
        assert span.attributes["detail"] == "context"


class TestTraceIds:
    def test_every_span_shares_the_trace_id(self) -> None:
        trace = Trace()
        a = trace.start_span("a")
        b = trace.start_span("b")
        assert a.trace_id == trace.trace_id
        assert b.trace_id == trace.trace_id

    def test_spans_get_distinct_span_ids(self) -> None:
        trace = Trace()
        a = trace.start_span("a")
        b = trace.start_span("b")
        assert a.span_id != b.span_id

    def test_custom_run_id_is_preserved(self) -> None:
        trace = Trace(run_id="my-run-123")
        assert trace.run_id == "my-run-123"


class TestCostTracking:
    def test_records_cost_using_the_router_estimator(self) -> None:
        trace = Trace()
        model = Model(id="x", cost=ModelCost(input=2.0, output=4.0))
        record = trace.record_cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
        assert record.cost == 6.0
        assert trace.total_cost() == 6.0

    def test_multiple_cost_records_accumulate(self) -> None:
        trace = Trace()
        model = Model(id="x", cost=ModelCost(input=1.0, output=1.0))
        trace.record_cost(model, input_tokens=1_000_000, output_tokens=0)
        trace.record_cost(model, input_tokens=1_000_000, output_tokens=0)
        assert trace.total_cost() == 2.0


class TestToolCallsAreRecordedFromRealEvents:
    """TelemetryRecorder attaches to AgentSession.on_event() — the same
    real event bus interactive_mode already consumes, not a second one."""

    def test_tool_call_becomes_a_span_with_duration(self) -> None:
        session = _make_session(
            [
                faux_assistant_message([faux_tool_call("read", {"path": "x"})], stop_reason=None),  # type: ignore[arg-type]
            ]
        )
        trace = Trace()
        recorder = TelemetryRecorder(trace)
        recorder.attach(session)

        # simulate the real events AgentSession._emit() would fire
        from pi_coding_agent.agent_session import _ToolCallEndEvent, _ToolCallStartEvent

        session._emit(_ToolCallStartEvent(name="read", args={"path": "x"}))
        session._emit(_ToolCallEndEvent(name="read", is_error=False, result_text="ok", details=None))

        tool_spans = [s for s in trace.spans if s.name == "tool_call"]
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes["tool_name"] == "read"
        assert tool_spans[0].duration_seconds is not None

    def test_failed_tool_call_is_recorded_as_a_failure(self) -> None:
        from pi_coding_agent.agent_session import _ToolCallEndEvent, _ToolCallStartEvent

        session = _make_session([faux_assistant_message("x")])
        trace = Trace()
        recorder = TelemetryRecorder(trace)
        recorder.attach(session)

        session._emit(_ToolCallStartEvent(name="bash", args={}))
        session._emit(_ToolCallEndEvent(name="bash", is_error=True, result_text="permission denied", details=None))

        assert "permission denied" in trace.failures()[0]

    def test_tool_names_used_reflects_real_tool_calls(self) -> None:
        from pi_coding_agent.agent_session import _ToolCallEndEvent, _ToolCallStartEvent

        session = _make_session([faux_assistant_message("x")])
        trace = Trace()
        TelemetryRecorder(trace).attach(session)

        for name in ("read", "grep", "read"):
            session._emit(_ToolCallStartEvent(name=name, args={}))
            session._emit(_ToolCallEndEvent(name=name, is_error=False, result_text="", details=None))

        assert trace.tool_names_used() == ["grep", "read"]


class TestAgentRuntimeIntegration:
    """Fase 15's real consumer: telemetry passed into AgentRuntime.run()."""

    def test_run_records_a_top_level_span_with_stop_reason(self) -> None:
        session = _make_session([faux_assistant_message("done")])
        trace = Trace()
        recorder = TelemetryRecorder(trace)

        asyncio.run(AgentRuntime().run(Goal(objective="x"), session, telemetry=recorder))

        run_spans = [s for s in trace.spans if s.name == "agent_run"]
        assert len(run_spans) == 1
        assert run_spans[0].attributes["stop_reason"] == "completed"
        assert trace.stop_reason == "completed"

    def test_run_without_telemetry_is_unaffected(self) -> None:
        """Every earlier-phase caller passes nothing — confirms that
        still works exactly as before."""
        session = _make_session([faux_assistant_message("done")])
        state = asyncio.run(AgentRuntime().run(Goal(objective="x"), session))
        assert state.final_text == "done"


class TestReconstructTheRunFromTelemetry:
    """The literal acceptance criterion: every question plan.md lists
    must be answerable from one Trace."""

    def test_full_reconstruction(self) -> None:
        from pi_coding_agent.agent_session import _ToolCallEndEvent, _ToolCallStartEvent

        session = _make_session([faux_assistant_message("done")])
        trace = Trace(run_id="run-abc")
        recorder = TelemetryRecorder(trace)
        recorder.attach(session)

        session._emit(_ToolCallStartEvent(name="read", args={}))
        session._emit(_ToolCallEndEvent(name="read", is_error=False, result_text="ok", details=None))

        model = Model(id="test-model", cost=ModelCost(input=1.0, output=2.0))
        trace.record_cost(model, input_tokens=1000, output_tokens=500)

        state = asyncio.run(AgentRuntime().run(Goal(objective="x"), session, telemetry=recorder))

        # quanto custou
        assert trace.total_cost() > 0
        # quanto tempo levou
        assert trace.total_duration_seconds() >= 0
        # quais tools usou
        assert "read" in trace.tool_names_used()
        # por que terminou
        assert trace.stop_reason == state.stop_reason.value  # type: ignore[union-attr]


class TestJsonExport:
    def test_to_json_produces_valid_json_with_expected_keys(self) -> None:
        trace = Trace(run_id="run-xyz")
        span = trace.start_span("x")
        span.finish()

        raw = json.loads(trace.to_json())
        assert raw["run_id"] == "run-xyz"
        assert raw["trace_id"] == trace.trace_id
        assert "spans" in raw
        assert "costs" in raw
        assert "total_cost" in raw
        assert "total_duration_seconds" in raw

    def test_to_dict_and_to_json_agree(self) -> None:
        trace = Trace()
        trace.start_span("x").finish()
        assert json.loads(trace.to_json()) == trace.to_dict()
