"""Tests for faux provider and event stream."""

from __future__ import annotations

import asyncio

from pi_ai import Context, StopReason, UserMessage
from pi_ai.event_stream import create_assistant_message_event_stream
from pi_ai.models import MutableModels
from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


class TestEventStream:
    def test_push_and_consume(self) -> None:
        async def run_test() -> None:
            from pi_ai import DoneEvent

            stream = create_assistant_message_event_stream()
            msg = faux_assistant_message("hello")
            stream.push(DoneEvent(reason="stop", message=msg))
            stream.end(msg)

            events: list[object] = []
            async for event in stream:
                events.append(event)
            assert len(events) >= 1

        asyncio.run(run_test())


class TestFauxProvider:
    def test_provider_creation(self) -> None:
        handle = faux_provider()
        assert handle.provider is not None
        assert len(handle.models) >= 1
        assert handle.models[0].id == "faux-1"

    def test_get_model(self) -> None:
        handle = faux_provider()
        model = handle.get_model()
        assert model is not None
        assert model.id == "faux-1"
        assert handle.get_model("nonexistent") is None

    def test_set_responses(self) -> None:
        handle = faux_provider()
        handle.set_responses([faux_assistant_message("hello")])
        assert handle.pending_response_count == 1

    def test_append_responses(self) -> None:
        handle = faux_provider()
        handle.set_responses([faux_assistant_message("a")])
        handle.append_responses([faux_assistant_message("b")])
        assert handle.pending_response_count == 2

    def test_state_call_count(self) -> None:
        handle = faux_provider()
        assert handle.state["callCount"] == 0


class TestFauxHelpers:
    def test_faux_text(self) -> None:
        tc = faux_text("hello")
        assert tc.type == "text"
        assert tc.text == "hello"

    def test_faux_thinking(self) -> None:
        tc = faux_thinking("reasoning")
        assert tc.type == "thinking"
        assert tc.thinking == "reasoning"

    def test_faux_tool_call(self) -> None:
        tc = faux_tool_call("read", {"path": "/tmp"})
        assert tc.type == "toolCall"
        assert tc.name == "read"
        assert tc.arguments == {"path": "/tmp"}

    def test_faux_assistant_message_string(self) -> None:
        msg = faux_assistant_message("hello")
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "hello"
        assert msg.stop_reason == StopReason.STOP

    def test_faux_assistant_message_with_tool_call(self) -> None:
        tc = faux_tool_call("read", {"path": "/tmp"})
        msg = faux_assistant_message([tc], stop_reason=StopReason.TOOL_USE)
        assert msg.stop_reason == StopReason.TOOL_USE
        assert len(msg.content) == 1
        assert msg.content[0].type == "toolCall"

    def test_faux_assistant_message_error(self) -> None:
        msg = faux_assistant_message("error", stop_reason=StopReason.ERROR, error_message="failed")
        assert msg.stop_reason == StopReason.ERROR
        assert msg.error_message == "failed"


class TestFauxProviderIntegration:
    def test_register_with_models(self) -> None:
        handle = faux_provider()
        models = MutableModels()
        models.set_provider(handle.provider)
        assert len(models.get_models()) >= 1
        assert models.get_model("faux", "faux-1") is not None

    def test_stream_produces_events(self) -> None:
        handle = faux_provider()
        handle.set_responses([faux_assistant_message("Hello from faux!")])
        models = MutableModels()
        models.set_provider(handle.provider)

        model = handle.get_model()
        assert model is not None

        context = Context(
            system_prompt="You are helpful",
            messages=[UserMessage(content="hi", timestamp=0)],
        )

        async def run_test() -> None:
            stream = models.stream(model, context)
            events: list[object] = []
            async for event in stream:
                events.append(event)
            event_types = [getattr(e, "type", "") for e in events]
            assert "done" in event_types or "error" in event_types

        asyncio.run(asyncio.wait_for(run_test(), timeout=10.0))

    def test_stream_no_responses_returns_error(self) -> None:
        handle = faux_provider()
        models = MutableModels()
        models.set_provider(handle.provider)

        model = handle.get_model()
        assert model is not None

        context = Context(
            messages=[UserMessage(content="hi", timestamp=0)],
        )

        async def run_test() -> None:
            stream = models.stream(model, context)
            events: list[object] = []
            async for event in stream:
                events.append(event)
            event_types = [getattr(e, "type", "") for e in events]
            assert "error" in event_types

        asyncio.run(asyncio.wait_for(run_test(), timeout=10.0))

    def test_stream_text_deltas(self) -> None:
        handle = faux_provider(tokens_per_second=None)
        handle.set_responses([faux_assistant_message("hello world")])
        models = MutableModels()
        models.set_provider(handle.provider)

        model = handle.get_model()
        assert model is not None

        context = Context(messages=[UserMessage(content="hi", timestamp=0)])

        async def run_test() -> None:
            stream = models.stream(model, context)
            events: list[object] = []
            async for event in stream:
                events.append(event)
            event_types = [getattr(e, "type", "") for e in events]
            assert "start" in event_types
            assert "text_start" in event_types
            assert "text_delta" in event_types
            assert "text_end" in event_types
            assert "done" in event_types

        asyncio.run(asyncio.wait_for(run_test(), timeout=10.0))
