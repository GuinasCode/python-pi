"""Tests for the NVIDIA Nemotron/gpt-oss multi-model provider, including
its nvidia/auto fallback chain.

Uses httpx.MockTransport to simulate NVIDIA's SSE responses without a
real network call, matching test_nvidia_moonshot.py's pattern.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai import Context, Model, StopReason, TextContent, ThinkingContent, ToolCall, UserMessage
from pi_ai.providers.nvidia_models import _MODEL_SPECS, AUTO_MODEL_ID, nvidia_models_provider


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class TestNvidiaModelsProvider:
    def test_requires_api_key(self) -> None:
        import os

        env_key = os.environ.pop("NVAPI_KEY", None)
        try:
            with pytest.raises(ValueError):
                nvidia_models_provider(Model(id="test"))
        finally:
            if env_key is not None:
                os.environ["NVAPI_KEY"] = env_key

    def test_registers_every_model_plus_auto(self) -> None:
        _model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        ids = {m.id for m in models.get_models("nvidia")}
        for model_id, *_rest in _MODEL_SPECS:
            assert model_id in ids
        assert AUTO_MODEL_ID in ids
        assert len(ids) == len(_MODEL_SPECS) + 1

    def test_returned_model_is_the_auto_fallback_model(self) -> None:
        model, _models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        assert model.id == AUTO_MODEL_ID


class TestSingleModelStreaming:
    @pytest.mark.asyncio
    async def test_text_response_accumulates_and_ends_with_message(self) -> None:
        body = _sse(
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": ", world"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        _auto, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = _MODEL_SPECS[0][0]
        specific = next(m for m in models.get_models("nvidia") if m.id == target)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(specific, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.STOP
        assert result.model == target
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "Hello, world"

    @pytest.mark.asyncio
    async def test_reasoning_content_becomes_a_thinking_block_before_text(self) -> None:
        body = _sse(
            {"choices": [{"delta": {"reasoning_content": "let me think"}}]},
            {"choices": [{"delta": {"reasoning_content": "..."}}]},
            {"choices": [{"delta": {"content": "42"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        _auto, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = _MODEL_SPECS[0][0]
        specific = next(m for m in models.get_models("nvidia") if m.id == target)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(specific, Context(messages=[UserMessage(content="what is 6*7?")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert len(result.content) == 2
        thinking, text = result.content
        assert isinstance(thinking, ThinkingContent)
        assert thinking.thinking == "let me think..."
        assert isinstance(text, TextContent)
        assert text.text == "42"

    @pytest.mark.asyncio
    async def test_tool_call_response_does_not_crash_and_parses_arguments(self) -> None:
        body = _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.py"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        _auto, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = _MODEL_SPECS[0][0]
        specific = next(m for m in models.get_models("nvidia") if m.id == target)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(specific, Context(messages=[UserMessage(content="read a.py")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.TOOL_USE
        call = result.content[0]
        assert isinstance(call, ToolCall)
        assert call.name == "read_file"
        assert call.arguments == {"path": "a.py"}

    @pytest.mark.asyncio
    async def test_single_model_error_does_not_retry(self) -> None:
        """A specific (non-auto) model selection has no fallback chain —
        a failure is just reported, never silently retried on another
        model the user didn't ask for."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b"internal error")

        _auto, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = _MODEL_SPECS[0][0]
        specific = next(m for m in models.get_models("nvidia") if m.id == target)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(specific, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message and "500" in result.error_message
        assert result.model == target


class TestAutoFallbackChain:
    @pytest.mark.asyncio
    async def test_falls_through_to_the_next_model_on_a_non_200(self) -> None:
        first_id = _MODEL_SPECS[0][0]
        second_id = _MODEL_SPECS[1][0]
        good_body = _sse(
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if payload["model"] == first_id:
                return httpx.Response(503, content=b"unavailable")
            return httpx.Response(200, content=good_body, headers={"content-type": "text/event-stream"})

        model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        assert model.id == AUTO_MODEL_ID

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(model, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.STOP
        assert result.model == second_id
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "ok"

    @pytest.mark.asyncio
    async def test_reports_a_combined_error_when_every_model_fails(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=b"down")

        model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(model, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None
        for model_id, *_rest in _MODEL_SPECS:
            assert model_id in result.error_message

    @pytest.mark.asyncio
    async def test_does_not_retry_past_a_mid_stream_failure(self) -> None:
        """Once a model's response has already started streaming content
        back, a later failure must not silently fall through to another
        model — the caller may already be showing that partial text."""
        first_id = _MODEL_SPECS[0][0]
        # Malformed SSE stream: starts fine, then the connection is cut
        # off mid-response (no [DONE], body just ends) — httpx surfaces
        # this as a clean end of iteration, not an exception, so this
        # actually exercises the "ended without a finish_reason" path
        # rather than a raised error; either way it must stay on the
        # first model, not fall through.
        partial_body = b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=partial_body, headers={"content-type": "text/event-stream"})

        model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(model, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.model == first_id
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "partial"
