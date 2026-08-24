"""Tests for the NVIDIA Nemotron/gpt-oss multi-model provider, including
its nvidia/auto fallback chain.

Uses httpx.MockTransport to simulate NVIDIA's SSE responses without a
real network call, matching test_nvidia_moonshot.py's pattern.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai import (
    Context,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.providers.nvidia_models import _MODEL_SPECS, AUTO_MODEL_ID, DEFAULT_MODEL_ID, nvidia_models_provider


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
        for spec in _MODEL_SPECS:
            assert spec.model_id in ids
        assert AUTO_MODEL_ID in ids
        assert len(ids) == len(_MODEL_SPECS) + 1

    def test_returned_model_is_the_configured_default(self) -> None:
        model, _models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        assert model.id == DEFAULT_MODEL_ID == "minimaxai/minimax-m3"

    def test_default_model_is_not_in_the_auto_fallback_chain(self) -> None:
        """MiniMax M3 is a deliberately different model family from the
        Nemotron/gpt-oss chain nvidia/auto walks — it's registered and
        directly selectable, just not part of that fallback sequence."""
        default_spec = next(spec for spec in _MODEL_SPECS if spec.model_id == DEFAULT_MODEL_ID)
        assert default_spec.in_fallback_chain is False

    def test_registers_the_two_vision_models(self) -> None:
        _model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        ids = {m.id for m in models.get_models("nvidia")}
        assert "google/gemma-4-31b-it" in ids
        assert "meta/llama-3.2-90b-vision-instruct" in ids

    def test_vision_models_declare_image_input_support(self) -> None:
        _model, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        by_id = {m.id: m for m in models.get_models("nvidia")}
        assert by_id["google/gemma-4-31b-it"].input == ["text", "image"]
        assert by_id["meta/llama-3.2-90b-vision-instruct"].input == ["text", "image"]
        assert by_id[DEFAULT_MODEL_ID].input == ["text"]

    def test_vision_models_are_not_in_the_auto_fallback_chain(self) -> None:
        for model_id in ("google/gemma-4-31b-it", "meta/llama-3.2-90b-vision-instruct"):
            spec = next(s for s in _MODEL_SPECS if s.model_id == model_id)
            assert spec.in_fallback_chain is False


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
        target = _MODEL_SPECS[0].model_id
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
        target = _MODEL_SPECS[0].model_id
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
        target = _MODEL_SPECS[0].model_id
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
        target = _MODEL_SPECS[0].model_id
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


class TestPerModelPayload:
    """Each model's own NVIDIA API example pins a specific top_p, and the
    Nemotron models need chat_template_kwargs.enable_thinking=True to
    stream reasoning_content at all — verify the actual request body sent
    matches, not just that a response comes back."""

    @pytest.mark.asyncio
    async def test_nemotron_sends_enable_thinking(self) -> None:
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        spec = next(s for s in _MODEL_SPECS if s.enable_thinking)
        target = next(m for m in models.get_models("nvidia") if m.id == spec.model_id)

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            await models.stream(target, Context(messages=[UserMessage(content="hi")])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert captured["top_p"] == spec.top_p
        assert captured["chat_template_kwargs"] == {"enable_thinking": True}
        assert captured["temperature"] == 1

    @pytest.mark.asyncio
    async def test_gpt_oss_does_not_send_enable_thinking(self) -> None:
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        spec = next(s for s in _MODEL_SPECS if s.model_id == "openai/gpt-oss-120b")
        target = next(m for m in models.get_models("nvidia") if m.id == spec.model_id)

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            await models.stream(target, Context(messages=[UserMessage(content="hi")])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert captured["top_p"] == 1.0
        assert "chat_template_kwargs" not in captured

    @pytest.mark.asyncio
    async def test_caller_supplied_temperature_overrides_the_default(self) -> None:
        from pi_ai import SimpleStreamOptions

        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = next(m for m in models.get_models("nvidia") if m.id == _MODEL_SPECS[0].model_id)

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            options = SimpleStreamOptions(temperature=0.2)
            await models.stream(target, Context(messages=[UserMessage(content="hi")]), options).result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert captured["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_image_content_becomes_an_image_url_data_uri(self) -> None:
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = next(m for m in models.get_models("nvidia") if m.id == "google/gemma-4-31b-it")

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            message = UserMessage(
                content=[
                    TextContent(text="What is in this image?"),
                    ImageContent(data="Zm9v", mime_type="image/jpeg"),
                ]
            )
            await models.stream(target, Context(messages=[message])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        sent_content = captured["messages"][0]["content"]  # type: ignore[index]
        assert sent_content == [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}},
        ]

    @pytest.mark.asyncio
    async def test_tool_result_with_image_adds_a_synthetic_user_message(self) -> None:
        """A ToolResultMessage carrying an image (e.g. a browser
        screenshot) used to be silently dropped — the tool role message
        can only carry text per the OpenAI-compatible wire format, so the
        image now rides along as a synthetic follow-up user message."""
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = next(m for m in models.get_models("nvidia") if m.id == "google/gemma-4-31b-it")

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            tool_result = ToolResultMessage(
                tool_call_id="call_1",
                tool_name="screenshot",
                content=[TextContent(text="captured"), ImageContent(data="abc", mime_type="image/png")],
            )
            await models.stream(target, Context(messages=[UserMessage(content="hi"), tool_result])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        sent_messages = captured["messages"]  # type: ignore[index]
        assert sent_messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "captured"}
        assert sent_messages[2] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
        }

    @pytest.mark.asyncio
    async def test_llama_vision_sends_its_penalty_params(self) -> None:
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = next(m for m in models.get_models("nvidia") if m.id == "meta/llama-3.2-90b-vision-instruct")

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            await models.stream(target, Context(messages=[UserMessage(content="hi")])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert captured["frequency_penalty"] == 0
        assert captured["presence_penalty"] == 0
        assert captured["top_p"] == 1.0
        assert "chat_template_kwargs" not in captured

    @pytest.mark.asyncio
    async def test_other_models_do_not_send_penalty_params(self) -> None:
        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        target = next(m for m in models.get_models("nvidia") if m.id == _MODEL_SPECS[0].model_id)

        body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            await models.stream(target, Context(messages=[UserMessage(content="hi")])).result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert "frequency_penalty" not in captured
        assert "presence_penalty" not in captured


class TestAutoFallbackChain:
    @pytest.mark.asyncio
    async def test_falls_through_to_the_next_model_on_a_non_200(self) -> None:
        first_id = _MODEL_SPECS[0].model_id
        second_id = _MODEL_SPECS[1].model_id
        good_body = _sse(
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if payload["model"] == first_id:
                return httpx.Response(503, content=b"unavailable")
            return httpx.Response(200, content=good_body, headers={"content-type": "text/event-stream"})

        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        auto = next(m for m in models.get_models("nvidia") if m.id == AUTO_MODEL_ID)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(auto, Context(messages=[UserMessage(content="hi")]))
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

        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        auto = next(m for m in models.get_models("nvidia") if m.id == AUTO_MODEL_ID)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(auto, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.stop_reason == StopReason.ERROR
        assert result.error_message is not None
        for spec in _MODEL_SPECS:
            if spec.in_fallback_chain:
                assert spec.model_id in result.error_message
        # MiniMax isn't part of the chain — it never gets attempted, so it
        # must not appear in the combined failure summary.
        assert DEFAULT_MODEL_ID not in result.error_message

    @pytest.mark.asyncio
    async def test_does_not_retry_past_a_mid_stream_failure(self) -> None:
        """Once a model's response has already started streaming content
        back, a later failure must not silently fall through to another
        model — the caller may already be showing that partial text."""
        first_id = _MODEL_SPECS[0].model_id
        # Malformed SSE stream: starts fine, then the connection is cut
        # off mid-response (no [DONE], body just ends) — httpx surfaces
        # this as a clean end of iteration, not an exception, so this
        # actually exercises the "ended without a finish_reason" path
        # rather than a raised error; either way it must stay on the
        # first model, not fall through.
        partial_body = b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=partial_body, headers={"content-type": "text/event-stream"})

        _default, models, _meta = nvidia_models_provider(Model(id="test"), api_key="test-key")
        auto = next(m for m in models.get_models("nvidia") if m.id == AUTO_MODEL_ID)

        import pi_ai.providers.nvidia_models as mod

        original_client = httpx.AsyncClient
        mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        try:
            stream = models.stream(auto, Context(messages=[UserMessage(content="hi")]))
            result = await stream.result()
        finally:
            mod.httpx.AsyncClient = original_client

        assert result.model == first_id
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "partial"
