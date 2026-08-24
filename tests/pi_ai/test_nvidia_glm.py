"""Tests for the NVIDIA GLM 5.2 provider streaming logic.

Uses httpx.MockTransport to simulate the NVIDIA SSE response without a real
network call. No test file existed for this provider before — it's one of
the four that used to silently drop ImageContent from user messages
(`"".join(c.text if hasattr(c, "text") else "" for c in msg.content)`).
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai import Context, ImageContent, Model, StopReason, TextContent, ToolCall, UserMessage
from pi_ai.providers.nvidia_glm import nvidia_glm_provider


def _sse(*chunks: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@pytest.mark.asyncio
async def test_text_response_accumulates_and_ends_with_message() -> None:
    body = _sse(
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", world"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model, models, _meta = nvidia_glm_provider(Model(id="z-ai/glm-5.2"), api_key="test-key")

    import pi_ai.providers.nvidia_glm as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        stream = models.stream(model, Context(messages=[UserMessage(content="hi")]))
        result = await stream.result()
    finally:
        mod.httpx.AsyncClient = original_client

    assert result.stop_reason == StopReason.STOP
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Hello, world"


@pytest.mark.asyncio
async def test_tool_call_response_does_not_crash_and_parses_arguments() -> None:
    body = _sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "a.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model, models, _meta = nvidia_glm_provider(Model(id="z-ai/glm-5.2"), api_key="test-key")

    import pi_ai.providers.nvidia_glm as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        stream = models.stream(model, Context(messages=[UserMessage(content="read a.py")]))
        result = await stream.result()
    finally:
        mod.httpx.AsyncClient = original_client

    assert result.stop_reason == StopReason.TOOL_USE
    call = result.content[0]
    assert isinstance(call, ToolCall)
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}


@pytest.mark.asyncio
async def test_image_content_reaches_the_wire_as_an_image_url_data_uri() -> None:
    """Regression: this provider used to build user-message content with
    `"".join(c.text if hasattr(c, "text") else "" for c in msg.content)`,
    which silently turned an ImageContent block into an empty string."""
    body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model, models, _meta = nvidia_glm_provider(Model(id="z-ai/glm-5.2"), api_key="test-key")

    import pi_ai.providers.nvidia_glm as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        message = UserMessage(
            content=[
                TextContent(text="what is in this image?"),
                ImageContent(data="Zm9v", mime_type="image/png"),
            ]
        )
        await models.stream(model, Context(messages=[message])).result()
    finally:
        mod.httpx.AsyncClient = original_client

    sent_content = captured["messages"][0]["content"]  # type: ignore[index]
    assert sent_content == [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,Zm9v"}},
    ]


def test_missing_api_key_raises() -> None:
    import os

    old = os.environ.pop("NVAPI_KEY", None)
    try:
        with pytest.raises(ValueError):
            nvidia_glm_provider(Model(id="z-ai/glm-5.2"))
    finally:
        if old is not None:
            os.environ["NVAPI_KEY"] = old
