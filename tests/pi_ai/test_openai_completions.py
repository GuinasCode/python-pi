"""Tests for pi_ai.providers.openai — no test file existed for this
provider before (it was the only one whose image conversion, correct
since it was first written, had never been exercised by CI).

Uses httpx.MockTransport to simulate the OpenAI SSE response without a
real network call.
"""

from __future__ import annotations

import json

import httpx
import pytest

from pi_ai import (
    Context,
    ImageContent,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.providers.openai import openai_provider


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

    model, stream_fn = openai_provider(api_key="test-key")

    import pi_ai.providers.openai as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        stream = stream_fn(model, Context(messages=[UserMessage(content="hi")]))
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

    model, stream_fn = openai_provider(api_key="test-key")

    import pi_ai.providers.openai as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        stream = stream_fn(model, Context(messages=[UserMessage(content="read a.py")]))
        result = await stream.result()
    finally:
        mod.httpx.AsyncClient = original_client

    assert result.stop_reason == StopReason.TOOL_USE
    call = result.content[0]
    assert isinstance(call, ToolCall)
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}


@pytest.mark.asyncio
async def test_image_content_becomes_an_image_url_data_uri() -> None:
    body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model, stream_fn = openai_provider(api_key="test-key")

    import pi_ai.providers.openai as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        message = UserMessage(
            content=[
                TextContent(text="what is in this image?"),
                ImageContent(data="Zm9v", mime_type="image/jpeg"),
            ]
        )
        await stream_fn(model, Context(messages=[message])).result()
    finally:
        mod.httpx.AsyncClient = original_client

    sent_content = captured["messages"][0]["content"]  # type: ignore[index]
    assert sent_content == [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Zm9v"}},
    ]


@pytest.mark.asyncio
async def test_tool_result_with_image_adds_a_synthetic_user_message() -> None:
    """Gap fix: a ToolResultMessage carrying an ImageContent block (e.g. a
    browser screenshot) used to be silently dropped everywhere — the tool
    role message can only carry text per the OpenAI wire format, so the
    image now rides along as a synthetic follow-up user message instead of
    disappearing."""
    body = _sse({"choices": [{"delta": {"content": "ok"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]})
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    model, stream_fn = openai_provider(api_key="test-key")

    import pi_ai.providers.openai as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        tool_result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="screenshot",
            content=[TextContent(text="captured"), ImageContent(data="abc", mime_type="image/png")],
        )
        await stream_fn(model, Context(messages=[UserMessage(content="hi"), tool_result])).result()
    finally:
        mod.httpx.AsyncClient = original_client

    sent_messages = captured["messages"]  # type: ignore[index]
    assert sent_messages[1] == {"role": "tool", "tool_call_id": "call_1", "content": "captured"}
    assert sent_messages[2] == {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
    }


@pytest.mark.asyncio
async def test_api_error_produces_error_message_not_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    model, stream_fn = openai_provider(api_key="test-key")

    import pi_ai.providers.openai as mod

    original_client = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
    try:
        stream = stream_fn(model, Context(messages=[UserMessage(content="hi")]), SimpleStreamOptions())
        result = await stream.result()
    finally:
        mod.httpx.AsyncClient = original_client

    assert result.stop_reason == StopReason.ERROR
