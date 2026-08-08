"""OpenAI Completions provider adapter.

Simplified Python port of packages/ai/src/api/openai-completions.ts.
Uses httpx for HTTP streaming, covering the core chat completions flow.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    Usage,
    UsageCost,
)
from pi_ai.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from pi_ai.utils import describe_exception

DEFAULT_API = "openai-completions"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _content_to_openai(
    content: str | list[TextContent | ImageContent],
) -> list[dict[str, Any]]:
    """Convert pi content blocks to OpenAI message content parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    return parts


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert pi messages to OpenAI chat completion format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            result.append({"role": "user", "content": _content_to_openai(msg.content)})  # type: ignore[arg-type]
        elif msg.role == "assistant":
            openai_msg: dict[str, Any] = {"role": "assistant"}
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    pass  # Thinking is not sent to OpenAI completions API
                elif block.type == "toolCall":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                        }
                    )
            if text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls
            result.append(openai_msg)
        elif msg.role == "toolResult":
            text_parts: list[str] = []
            for block in msg.content:
                if block.type == "text":
                    text_parts.append(block.text)
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": "\n".join(text_parts),
                }
            )
    return result


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi tools to OpenAI function tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _parse_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Parse an SSE chunk from the OpenAI streaming response."""
    return chunk


def _create_openai_provider(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Create a streaming connection to OpenAI and return an event stream."""
    stream = create_assistant_message_event_stream()

    api_key = None
    if options and options.api_key:
        api_key = options.api_key
    else:
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")

    base_url = model.base_url or DEFAULT_BASE_URL

    async def _run() -> None:
        from pi_ai import (
            DoneEvent,
            ErrorEvent,
            StartEvent,
            TextDeltaEvent,
            TextEndEvent,
            TextStartEvent,
            ToolCallEndEvent,
        )

        if httpx is None:
            msg = AssistantMessage(
                content=[],
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason=StopReason.ERROR,
                error_message="httpx not installed",
                timestamp=int(time.time() * 1000),
            )
            stream.push(ErrorEvent(reason="error", error=msg))
            stream.end(msg)
            return

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if model.headers:
            headers.update(model.headers)
        if options and options.headers:
            for k, v in options.headers.items():
                if v is not None:
                    headers[k] = v

        messages = _convert_messages(context.messages)
        if context.system_prompt:
            messages.insert(0, {"role": "system", "content": context.system_prompt})

        body: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if context.tools:
            body["tools"] = _convert_tools(context.tools)
        if options and options.temperature is not None:
            body["temperature"] = options.temperature
        if options and options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens

        try:
            async with (
                httpx.AsyncClient() as client,
                client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=options.timeout_ms / 1000 if options and options.timeout_ms else 600,
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = AssistantMessage(
                        content=[],
                        api=model.api,
                        provider=model.provider,
                        model=model.id,
                        stop_reason=StopReason.ERROR,
                        error_message=f"OpenAI API error {response.status_code}: {error_text.decode()}",
                        timestamp=int(time.time() * 1000),
                    )
                    stream.push(ErrorEvent(reason="error", error=error_msg))
                    stream.end(error_msg)
                    return

                partial = AssistantMessage(
                    content=[],
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    stop_reason=StopReason.PENDING,
                    timestamp=int(time.time() * 1000),
                )
                stream.push(StartEvent(partial=partial))

                content_index = 0
                text_buffer = ""
                tool_calls: dict[int, dict[str, Any]] = {}
                finish_reason = None
                usage_data: dict[str, Any] | None = None
                current_text_started = False

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage_data = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if delta.get("content"):
                        if not current_text_started:
                            partial.content.append(TextContent(text=""))
                            stream.push(TextStartEvent(content_index=content_index, partial=partial))
                            current_text_started = True
                        text_chunk = delta["content"]
                        text_buffer += text_chunk
                        partial.content[content_index].text += text_chunk
                        stream.push(TextDeltaEvent(content_index=content_index, delta=text_chunk, partial=partial))

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)
                            if tc_index not in tool_calls:
                                tool_calls[tc_index] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                                if tc.get("function"):
                                    tool_calls[tc_index]["name"] = tc["function"].get("name", "")
                            if tc.get("function", {}).get("arguments"):
                                tool_calls[tc_index]["arguments"] += tc["function"]["arguments"]

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                # Finalize text
                if current_text_started:
                    stream.push(TextEndEvent(content_index=content_index, content=text_buffer, partial=partial))
                    content_index += 1

                # Finalize tool calls
                for tc_idx in sorted(tool_calls):
                    tc_data = tool_calls[tc_idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_call = ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args)
                    partial.content.append(tool_call)
                    stream.push(ToolCallEndEvent(content_index=content_index, tool_call=tool_call, partial=partial))
                    content_index += 1

                # Determine stop reason
                if finish_reason == "tool_calls":
                    partial.stop_reason = StopReason.TOOL_USE
                elif finish_reason == "length":
                    partial.stop_reason = StopReason.LENGTH
                elif finish_reason == "stop" or finish_reason is None:
                    partial.stop_reason = StopReason.STOP
                else:
                    partial.stop_reason = StopReason.STOP

                # Set usage
                if usage_data:
                    partial.usage = Usage(
                        input=usage_data.get("prompt_tokens", 0),
                        output=usage_data.get("completion_tokens", 0),
                        cache_read=0,
                        cache_write=0,
                        total_tokens=usage_data.get("total_tokens", 0),
                        cost=UsageCost(),
                    )

                stream.push(DoneEvent(reason=partial.stop_reason.value, message=partial))  # type: ignore[arg-type]
                stream.end(partial)

        except Exception as exc:
            error_msg = AssistantMessage(
                content=[],
                api=model.api,
                provider=model.provider,
                model=model.id,
                stop_reason=StopReason.ERROR,
                error_message=describe_exception(exc),
                timestamp=int(time.time() * 1000),
            )
            stream.push(ErrorEvent(reason="error", error=error_msg))
            stream.end(error_msg)

    import asyncio

    _run_task = asyncio.ensure_future(_run())  # noqa: RUF006
    return stream


def openai_provider(
    *,
    model_id: str = "gpt-4o",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    context_window: int = 128000,
    max_tokens: int = 16384,
    cost_input: float = 2.50,
    cost_output: float = 10.00,
) -> tuple[Model, Callable[..., AssistantMessageEventStream]]:
    """Create an OpenAI model and stream function.

    Returns (model, stream_fn) that can be registered with MutableModels.
    """
    model = Model(
        id=model_id,
        name=model_name or model_id,
        api=DEFAULT_API,
        provider="openai",
        base_url=base_url,
        reasoning=False,
        input=["text", "image"],
        cost=__import__("pi_ai").ModelCost(input=cost_input, output=cost_output),
        context_window=context_window,
        max_tokens=max_tokens,
    )

    def stream_fn(
        m: Model,
        ctx: Context,
        opts: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        effective_opts = opts or SimpleStreamOptions()
        if api_key and not effective_opts.api_key:
            effective_opts.api_key = api_key
        return _create_openai_provider(m, ctx, effective_opts)

    return model, stream_fn


__all__ = [
    "openai_provider",
]
