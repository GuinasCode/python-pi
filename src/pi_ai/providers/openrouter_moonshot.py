"""Moonshot/Kimi K2.6 provider via OpenRouter.

The Kimi K2.6 model is available on OpenRouter with no API key required
for basic usage. This provider uses OpenRouter as the backend.

Usage:
    from pi_ai.providers.openrouter_moonshot import openrouter_moonshot_provider
    from pi_ai import Model

    model, models, meta = openrouter_moonshot_provider(Model(id="test"))
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, cast

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from pi_ai import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    StartEvent,
    StopReason,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from pi_ai.models import MutableModels, Provider
from pi_ai.providers._openai_compat import tool_result_to_openai_messages, user_content_to_openai
from pi_ai.utils import describe_exception, spawn_background_task

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 16384


def openrouter_moonshot_provider(
    model: Model,
    api_key: str | None = None,
) -> tuple[Model, MutableModels, dict[str, Any]]:
    """
    Create a models collection configured for Moonshot Kimi K2.6 via OpenRouter.

    Returns: (model, models_collection, empty_metadata_dict)
    """
    import os

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY environment variable or api_key parameter required")

    models = MutableModels()

    kimi_model = Model(
        id="moonshotai/kimi-k2.6",
        name="Kimi K2.6",
        api="openai",
        provider="openrouter",
        context_window=16384,
        max_tokens=16384,
    )

    def stream_fn(
        m: Model,
        ctx: Context,
        options: Any | None = None,
    ) -> AssistantMessageEventStream:
        return _stream_kimi(ctx, key, options)

    provider: Provider[Any] = Provider(
        id="openrouter",
        name="OpenRouter Moonshot Kimi K2.6",
        models=[kimi_model],
        stream_fn=stream_fn,
    )
    models.set_provider(provider)

    return kimi_model, models, {"base_url": DEFAULT_BASE_URL}


def _make_error_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai",
        provider="openrouter",
        model="moonshotai/kimi-k2.6",
        stop_reason=StopReason.ERROR,
        error_message=text,
        timestamp=int(time.time() * 1000),
    )


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    """Convert pi messages to the OpenAI-compatible chat format OpenRouter expects."""
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append({"role": "user", "content": user_content_to_openai(msg.content)})
        elif isinstance(msg, AssistantMessage):
            text_parts: list[str] = []
            tool_calls_out: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolCall):
                    tool_calls_out.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                        }
                    )
            openai_msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls_out:
                openai_msg["tool_calls"] = tool_calls_out
            messages.append(openai_msg)
        elif isinstance(msg, ToolResultMessage):
            messages.extend(tool_result_to_openai_messages(msg))

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})
    return messages


def _stream_kimi(
    context: Context,
    api_key: str,
    options: Any | None,
) -> AssistantMessageEventStream:
    """Stream from Kimi K2.6 via OpenRouter."""
    stream = create_assistant_message_event_stream()

    if httpx is None:
        err = _make_error_message("httpx not installed")
        stream.push(ErrorEvent(reason="error", error=err))
        stream.end(err)
        return stream

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/piegames/python-pi",
        "X-Title": "Python Pi Agent",
    }

    messages = _convert_messages(context)

    payload: dict[str, Any] = {
        "model": "moonshotai/kimi-k2.6",
        "messages": messages,
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else MAX_TOKENS,
        "stream": True,
    }
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature

    if context.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in context.tools
        ]

    async def _run() -> None:
        try:
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream(
                    "POST",
                    DEFAULT_BASE_URL,
                    headers=headers,
                    json=payload,
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    err = _make_error_message(f"OpenRouter API error {response.status_code}: {error_text.decode()}")
                    stream.push(ErrorEvent(reason="error", error=err))
                    stream.end(err)
                    return

                partial = AssistantMessage(
                    content=[],
                    api="openai",
                    provider="openrouter",
                    model="moonshotai/kimi-k2.6",
                    stop_reason=StopReason.PENDING,
                    timestamp=int(time.time() * 1000),
                )
                stream.push(StartEvent(partial=partial))

                text_content: TextContent | None = None
                text_buffer = ""
                content_index = 0
                tool_calls_acc: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None

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

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    if delta.get("content"):
                        if text_content is None:
                            text_content = TextContent(text="")
                            partial.content.append(text_content)
                        text_chunk = delta["content"]
                        text_buffer += text_chunk
                        text_content.text += text_chunk
                        stream.push(TextDeltaEvent(content_index=content_index, delta=text_chunk, partial=partial))

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            tc_idx = tc.get("index", 0)
                            if tc_idx not in tool_calls_acc:
                                tool_calls_acc[tc_idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                                if tc.get("function"):
                                    tool_calls_acc[tc_idx]["name"] = tc["function"].get("name", "")
                            if tc.get("id"):
                                tool_calls_acc[tc_idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls_acc[tc_idx]["arguments"] += tc["function"]["arguments"]

                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                if text_content is not None:
                    stream.push(TextEndEvent(content_index=content_index, content=text_buffer, partial=partial))
                    content_index += 1

                for tc_idx in sorted(tool_calls_acc):
                    tc_data = tool_calls_acc[tc_idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_call = ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args)
                    partial.content.append(tool_call)
                    stream.push(ToolCallEndEvent(content_index=content_index, tool_call=tool_call, partial=partial))
                    content_index += 1

                if finish_reason == "tool_calls":
                    partial.stop_reason = StopReason.TOOL_USE
                elif finish_reason == "length":
                    partial.stop_reason = StopReason.LENGTH
                else:
                    partial.stop_reason = StopReason.STOP

                done_reason = cast('Literal["stop", "length", "toolUse"]', partial.stop_reason.value)
                stream.push(DoneEvent(reason=done_reason, message=partial))
                stream.end(partial)
        except Exception as exc:
            err = _make_error_message(describe_exception(exc))
            stream.push(ErrorEvent(reason="error", error=err))
            stream.end(err)

    spawn_background_task(_run())
    return stream
