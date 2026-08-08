"""Moonshot/Kimi K2.6 provider via NVIDIA Inference API.

Mirrors packages/ai/src/providers/nvidia.ts for moonshotai/kimi-k2.6.
Uses the NVIDIA Inference API endpoint for Kimi K2.6 access.

Usage:
    export NVAPI_KEY="nvapi-..."

    from pi_ai.providers.nvidia_moonshot import nvidia_moonshot_provider
    from pi_ai import Model

    model, models, auth = nvidia_moonshot_provider(Model(id="moonshotai/kimi-k2.6"))
"""

from __future__ import annotations

import asyncio
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
from pi_ai.utils import describe_exception

NIGHTLY_API = "openai"  # Uses OpenAI-compatible API format
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS = 16384


def nvidia_moonshot_provider(
    model: Model,
    api_key: str | None = None,
) -> tuple[Model, MutableModels, dict[str, Any]]:
    """
    Create a models collection configured for Moonshot Kimi K2.6 via NVIDIA.

    Returns: (model, models_collection, empty_metadata_dict)
    """
    import os

    key = api_key or os.environ.get("NVAPI_KEY")
    if not key:
        raise ValueError("NVAPI_KEY environment variable or api_key parameter required")

    models = MutableModels()

    kimi_model = Model(
        id="moonshotai/kimi-k2.6",
        name="Kimi K2.6",
        api="openai",
        provider="nvidia",
        context_window=16384,
        max_tokens=16384,
    )

    def stream_fn(
        m: Model,
        ctx: Context,
        options: Any | None = None,
    ) -> AssistantMessageEventStream:
        return _stream_kimi(ctx, key, m.id, options)

    provider: Provider[Any] = Provider(
        id="nvidia",
        name="NVIDIA Moonshot Kimi K2.6",
        models=[kimi_model],
        stream_fn=stream_fn,
    )
    models.set_provider(provider)

    return kimi_model, models, {"base_url": DEFAULT_BASE_URL}


def _make_error_message(text: str, model_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=NIGHTLY_API,
        provider="nvidia",
        model=model_id,
        stop_reason=StopReason.ERROR,
        error_message=text,
        timestamp=int(time.time() * 1000),
    )


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    """Convert pi messages to the OpenAI-compatible chat format Kimi expects."""
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            content = (
                msg.content
                if isinstance(msg.content, str)
                else "".join(c.text if isinstance(c, TextContent) else "" for c in msg.content)
            )
            messages.append({"role": "user", "content": content})
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
            result_text = "\n".join(b.text for b in msg.content if isinstance(b, TextContent))
            messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": result_text})

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})
    return messages


def _stream_kimi(
    context: Context,
    api_key: str,
    model_id: str,
    options: Any | None,
) -> AssistantMessageEventStream:
    """Stream from Kimi K2.6 via NVIDIA Inference API."""
    stream = create_assistant_message_event_stream()

    if httpx is None:
        err = _make_error_message("httpx not installed", model_id)
        stream.push(ErrorEvent(reason="error", error=err))
        stream.end(err)
        return stream

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    messages = _convert_messages(context)

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else MAX_TOKENS,
        "seed": 0,
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
                    err = _make_error_message(
                        f"NVIDIA API error {response.status_code}: {error_text.decode()}", model_id
                    )
                    stream.push(ErrorEvent(reason="error", error=err))
                    stream.end(err)
                    return

                partial = AssistantMessage(
                    content=[],
                    api=NIGHTLY_API,
                    provider="nvidia",
                    model=model_id,
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
            err = _make_error_message(describe_exception(exc), model_id)
            stream.push(ErrorEvent(reason="error", error=err))
            stream.end(err)

    _run_task = asyncio.ensure_future(_run())  # noqa: RUF006
    return stream
