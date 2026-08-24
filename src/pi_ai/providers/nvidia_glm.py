"""NVIDIA GLM 5.2 provider via NVIDIA Inference API.

A powerful open-source model available via NVIDIA's Inference API.
Uses the OpenAI-compatible API format at nvidia endpoint.

Usage:
    export NVAPI_KEY=***
    from pi_ai.providers.nvidia_glm import nvidia_glm_provider
    from pi_ai import Model

    model, models, meta = nvidia_glm_provider(Model(id="test"))
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
from pi_ai.event_stream import create_assistant_message_event_stream
from pi_ai.models import MutableModels, Provider
from pi_ai.providers._openai_compat import tool_result_to_openai_messages, user_content_to_openai
from pi_ai.utils import describe_exception, spawn_background_task

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS = 16384
MODEL_ID = "z-ai/glm-5.2"


def nvidia_glm_provider(
    model: Model,
    api_key: str | None = None,
) -> tuple[Model, MutableModels, dict[str, Any]]:
    """Create a models collection configured for NVIDIA GLM 5.2.

    Returns: (model, models_collection, empty_metadata_dict)
    """
    import os

    key = api_key or os.environ.get("NVAPI_KEY")
    if not key:
        raise ValueError("NVAPI_KEY environment variable or api_key parameter required")

    models = MutableModels()

    glm_model = Model(
        id=MODEL_ID,
        name="GLM 5.2",
        api="openai-completions",
        provider="nvidia",
        context_window=16384,
        max_tokens=16384,
    )

    def stream_fn(
        model_obj: Model,
        ctx: Context,
        options: Any | None = None,
    ) -> Any:
        return _create_stream(key, ctx, options)

    provider: Provider[Any] = Provider(
        id="nvidia",
        name="NVIDIA GLM 5.2",
        models=[glm_model],
        stream_fn=stream_fn,
    )
    models.set_provider(provider)

    return glm_model, models, {"base_url": DEFAULT_BASE_URL}


def _make_error_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="nvidia",
        model=MODEL_ID,
        stop_reason=StopReason.ERROR,
        error_message=text,
        timestamp=int(time.time() * 1000),
    )


def _create_stream(
    api_key: str,
    context: Context,
    options: Any | None,
) -> Any:
    """Create an async event stream for GLM 5.2 response."""
    stream = create_assistant_message_event_stream()

    if httpx is None:
        err = _make_error_message("httpx not installed")
        stream.push(ErrorEvent(reason="error", error=err))
        stream.end(err)
        return stream

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append({"role": "user", "content": user_content_to_openai(msg.content)})
        elif isinstance(msg, AssistantMessage):
            openai_msg: dict[str, Any] = {"role": "assistant"}
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
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                        }
                    )
            if text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            if tool_calls_out:
                openai_msg["tool_calls"] = tool_calls_out
            messages.append(openai_msg)
        elif isinstance(msg, ToolResultMessage):
            messages.extend(tool_result_to_openai_messages(msg))

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})

    payload: dict[str, Any] = {
        "model": MODEL_ID,
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

    async def _run_stream() -> None:
        try:
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream(
                    "POST",
                    DEFAULT_BASE_URL + "/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    err = _make_error_message(f"NVIDIA API error {response.status_code}: {error_text.decode()}")
                    stream.push(ErrorEvent(reason="error", error=err))
                    stream.end(err)
                    return

                partial = AssistantMessage(
                    content=[],
                    api="openai-completions",
                    provider="nvidia",
                    model=MODEL_ID,
                    stop_reason=StopReason.PENDING,
                    timestamp=int(time.time() * 1000),
                )
                stream.push(StartEvent(partial=partial))

                text_buffer = ""
                current_text_started = False
                content_index = 0
                # Accumulator: index → {id, name, arguments}
                tool_calls_acc: dict[int, dict[str, Any]] = {}
                finish_reason: str | None = None
                done_pushed = False

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

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # --- text content ---
                    if delta.get("content"):
                        if not current_text_started:
                            partial.content.append(TextContent(text=""))
                            current_text_started = True
                        text_chunk = delta["content"]
                        text_buffer += text_chunk
                        partial.content[content_index].text += text_chunk  # type: ignore[union-attr]
                        stream.push(TextDeltaEvent(content_index=content_index, delta=text_chunk, partial=partial))

                    # --- tool calls ---
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            tc_idx = tc.get("index", 0)
                            if tc_idx not in tool_calls_acc:
                                tool_calls_acc[tc_idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                                if tc.get("function"):
                                    tool_calls_acc[tc_idx]["name"] = tc["function"].get("name", "")
                            if tc.get("function", {}).get("arguments"):
                                tool_calls_acc[tc_idx]["arguments"] += tc["function"]["arguments"]
                            if tc.get("id"):
                                tool_calls_acc[tc_idx]["id"] = tc["id"]

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                # --- finalise after loop ---
                if current_text_started:
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

                # Determine stop reason
                if finish_reason == "tool_calls":
                    partial.stop_reason = StopReason.TOOL_USE
                elif finish_reason == "length":
                    partial.stop_reason = StopReason.LENGTH
                else:
                    partial.stop_reason = StopReason.STOP

                done_reason = cast('Literal["stop", "length", "toolUse"]', partial.stop_reason.value)
                stream.push(DoneEvent(reason=done_reason, message=partial))
                stream.end(partial)
                done_pushed = True

                if not done_pushed:
                    # Fallback: stream ended with no finish_reason.
                    partial.stop_reason = StopReason.STOP
                    stream.push(DoneEvent(reason="stop", message=partial))
                    stream.end(partial)

        except Exception as exc:
            err = _make_error_message(describe_exception(exc))
            stream.push(ErrorEvent(reason="error", error=err))
            stream.end(err)

    spawn_background_task(_run_stream())
    return stream
