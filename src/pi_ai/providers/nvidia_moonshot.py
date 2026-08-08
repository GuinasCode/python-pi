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

import json
from typing import Any

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
    TextDeltaEvent,
    TextEndEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pi_ai.event_stream import create_assistant_message_event_stream
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
        ctx: Context,
        options: Any | None = None,
        state: dict[str, int] | None = None,
    ) -> Any:
        return _stream_kimi(ctx, key, model.id, options, state or {})

    provider = Provider(
        id="nvidia",
        name="NVIDIA Moonshot Kimi K2.6",
        models=[kimi_model],
        stream_fn=stream_fn,  # type: ignore[arg-type]
    )
    models.set_provider(provider)

    return kimi_model, models, {"base_url": DEFAULT_BASE_URL}


async def _stream_kimi(
    context: Context,
    api_key: str,
    model_id: str,
    options: Any | None,
    state: dict[str, int],
) -> None:
    """Stream from Kimi K2.6 via NVIDIA Inference API."""

    if httpx is None:
        stream = create_assistant_message_event_stream()
        stream.push(ErrorEvent(reason="error", error="httpx not installed"))
        stream.end("httpx not installed")
        return

    stream = create_assistant_message_event_stream()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    # Convert messages

    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if msg.role == "user":
            content = msg.content if isinstance(msg.content, str) else "".join(
                c.text if hasattr(c, "text") else "" for c in msg.content
            )
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            content = "".join(
                block.text if block.type == "text" else ""
                for block in msg.content
            )
            messages.append({"role": "assistant", "content": content})

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else MAX_TOKENS,
        "seed": 0,
        "stream": True,
    }
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature

    # Add tools if present
    if context.tools:
        tools_list: list[dict[str, Any]] = []
        for tool in context.tools:
            tools_list.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        payload["tools"] = tools_list

    try:
        async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
            "POST",
            DEFAULT_BASE_URL,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                stream.push(
                    ErrorEvent(reason="error", error=f"NVIDIA API error {response.status_code}: {error_text.decode()}")
                )
                stream.end(f"NVIDIA API error {response.status_code}")
                return

            stream.push(StartEvent(partial=AssistantMessage(stop_reason=StopReason.PENDING)))

            text_buffer = ""
            tool_calls: dict[str, dict[str, Any]] = {}

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

                await _process_kimi_chunk(chunk, stream, text_buffer, tool_calls)
                text_buffer = await _get_and_reset_text_buffer(stream)
                tool_calls = await _get_and_reset_tool_calls(stream)

            stream.end("DONE")
    except Exception as exc:
        msg = describe_exception(exc)
        stream.push(ErrorEvent(reason="error", error=msg))
        stream.end(msg)


async def _process_kimi_chunk(
    chunk: dict[str, Any],
    stream: Any,
    text_buffer: str,
    tool_calls: dict[str, dict[str, Any]],
) -> None:
    """Process a single chunk from Kimi K2.6 response."""
    delta = chunk.get("choices", [{}])[0].get("delta", {})

    if delta.get("content"):
        stream.push(TextDeltaEvent(delta=delta["content"]))

    if delta.get("tool_calls"):
        for tc in delta["tool_calls"]:
            stream.push(ToolCallStartEvent(name=tc.get("function", {}).get("name", "")))
            stream.push(ToolCallEndEvent(name=tc.get("function", {}).get("name", "")))

    if chunk.get("choices", [{}])[0].get("finish_reason") is not None:
        stream.push(TextEndEvent(content=text_buffer))
        stream.push(DoneEvent(reason=chunk["choices"][0]["finish_reason"]))


async def _get_and_reset_text_buffer(stream: Any) -> str:
    """Helper to maintain text buffer state."""
    return ""


async def _get_and_reset_tool_calls(stream: Any) -> dict[str, Any]:
    """Helper to maintain tool calls state."""
    return {}