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
)
from pi_ai.event_stream import create_assistant_message_event_stream
from pi_ai.models import MutableModels, Provider
from pi_ai.utils import describe_exception

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
        ctx: Context,
        options: Any | None = None,
        state: dict[str, int] | None = None,
    ) -> Any:
        return _stream_kimi(ctx, key, options, state or {})

    provider = Provider(
        id="openrouter",
        name="OpenRouter Moonshot Kimi K2.6",
        models=[kimi_model],
        stream_fn=stream_fn,  # type: ignore[arg-type]
    )
    models.set_provider(provider)

    return kimi_model, models, {"base_url": DEFAULT_BASE_URL}


async def _stream_kimi(
    context: Context,
    api_key: str,
    options: Any | None,
    state: dict[str, int],
) -> None:
    """Stream from Kimi K2.6 via OpenRouter."""

    if httpx is None:
        stream = create_assistant_message_event_stream()
        stream.push(ErrorEvent(reason="error", error="httpx not installed"))
        stream.end("httpx not installed")
        return

    stream = create_assistant_message_event_stream()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/piegames/python-pi",
        "X-Title": "Python Pi Agent",
    }

    # Convert messages
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if msg.role == "user":
            content = (
                msg.content
                if isinstance(msg.content, str)
                else "".join(c.text if hasattr(c, "text") else "" for c in msg.content)
            )
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            content = "".join(block.text if block.type == "text" else "" for block in msg.content)
            messages.append({"role": "assistant", "content": content})

    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})

    payload: dict[str, Any] = {
        "model": "moonshotai/kimi-k2.6",
        "messages": messages,
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else MAX_TOKENS,
        "stream": True,
    }
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature

    # Add tools if present
    if context.tools:
        tools_list: list[dict[str, Any]] = []
        for tool in context.tools:
            tools_list.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        payload["tools"] = tools_list

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
                error_msg = f"OpenRouter API error {response.status_code}: {error_text.decode()}"
                stream.push(ErrorEvent(reason="error", error=error_msg))
                stream.end(f"OpenRouter API error {response.status_code}")
                return

            stream.push(StartEvent(partial=AssistantMessage(stop_reason=StopReason.PENDING)))

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

                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        stream.push(TextDeltaEvent(delta=delta["content"]))

                    if chunk["choices"][0].get("finish_reason"):
                        stream.push(TextEndEvent(content=delta.get("content", "")))
                        stream.push(DoneEvent(reason=chunk["choices"][0]["finish_reason"]))

    except Exception as exc:
        msg = describe_exception(exc)
        stream.push(ErrorEvent(reason="error", error=msg))
        stream.end(msg)
