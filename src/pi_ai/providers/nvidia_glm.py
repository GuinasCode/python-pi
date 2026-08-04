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
from collections.abc import AsyncIterator
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


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_TOKENS = 16384
MODEL_ID = "z-ai/glm-5.2"


def nvidia_glm_provider(
    model: Model,
    api_key: str | None = None,
) -> tuple[Model, MutableModels, dict[str, Any]]:
    """
    Create a models collection configured for NVIDIA GLM 5.2.

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
        api="openai",
        provider="nvidia",
        context_window=16384,
        max_tokens=16384,
    )

    def stream_fn(
        model_obj: Model,
        ctx: Context,
        options: Any | None = None,
    ):
        """Stream function that returns an event stream."""
        return _create_stream(key, ctx, options)

    provider = Provider(
        id="nvidia",
        name="NVIDIA GLM 5.2",
        models=[glm_model],
        stream_fn=stream_fn,  # type: ignore[arg-type]
    )
    models.set_provider(provider)

    return glm_model, models, {"base_url": DEFAULT_BASE_URL}


def _create_stream(
    api_key: str,
    context: Context,
    options: Any | None,
) -> Any:
    """Create an async event stream for GLM 5.2 response."""
    stream = create_assistant_message_event_stream()
    
    if httpx is None:
        stream.push(ErrorEvent(reason="error", error="httpx not installed"))
        stream.end("httpx not installed")
        return stream

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": options.max_tokens if options else MAX_TOKENS,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": True,
    }

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

    # Run the stream in a background task
    import asyncio
    
    async def _run_stream() -> None:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    DEFAULT_BASE_URL + "/chat/completions",
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
                            if "content" in delta and delta["content"]:
                                stream.push(TextDeltaEvent(delta=delta["content"]))
                            
                            if chunk["choices"][0].get("finish_reason"):
                                stream.push(TextEndEvent(content=delta.get("content", "")))
                                stream.push(DoneEvent(reason=chunk["choices"][0]["finish_reason"]))

        except Exception as exc:
            stream.push(ErrorEvent(reason="error", error=str(exc)))
            stream.end(str(exc))

    # Start the streaming in a background task
    asyncio.create_task(_run_stream())
    
    return stream