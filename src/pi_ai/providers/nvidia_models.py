"""NVIDIA-hosted OpenAI-compatible chat models, with an automatic
fallback chain across several of them.

Registers Nemotron 3 Super/Ultra and gpt-oss 120B/20B (all served from
the same ``integrate.api.nvidia.com`` OpenAI-compatible endpoint as
``nvidia_glm.py``'s GLM 5.2 — this module generalizes that file's
streaming/parsing logic instead of duplicating it once per model id) as
individually selectable models, plus one extra "auto" model
(``nvidia/auto``) that tries them in the order below and automatically
retries the next one on a connection failure or non-2xx status —
*before* any content has streamed back to the caller. A failure partway
through an already-started response is reported as a normal error
instead, never retried: the caller may already be showing that partial
content, and silently discarding it to start over on a different model
would be a worse experience than just surfacing the error.

Usage:
    export NVAPI_KEY=***
    from pi_ai.providers.nvidia_models import nvidia_models_provider
    from pi_ai import Model

    model, models, meta = nvidia_models_provider(Model(id="test"))
    # model is nvidia/auto by default — models.get_models("nvidia") lists
    # every individual model too, selectable directly if a specific one
    # (not the fallback chain) is wanted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
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
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolResultMessage,
    UserMessage,
)
from pi_ai.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from pi_ai.models import MutableModels, Provider
from pi_ai.utils import describe_exception, spawn_background_task

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
AUTO_MODEL_ID = "nvidia/auto"

# (model_id, display_name, context_window, default_max_tokens) — default
# max_tokens matches what each model's own NVIDIA API example uses.
_MODEL_SPECS: list[tuple[str, str, int, int]] = [
    ("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B", 131072, 16384),
    ("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B", 131072, 16384),
    ("openai/gpt-oss-120b", "GPT-OSS 120B", 131072, 4096),
    ("openai/gpt-oss-20b", "GPT-OSS 20B", 131072, 4096),
]
_MAX_TOKENS_BY_ID = {model_id: max_tokens for model_id, _name, _cw, max_tokens in _MODEL_SPECS}


class _UpstreamUnavailable(Exception):
    """Raised only for a failure before any content reached the caller
    (connection error, timeout, non-2xx status) — the signal the
    fallback chain retries the next model for."""


def _openai_messages(context: Context) -> list[dict[str, Any]]:
    """Translate pi_ai's Context.messages into OpenAI chat-completions
    message dicts — every model here is on the same OpenAI-compatible
    endpoint, so this is shared rather than duplicated per model."""
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            content = (
                msg.content
                if isinstance(msg.content, str)
                else "".join(c.text if hasattr(c, "text") else "" for c in msg.content)
            )
            messages.append({"role": "user", "content": content})
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
                            "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                        }
                    )
            if text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            if tool_calls_out:
                openai_msg["tool_calls"] = tool_calls_out
            messages.append(openai_msg)
        elif isinstance(msg, ToolResultMessage):
            text_parts_tr: list[str] = []
            for result_block in msg.content:
                if isinstance(result_block, TextContent):
                    text_parts_tr.append(result_block.text)
            messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": "\n".join(text_parts_tr)})
    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})
    return messages


def _make_error_message(model_id: str, text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api="openai-completions",
        provider="nvidia",
        model=model_id,
        stop_reason=StopReason.ERROR,
        error_message=text,
        timestamp=int(time.time() * 1000),
    )


async def _consume_and_finish(model_id: str, response: Any, stream: AssistantMessageEventStream) -> None:
    """Parse one SSE chat-completion response into stream events and
    resolve the stream — success or failure. Never raises: once this is
    called, the request already succeeded (200 status), so a failure from
    here on (a dropped connection mid-stream, malformed data, ...) is
    reported as a normal terminal error, not something a fallback chain
    should retry — the caller may already be showing partial content
    from this attempt, so silently discarding it and starting over on a
    different model would be worse than just surfacing the error.
    """
    try:
        partial = AssistantMessage(
            content=[],
            api="openai-completions",
            provider="nvidia",
            model=model_id,
            stop_reason=StopReason.PENDING,
            timestamp=int(time.time() * 1000),
        )
        stream.push(StartEvent(partial=partial))

        text_buffer = ""
        current_text_started = False
        reasoning_started = False
        content_index = 0
        # Accumulator: index → {id, name, arguments}
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

            # --- reasoning ("thinking") content, if the model streamed any ---
            reasoning_chunk = delta.get("reasoning_content")
            if reasoning_chunk:
                if not reasoning_started:
                    partial.content.append(ThinkingContent(thinking=""))
                    stream.push(ThinkingStartEvent(content_index=content_index))
                    reasoning_started = True
                thinking_block = partial.content[content_index]
                assert isinstance(thinking_block, ThinkingContent)
                thinking_block.thinking += reasoning_chunk
                stream.push(ThinkingDeltaEvent(content_index=content_index, delta=reasoning_chunk, partial=partial))

            # --- text content ---
            if delta.get("content"):
                if reasoning_started:
                    thinking_block = partial.content[content_index]
                    assert isinstance(thinking_block, ThinkingContent)
                    stream.push(ThinkingEndEvent(content_index=content_index, content=thinking_block.thinking))
                    content_index += 1
                    reasoning_started = False
                if not current_text_started:
                    partial.content.append(TextContent(text=""))
                    current_text_started = True
                text_chunk = delta["content"]
                text_buffer += text_chunk
                text_block = partial.content[content_index]
                assert isinstance(text_block, TextContent)
                text_block.text += text_chunk
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

            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]

        # --- finalize after loop ---
        if reasoning_started:
            thinking_block = partial.content[content_index]
            assert isinstance(thinking_block, ThinkingContent)
            stream.push(ThinkingEndEvent(content_index=content_index, content=thinking_block.thinking))
            content_index += 1
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
        err = _make_error_message(model_id, describe_exception(exc))
        stream.push(ErrorEvent(reason="error", error=err))
        stream.end(err)


async def _stream_one_model(
    model_id: str,
    api_key: str,
    context: Context,
    options: Any | None,
    stream: AssistantMessageEventStream,
    max_tokens: int,
) -> None:
    """Connect to `model_id` and stream its response into `stream`.

    Raises _UpstreamUnavailable for a failure before any content reached
    the caller (httpx missing, connection/DNS failure, non-2xx status) —
    everything after a validated 200 response is _consume_and_finish's
    responsibility, which never raises (see its docstring).
    """
    if httpx is None:
        raise _UpstreamUnavailable("httpx not installed")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": _openai_messages(context),
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else max_tokens,
        "stream": True,
    }
    if options and options.temperature is not None:
        payload["temperature"] = options.temperature
    if context.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in context.tools
        ]

    try:
        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream("POST", DEFAULT_BASE_URL + "/chat/completions", headers=headers, json=payload) as response,
        ):
            if response.status_code != 200:
                error_text = await response.aread()
                raise _UpstreamUnavailable(f"HTTP {response.status_code}: {error_text.decode(errors='replace')}")
            await _consume_and_finish(model_id, response, stream)
    except httpx.HTTPError as exc:
        raise _UpstreamUnavailable(str(exc)) from exc


def _single_model_stream_fn(model_id: str, api_key: str, max_tokens: int) -> Any:
    """stream_fn for one individually-selected model — no fallback, a
    connection/status failure just becomes a normal terminal error."""

    def stream_fn(_model: Model, context: Context, options: Any | None = None) -> Any:
        stream = create_assistant_message_event_stream()

        async def _run() -> None:
            try:
                await _stream_one_model(model_id, api_key, context, options, stream, max_tokens)
            except _UpstreamUnavailable as exc:
                err = _make_error_message(model_id, str(exc))
                stream.push(ErrorEvent(reason="error", error=err))
                stream.end(err)

        spawn_background_task(_run())
        return stream

    return stream_fn


def _fallback_stream_fn(model_ids: Sequence[str], api_key: str) -> Any:
    """stream_fn for the `nvidia/auto` model — tries each of `model_ids`
    in order, moving on to the next only on _UpstreamUnavailable (a
    pre-content failure); reports all of them failing as one error."""

    def stream_fn(_model: Model, context: Context, options: Any | None = None) -> Any:
        stream = create_assistant_message_event_stream()

        async def _run() -> None:
            errors: list[str] = []
            for model_id in model_ids:
                try:
                    await _stream_one_model(model_id, api_key, context, options, stream, _MAX_TOKENS_BY_ID[model_id])
                    return
                except _UpstreamUnavailable as exc:
                    errors.append(f"{model_id}: {exc}")
                    continue
            summary = "\n".join(errors) if errors else "no models configured"
            err = _make_error_message(AUTO_MODEL_ID, f"All fallback models failed:\n{summary}")
            stream.push(ErrorEvent(reason="error", error=err))
            stream.end(err)

        spawn_background_task(_run())
        return stream

    return stream_fn


def nvidia_models_provider(
    model: Model,
    api_key: str | None = None,
) -> tuple[Model, MutableModels, dict[str, Any]]:
    """Create a models collection with Nemotron 3 Super/Ultra, gpt-oss
    120B/20B, and an `nvidia/auto` model that tries them in that order
    with automatic fallback on a pre-content failure.

    Returns: (nvidia/auto model, models_collection, {"base_url": ...})
    — `model` (the first argument) is accepted for symmetry with the
    other single-model provider factories in this package but otherwise
    unused; the returned model is always `nvidia/auto`.
    """
    import os

    key = api_key or os.environ.get("NVAPI_KEY")
    if not key:
        raise ValueError("NVAPI_KEY environment variable or api_key parameter required")

    models = MutableModels()
    catalog: list[Model] = []
    provider_models: dict[str, Any] = {}

    for model_id, name, context_window, max_tokens in _MODEL_SPECS:
        m = Model(
            id=model_id,
            name=name,
            api="openai-completions",
            provider="nvidia",
            context_window=context_window,
            max_tokens=max_tokens,
        )
        catalog.append(m)
        provider_models[model_id] = _single_model_stream_fn(model_id, key, max_tokens)

    auto_model = Model(
        id=AUTO_MODEL_ID,
        name="Auto (Nemotron/GPT-OSS fallback chain)",
        api="openai-completions",
        provider="nvidia",
        context_window=min(cw for _i, _n, cw, _mt in _MODEL_SPECS),
        max_tokens=max(mt for _i, _n, _cw, mt in _MODEL_SPECS),
    )
    catalog.append(auto_model)
    fallback_chain = [model_id for model_id, _name, _cw, _mt in _MODEL_SPECS]

    # A single Provider only has one stream_fn, dispatched by the model
    # object passed to it — route by model.id to each model's own
    # pre-built stream_fn instead.
    per_model_stream_fns = {**provider_models, AUTO_MODEL_ID: _fallback_stream_fn(fallback_chain, key)}

    def stream_fn(model_obj: Model, context: Context, options: Any | None = None) -> Any:
        return per_model_stream_fns[model_obj.id](model_obj, context, options)

    provider: Provider[Any] = Provider(
        id="nvidia",
        name="NVIDIA (Nemotron / GPT-OSS)",
        models=catalog,
        stream_fn=stream_fn,
    )
    models.set_provider(provider)

    return auto_model, models, {"base_url": DEFAULT_BASE_URL}
