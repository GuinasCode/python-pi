"""NVIDIA-hosted OpenAI-compatible chat models, with an automatic
fallback chain across several of them.

Registers Nemotron 3 Super/Ultra, gpt-oss 120B/20B, MiniMax M3, and two
vision-capable models (Gemma 4 31B, Llama 3.2 90B Vision) — all served
from the same ``integrate.api.nvidia.com`` OpenAI-compatible endpoint as
``nvidia_glm.py``'s GLM 5.2 — this module generalizes that file's
streaming/parsing logic instead of duplicating it once per model id) as
individually selectable models, plus one extra "auto" model
(``nvidia/auto``) that tries the four Nemotron/gpt-oss models in order
and automatically retries the next one on a connection failure or
non-2xx status — *before* any content has streamed back to the caller. A
failure partway through an already-started response is reported as a
normal error instead, never retried: the caller may already be showing
that partial content, and silently discarding it to start over on a
different model would be a worse experience than just surfacing the
error. MiniMax and the two vision models are deliberately not part of
that chain — different model families, not equivalent substitutes for
the text-only Nemotron/gpt-oss lineup it's for.

User messages with image content translate to OpenAI's
``image_url``/data-URI content-part shape, matching how
``pi_ai.providers.openai`` already does the same conversion for
``ImageContent`` blocks (``pi_ai``'s images always carry base64 data,
not a bare remote URL, so a data URI is the only faithful translation
regardless of what the original source image was).

Usage:
    export NVAPI_KEY=***
    from pi_ai.providers.nvidia_models import nvidia_models_provider
    from pi_ai import Model

    model, models, meta = nvidia_models_provider(Model(id="test"))
    # model is DEFAULT_MODEL_ID (MiniMax M3) by default —
    # models.get_models("nvidia") lists every other model too (including
    # nvidia/auto, the fallback chain), individually selectable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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
    ImageContent,
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
DEFAULT_MODEL_ID = "minimaxai/minimax-m3"


@dataclass(frozen=True)
class _ModelSpec:
    model_id: str
    name: str
    context_window: int
    max_tokens: int
    top_p: float = 0.95
    # Nemotron's reasoning ("thinking") output is opt-in per NVIDIA's own
    # examples — off unless this sends chat_template_kwargs.enable_thinking.
    enable_thinking: bool = False
    # Included in the fallback chain nvidia/auto walks through.
    in_fallback_chain: bool = True
    # Whether this model accepts ImageContent in a user message — set on
    # Model.input so a caller can tell which models can actually use one.
    supports_images: bool = False
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


# default max_tokens/top_p/penalties match each model's own NVIDIA API example.
_MODEL_SPECS: list[_ModelSpec] = [
    _ModelSpec(
        "nvidia/nemotron-3-super-120b-a12b",
        "Nemotron 3 Super 120B",
        131072,
        16384,
        top_p=0.95,
        enable_thinking=True,
    ),
    _ModelSpec(
        "nvidia/nemotron-3-ultra-550b-a55b",
        "Nemotron 3 Ultra 550B",
        131072,
        16384,
        top_p=0.95,
        enable_thinking=True,
    ),
    _ModelSpec("openai/gpt-oss-120b", "GPT-OSS 120B", 131072, 4096, top_p=1.0),
    _ModelSpec("openai/gpt-oss-20b", "GPT-OSS 20B", 131072, 4096, top_p=1.0),
    _ModelSpec(DEFAULT_MODEL_ID, "MiniMax M3", 8192, 8192, top_p=0.95, in_fallback_chain=False),
    _ModelSpec(
        "google/gemma-4-31b-it",
        "Gemma 4 31B (vision)",
        131072,
        16384,
        top_p=0.95,
        enable_thinking=True,
        in_fallback_chain=False,
        supports_images=True,
    ),
    _ModelSpec(
        "meta/llama-3.2-90b-vision-instruct",
        "Llama 3.2 90B Vision",
        131072,
        512,
        top_p=1.0,
        in_fallback_chain=False,
        supports_images=True,
        frequency_penalty=0,
        presence_penalty=0,
    ),
]


class _UpstreamUnavailable(Exception):
    """Raised only for a failure before any content reached the caller
    (connection error, timeout, non-2xx status) — the signal the
    fallback chain retries the next model for."""


def _user_content_to_openai(content: str | list[Any]) -> str | list[dict[str, Any]]:
    """OpenAI content-part translation for a user message — plain string
    stays a string, otherwise TextContent/ImageContent blocks become
    ``{"type": "text", ...}``/``{"type": "image_url", ...}`` parts,
    mirroring ``pi_ai.providers.openai``'s ``_content_to_openai`` exactly
    (ImageContent only ever carries base64 data, never a bare remote URL,
    so a data URI is the only faithful translation here too)."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append({"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"}})
    return parts


def _openai_messages(context: Context) -> list[dict[str, Any]]:
    """Translate pi_ai's Context.messages into OpenAI chat-completions
    message dicts — every model here is on the same OpenAI-compatible
    endpoint, so this is shared rather than duplicated per model."""
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append({"role": "user", "content": _user_content_to_openai(msg.content)})
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
    spec: _ModelSpec,
    api_key: str,
    context: Context,
    options: Any | None,
    stream: AssistantMessageEventStream,
) -> None:
    """Connect to `spec`'s model and stream its response into `stream`.

    Raises _UpstreamUnavailable for a failure before any content reached
    the caller (httpx missing, connection/DNS failure, non-2xx status) —
    everything after a validated 200 response is _consume_and_finish's
    responsibility, which never raises (see its docstring).
    """
    if httpx is None:
        raise _UpstreamUnavailable("httpx not installed")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": spec.model_id,
        "messages": _openai_messages(context),
        "max_tokens": options.max_tokens if options and options.max_tokens is not None else spec.max_tokens,
        # Every one of these models' own NVIDIA API examples pins
        # temperature=1 explicitly rather than omitting it — matched here
        # the same way, only overridden when the caller actually asks for
        # something else.
        "temperature": options.temperature if options and options.temperature is not None else 1,
        "top_p": spec.top_p,
        "stream": True,
    }
    if spec.enable_thinking:
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    if spec.frequency_penalty is not None:
        payload["frequency_penalty"] = spec.frequency_penalty
    if spec.presence_penalty is not None:
        payload["presence_penalty"] = spec.presence_penalty
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
            await _consume_and_finish(spec.model_id, response, stream)
    except httpx.HTTPError as exc:
        raise _UpstreamUnavailable(str(exc)) from exc


def _single_model_stream_fn(spec: _ModelSpec, api_key: str) -> Any:
    """stream_fn for one individually-selected model — no fallback, a
    connection/status failure just becomes a normal terminal error."""

    def stream_fn(_model: Model, context: Context, options: Any | None = None) -> Any:
        stream = create_assistant_message_event_stream()

        async def _run() -> None:
            try:
                await _stream_one_model(spec, api_key, context, options, stream)
            except _UpstreamUnavailable as exc:
                err = _make_error_message(spec.model_id, str(exc))
                stream.push(ErrorEvent(reason="error", error=err))
                stream.end(err)

        spawn_background_task(_run())
        return stream

    return stream_fn


def _fallback_stream_fn(chain: list[_ModelSpec], api_key: str) -> Any:
    """stream_fn for the `nvidia/auto` model — tries each spec in `chain`
    in order, moving on to the next only on _UpstreamUnavailable (a
    pre-content failure); reports all of them failing as one error."""

    def stream_fn(_model: Model, context: Context, options: Any | None = None) -> Any:
        stream = create_assistant_message_event_stream()

        async def _run() -> None:
            errors: list[str] = []
            for spec in chain:
                try:
                    await _stream_one_model(spec, api_key, context, options, stream)
                    return
                except _UpstreamUnavailable as exc:
                    errors.append(f"{spec.model_id}: {exc}")
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
    120B/20B, MiniMax M3, and an `nvidia/auto` model that tries the first
    four (in that order) with automatic fallback on a pre-content
    failure.

    Returns: (default model, models_collection, {"base_url": ...}) —
    `model` (the first argument) is accepted for symmetry with the other
    single-model provider factories in this package but otherwise
    unused; the returned model is always DEFAULT_MODEL_ID (MiniMax M3).
    """
    import os

    key = api_key or os.environ.get("NVAPI_KEY")
    if not key:
        raise ValueError("NVAPI_KEY environment variable or api_key parameter required")

    models = MutableModels()
    catalog: list[Model] = []
    per_model_stream_fns: dict[str, Any] = {}
    default_model: Model | None = None

    for spec in _MODEL_SPECS:
        m = Model(
            id=spec.model_id,
            name=spec.name,
            api="openai-completions",
            provider="nvidia",
            context_window=spec.context_window,
            max_tokens=spec.max_tokens,
            input=["text", "image"] if spec.supports_images else ["text"],
        )
        catalog.append(m)
        per_model_stream_fns[spec.model_id] = _single_model_stream_fn(spec, key)
        if spec.model_id == DEFAULT_MODEL_ID:
            default_model = m

    assert default_model is not None, f"{DEFAULT_MODEL_ID!r} must be one of _MODEL_SPECS"

    fallback_chain = [spec for spec in _MODEL_SPECS if spec.in_fallback_chain]
    auto_model = Model(
        id=AUTO_MODEL_ID,
        name="Auto (Nemotron/GPT-OSS fallback chain)",
        api="openai-completions",
        provider="nvidia",
        context_window=min(spec.context_window for spec in fallback_chain),
        max_tokens=max(spec.max_tokens for spec in fallback_chain),
    )
    catalog.append(auto_model)
    per_model_stream_fns[AUTO_MODEL_ID] = _fallback_stream_fn(fallback_chain, key)

    # A single Provider only has one stream_fn, dispatched by the model
    # object passed to it — route by model.id to each model's own
    # pre-built stream_fn instead.
    def stream_fn(model_obj: Model, context: Context, options: Any | None = None) -> Any:
        return per_model_stream_fns[model_obj.id](model_obj, context, options)

    provider: Provider[Any] = Provider(
        id="nvidia",
        name="NVIDIA (Nemotron / GPT-OSS / MiniMax)",
        models=catalog,
        stream_fn=stream_fn,
    )
    models.set_provider(provider)

    return default_model, models, {"base_url": DEFAULT_BASE_URL}
