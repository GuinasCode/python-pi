"""Faux provider for testing.

Mirrors packages/ai/src/providers/faux.ts.
Provides a deterministic, no-network provider that replays queued responses.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Callable
from typing import Any

from pi_ai import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
)
from pi_ai.event_stream import AssistantMessageEventStream, create_assistant_message_event_stream
from pi_ai.models import Provider

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 8

DEFAULT_USAGE = Usage()
DEFAULT_USAGE.cost = UsageCost()


def _random_id(prefix: str) -> str:
    return f"{prefix}:{int(time.time() * 1000)}:{id(object())}"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(name: str, arguments: dict[str, Any], *, id: str | None = None) -> ToolCall:
    return ToolCall(id=id or _random_id("tool"), name=name, arguments=arguments)


def faux_assistant_message(
    content: str | TextContent | ThinkingContent | ToolCall | list[Any],
    *,
    stop_reason: StopReason = StopReason.STOP,
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    if isinstance(content, str):
        blocks: list[Any] = [faux_text(content)]
    elif isinstance(content, list):
        blocks = content
    else:
        blocks = [content]
    return AssistantMessage(
        content=blocks,
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=copy.deepcopy(DEFAULT_USAGE),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp or int(time.time() * 1000),
    )


FauxContentBlock = TextContent | ThinkingContent | ToolCall
FauxResponseFactory = Callable[[Context, StreamOptions | None, dict[str, int], Model], AssistantMessage]
FauxResponseStep = AssistantMessage | FauxResponseFactory


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _content_to_text(content: str | list[TextContent | ImageContent]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(content: list[TextContent | ThinkingContent | ToolCall]) -> str:
    parts: list[str] = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "thinking":
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{json.dumps(block.arguments)}")
    return "\n".join(parts)


def _tool_result_to_text(message: ToolResultMessage) -> str:
    parts = [message.tool_name]
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _message_to_text(message: Message) -> str:
    if message.role == "user":
        return _content_to_text(message.content)  # type: ignore[arg-type]
    if message.role == "assistant":
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)  # type: ignore[arg-type]


def _serialize_context(context: Context) -> str:
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for msg in context.messages:
        parts.append(f"{msg.role}:{_message_to_text(msg)}")
    if context.tools:
        parts.append(f"tools:{json.dumps([t.name for t in context.tools])}")
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    idx = 0
    while idx < length and a[idx] == b[idx]:
        idx += 1
    return idx


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: StreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(_assistant_content_to_text(message.content))
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = options.session_id if options else None

    if session_id and (options.cache_retention if options else None) != "none":
        previous = prompt_cache.get(session_id)
        if previous:
            cached_chars = _common_prefix_length(previous, prompt_text)
            cache_read = _estimate_tokens(previous[:cached_chars])
            cache_write = _estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_write = prompt_tokens
        prompt_cache[session_id] = prompt_text

    message.usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
        cost=UsageCost(),
    )
    return message


def _split_by_token_size(text: str, min_size: int, max_size: int) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    idx = 0
    import random

    while idx < len(text):
        token_size = random.randint(min_size, max_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[idx : idx + char_size])
        idx += char_size
    return chunks


def _clone_message(message: AssistantMessage, api: str, provider: str, model_id: str) -> AssistantMessage:
    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    if cloned.timestamp == 0:
        cloned.timestamp = int(time.time() * 1000)
    if cloned.usage is None:
        cloned.usage = copy.deepcopy(DEFAULT_USAGE)
    return cloned


def _create_error_message(error: Exception, api: str, provider: str, model_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=copy.deepcopy(DEFAULT_USAGE),
        stop_reason=StopReason.ERROR,
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
) -> None:
    """Stream message content as deltas, simulating real LLM streaming."""
    from pi_ai import (
        DoneEvent,
        ErrorEvent,
        StartEvent,
        TextDeltaEvent,
        TextEndEvent,
        TextStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
        ToolCallStartEvent,
    )

    partial = copy.deepcopy(message)
    partial.content = []
    partial.stop_reason = StopReason.PENDING

    stream.push(StartEvent(partial=copy.deepcopy(partial)))

    for idx, block in enumerate(message.content):
        if block.type == "thinking":
            partial.content.append(ThinkingContent(thinking=""))
            stream.push(ThinkingStartEvent(content_index=idx, partial=copy.deepcopy(partial)))
            for chunk in _split_by_token_size(block.thinking, min_token_size, max_token_size):
                if tokens_per_second and tokens_per_second > 0:
                    await asyncio.sleep(_estimate_tokens(chunk) / tokens_per_second)
                partial.content[idx].thinking += chunk
                stream.push(ThinkingDeltaEvent(content_index=idx, delta=chunk, partial=copy.deepcopy(partial)))
            stream.push(ThinkingEndEvent(content_index=idx, content=block.thinking, partial=copy.deepcopy(partial)))

        elif block.type == "text":
            partial.content.append(TextContent(text=""))
            stream.push(TextStartEvent(content_index=idx, partial=copy.deepcopy(partial)))
            for chunk in _split_by_token_size(block.text, min_token_size, max_token_size):
                if tokens_per_second and tokens_per_second > 0:
                    await asyncio.sleep(_estimate_tokens(chunk) / tokens_per_second)
                partial.content[idx].text += chunk
                stream.push(TextDeltaEvent(content_index=idx, delta=chunk, partial=copy.deepcopy(partial)))
            stream.push(TextEndEvent(content_index=idx, content=block.text, partial=copy.deepcopy(partial)))

        elif block.type == "toolCall":
            partial.content.append(ToolCall(id=block.id, name=block.name, arguments={}))
            stream.push(ToolCallStartEvent(content_index=idx, partial=copy.deepcopy(partial)))
            args_json = json.dumps(block.arguments)
            for chunk in _split_by_token_size(args_json, min_token_size, max_token_size):
                if tokens_per_second and tokens_per_second > 0:
                    await asyncio.sleep(_estimate_tokens(chunk) / tokens_per_second)
                stream.push(ToolCallDeltaEvent(content_index=idx, delta=chunk, partial=copy.deepcopy(partial)))
            partial.content[idx].arguments = block.arguments
            stream.push(ToolCallEndEvent(content_index=idx, tool_call=block, partial=copy.deepcopy(partial)))

    if message.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
        stream.push(ErrorEvent(reason=message.stop_reason.value, error=message))  # type: ignore[arg-type]
        stream.end(message)
        return

    done_reason = (
        message.stop_reason.value
        if message.stop_reason in (StopReason.STOP, StopReason.LENGTH, StopReason.TOOL_USE)
        else "stop"
    )
    stream.push(DoneEvent(reason=done_reason, message=message))  # type: ignore[arg-type]
    stream.end(message)


class FauxProviderHandle:
    """Handle to a registered faux provider."""

    def __init__(
        self,
        provider: Provider,
        api: str,
        models: list[Model],
        state: dict[str, int],
        set_responses_fn: Callable[[list[FauxResponseStep]], None],
        append_responses_fn: Callable[[list[FauxResponseStep]], None],
        get_pending_count_fn: Callable[[], int],
    ) -> None:
        self.provider = provider
        self.api = api
        self.models = models
        self.state = state
        self._set_responses = set_responses_fn
        self._append_responses = append_responses_fn
        self._get_pending_count = get_pending_count_fn

    def get_model(self, model_id: str | None = None) -> Model | None:
        if model_id is None:
            return self.models[0] if self.models else None
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._set_responses(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._append_responses(responses)

    @property
    def pending_response_count(self) -> int:
        return self._get_pending_count()


def faux_provider(
    *,
    api: str | None = None,
    provider_id: str | None = None,
    models: list[dict[str, Any]] | None = None,
    tokens_per_second: float | None = None,
    token_size_min: int = DEFAULT_MIN_TOKEN_SIZE,
    token_size_max: int = DEFAULT_MAX_TOKEN_SIZE,
) -> FauxProviderHandle:
    """Create a faux provider for testing.

    Args:
        api: API name (default: random).
        provider_id: Provider ID (default: "faux").
        models: List of model definitions.
        tokens_per_second: Simulated streaming speed (None = instant).
        token_size_min: Minimum token chunk size for streaming.
        token_size_max: Maximum token chunk size for streaming.
    """
    api_name = api or _random_id(DEFAULT_API)
    provider_name = provider_id or DEFAULT_PROVIDER
    min_ts = max(1, min(token_size_min, token_size_max))
    max_ts = max(min_ts, token_size_max)

    pending_responses: list[FauxResponseStep] = []
    state: dict[str, int] = {"callCount": 0}
    prompt_cache: dict[str, str] = {}

    model_defs = models or [
        {
            "id": DEFAULT_MODEL_ID,
            "name": DEFAULT_MODEL_NAME,
            "reasoning": False,
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            "context_window": 128000,
            "max_tokens": 16384,
        }
    ]

    model_list = [
        Model(
            id=d["id"],
            name=d.get("name", d["id"]),
            api=api_name,
            provider=provider_name,
            base_url=DEFAULT_BASE_URL,
            reasoning=d.get("reasoning", False),
            input=d.get("input", ["text", "image"]),
            cost=ModelCost(
                input=d.get("cost", {}).get("input", 0),
                output=d.get("cost", {}).get("output", 0),
                cache_read=d.get("cost", {}).get("cache_read", 0),
                cache_write=d.get("cost", {}).get("cache_write", 0),
            ),
            context_window=d.get("context_window", 128000),
            max_tokens=d.get("max_tokens", 16384),
        )
        for d in model_defs
    ]

    def _stream(
        request_model: Model,
        context: Context,
        stream_options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = create_assistant_message_event_stream()
        step = pending_responses.pop(0) if pending_responses else None
        state["callCount"] += 1

        async def _run() -> None:
            from pi_ai import ErrorEvent

            try:
                if step is None:
                    msg = _create_error_message(
                        ValueError("No more faux responses queued"),
                        api_name,
                        provider_name,
                        request_model.id,
                    )
                    msg = _with_usage_estimate(msg, context, stream_options, prompt_cache)
                    outer.push(ErrorEvent(reason="error", error=msg))
                    outer.end(msg)
                    return

                resolved = step(context, stream_options, state, request_model) if callable(step) else step

                message = _clone_message(resolved, api_name, provider_name, request_model.id)
                message = _with_usage_estimate(message, context, stream_options, prompt_cache)
                await _stream_with_deltas(outer, message, min_ts, max_ts, tokens_per_second)
            except Exception as exc:
                msg = _create_error_message(exc, api_name, provider_name, request_model.id)
                outer.push(ErrorEvent(reason="error", error=msg))
                outer.end(msg)

        _run_task = asyncio.ensure_future(_run())  # noqa: RUF006
        return outer

    provider = Provider(
        id=provider_name,
        name="Faux",
        models=model_list,
        stream_fn=_stream,  # type: ignore[arg-type]
    )

    def set_responses(responses: list[FauxResponseStep]) -> None:
        nonlocal pending_responses
        pending_responses = list(responses)

    def append_responses(responses: list[FauxResponseStep]) -> None:
        pending_responses.extend(responses)

    def get_pending_count() -> int:
        return len(pending_responses)

    return FauxProviderHandle(
        provider=provider,
        api=api_name,
        models=model_list,
        state=state,
        set_responses_fn=set_responses,
        append_responses_fn=append_responses,
        get_pending_count_fn=get_pending_count,
    )


__all__ = [
    "FauxContentBlock",
    "FauxProviderHandle",
    "FauxResponseFactory",
    "FauxResponseStep",
    "faux_assistant_message",
    "faux_provider",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
]
