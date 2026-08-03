"""Core type definitions for the pi_agent_core package.

Mirrors ``packages/agent/src/types.ts`` from the TypeScript Pi agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Message,
    Model,
    ModelThinkingLevel,
    SimpleStreamOptions,
    TextContent,
    ThinkingBudgets,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
)

# --- Stream function ---


@runtime_checkable
class AssistantMessageEventStream(Protocol):
    """Async iterator of assistant message events with a final result accessor."""

    def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]: ...

    async def result(self) -> AssistantMessage: ...


StreamFn: TypeAlias = Callable[
    [Model, Context, "SimpleStreamOptions | None"],
    "AssistantMessageEventStream | Awaitable[AssistantMessageEventStream]",
]

# --- Execution and queue modes ---

ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]
QueueMode: TypeAlias = Literal["all", "one-at-a-time"]

# --- Tool call alias ---

AgentToolCall: TypeAlias = ToolCall

# --- Thinking level ---

ThinkingLevel: TypeAlias = ModelThinkingLevel

# --- Before/after tool call hooks ---


@dataclass
class BeforeToolCallResult:
    """Result returned from ``before_tool_call``.

    ``block=True`` prevents the tool from executing; the loop emits an error
    tool result instead. ``reason`` becomes the text shown in that error result.
    """

    block: bool | None = None
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    """Partial override returned from ``after_tool_call``.

    Omitted fields keep the original executed tool result values. No deep merge
    is performed for ``content``, ``details``, or ``usage``.
    """

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    """Context passed to ``before_tool_call``."""

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: "AgentContext"


@dataclass
class AfterToolCallContext:
    """Context passed to ``after_tool_call``."""

    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: "AgentToolResult[Any]"
    is_error: bool
    context: "AgentContext"


@dataclass
class ShouldStopAfterTurnContext:
    """Context passed to ``should_stop_after_turn``."""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: "AgentContext"
    new_messages: list["AgentMessage"]


@dataclass
class PrepareNextTurnContext(ShouldStopAfterTurnContext):
    """Context passed to ``prepare_next_turn``."""


@dataclass
class AgentLoopTurnUpdate:
    """Replacement runtime state used before starting another provider request."""

    context: "AgentContext | None" = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


# --- Agent tool result ---


@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool | None = None


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], None]

# --- Agent tool ---


@dataclass
class AgentTool(Tool):
    """Tool definition used by the agent runtime.

    Wraps a :class:`pi_ai.Tool` with an async execution callback, a human-readable
    label, an optional argument-preparation shim, and a per-tool execution mode
    override.
    """

    label: str = ""
    prepare_arguments: Callable[[Any], Any] | None = None
    execute: Callable[
        [str, Any, Any, AgentToolUpdateCallback],
        Awaitable[AgentToolResult],
    ] | None = None
    execution_mode: ToolExecutionMode | None = None


# --- Agent message ---

AgentMessage: TypeAlias = Message

# --- Agent context ---


@dataclass
class AgentContext:
    """Context snapshot passed into the low-level agent loop."""

    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None


# --- Agent state ---


class AgentState:
    """Public agent state.

    ``tools`` and ``messages`` copy the top-level array on assignment, mirroring
    the TypeScript accessor semantics.
    """

    def __init__(
        self,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
        is_streaming: bool = False,
        streaming_message: AgentMessage | None = None,
        pending_tool_calls: set[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        self._tools: list[AgentTool] = list(tools) if tools else []
        self._messages: list[AgentMessage] = list(messages) if messages else []
        self.system_prompt: str = system_prompt
        self.model: Model = model if model is not None else _DEFAULT_MODEL
        self.thinking_level: ThinkingLevel = thinking_level
        self.is_streaming: bool = is_streaming
        self.streaming_message: AgentMessage | None = streaming_message
        self.pending_tool_calls: set[str] = set(pending_tool_calls) if pending_tool_calls else set()
        self.error_message: str | None = error_message

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)


# --- Agent events ---


@dataclass
class AgentStartEvent:
    type: str = field(default="agent_start", init=False)


@dataclass
class AgentEndEvent:
    type: str = field(default="agent_end", init=False)
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class TurnStartEvent:
    type: str = field(default="turn_start", init=False)


@dataclass
class TurnEndEvent:
    type: str = field(default="turn_end", init=False)
    message: AgentMessage | None = None
    tool_results: list[ToolResultMessage] = field(default_factory=list)


@dataclass
class MessageStartEvent:
    type: str = field(default="message_start", init=False)
    message: AgentMessage | None = None


@dataclass
class MessageUpdateEvent:
    type: str = field(default="message_update", init=False)
    message: AgentMessage | None = None
    assistant_message_event: AssistantMessageEvent | None = None


@dataclass
class MessageEndEvent:
    type: str = field(default="message_end", init=False)
    message: AgentMessage | None = None


@dataclass
class ToolExecutionStartEvent:
    type: str = field(default="tool_execution_start", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None


@dataclass
class ToolExecutionUpdateEvent:
    type: str = field(default="tool_execution_update", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: Any = None
    partial_result: Any = None


@dataclass
class ToolExecutionEndEvent:
    type: str = field(default="tool_execution_end", init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    result: Any = None
    is_error: bool = False


AgentEvent: TypeAlias = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)

AgentEventSink: TypeAlias = Callable[[AgentEvent], Awaitable[None] | None]

# --- Agent loop config ---


ConvertToLlmFn: TypeAlias = Callable[[list[AgentMessage]], "list[Message] | Awaitable[list[Message]]"]
TransformContextFn: TypeAlias = Callable[
    [list[AgentMessage], Any],
    Awaitable[list[AgentMessage]],
]
GetApiKeyFn: TypeAlias = Callable[[str], "str | Awaitable[str | None] | None"]
ShouldStopAfterTurnFn: TypeAlias = Callable[[ShouldStopAfterTurnContext], "bool | Awaitable[bool]"]
PrepareNextTurnFn: TypeAlias = Callable[
    [PrepareNextTurnContext],
    "AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None]",
]
PrepareNextTurnSignalFn: TypeAlias = Callable[
    [Any],
    "AgentLoopTurnUpdate | None | Awaitable[AgentLoopTurnUpdate | None]",
]
GetMessagesFn: TypeAlias = Callable[[], Awaitable[list[AgentMessage]]]
BeforeToolCallFn: TypeAlias = Callable[
    [BeforeToolCallContext, Any],
    Awaitable[BeforeToolCallResult | None],
]
AfterToolCallFn: TypeAlias = Callable[
    [AfterToolCallContext, Any],
    Awaitable[AfterToolCallResult | None],
]


@dataclass
class AgentLoopConfig(SimpleStreamOptions):
    """Configuration for the agent loop, extending :class:`SimpleStreamOptions`."""

    model: Model = field(default_factory=lambda: _DEFAULT_MODEL)
    convert_to_llm: ConvertToLlmFn = field(default=lambda: _default_convert_to_llm)
    transform_context: TransformContextFn | None = None
    get_api_key: GetApiKeyFn | None = None
    should_stop_after_turn: ShouldStopAfterTurnFn | None = None
    prepare_next_turn: PrepareNextTurnFn | None = None
    get_steering_messages: GetMessagesFn | None = None
    get_follow_up_messages: GetMessagesFn | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: BeforeToolCallFn | None = None
    after_tool_call: AfterToolCallFn | None = None


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [m for m in messages if m.role in ("user", "assistant", "toolResult")]


_DEFAULT_MODEL = Model(
    id="unknown",
    name="unknown",
    api="unknown",
    provider="unknown",
    base_url="",
    reasoning=False,
    context_window=0,
    max_tokens=0,
)

EMPTY_USAGE = Usage()

__all__ = [
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentEventSink",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentStartEvent",
    "AgentState",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "AssistantMessageEventStream",
    "AfterToolCallContext",
    "AfterToolCallFn",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "BeforeToolCallFn",
    "BeforeToolCallResult",
    "ConvertToLlmFn",
    "GetApiKeyFn",
    "GetMessagesFn",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "PrepareNextTurnContext",
    "PrepareNextTurnFn",
    "PrepareNextTurnSignalFn",
    "QueueMode",
    "ShouldStopAfterTurnContext",
    "ShouldStopAfterTurnFn",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionEndEvent",
    "ToolExecutionMode",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TransformContextFn",
    "TurnEndEvent",
    "TurnStartEvent",
    "EMPTY_USAGE",
]
