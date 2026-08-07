"""Stateful Agent wrapping the low-level agent loop.

Mirrors ``packages/agent/src/agent.ts``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_ai import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    StopReason,
    TextContent,
    ThinkingBudgets,
    Transport,
    UserMessage,
)

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .stream_fn import get_default_stream_fn
from .types import (
    _DEFAULT_MODEL,
    EMPTY_USAGE,
    AfterToolCallFn,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallFn,
    ConvertToLlmFn,
    GetApiKeyFn,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    PrepareNextTurnFn,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionEndEvent,
    ToolExecutionMode,
    ToolExecutionStartEvent,
    TransformContextFn,
    TurnEndEvent,
)

__all__ = ["Agent", "AgentOptions", "create_mutable_agent_state"]


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [m for m in messages if m.role in ("user", "assistant", "toolResult")]


@dataclass
class AgentOptions:
    """Options for constructing an :class:`Agent`."""

    initial_state: PartialAgentState | None = None
    convert_to_llm: ConvertToLlmFn | None = None
    transform_context: TransformContextFn | None = None
    stream_fn: StreamFn | None = None
    get_api_key: GetApiKeyFn | None = None
    on_payload: Callable[[Any, Any], Any] | None = None
    on_response: Callable[[Any, Any], None] | None = None
    before_tool_call: BeforeToolCallFn | None = None
    after_tool_call: AfterToolCallFn | None = None
    prepare_next_turn: PrepareNextTurnFn | None = None
    prepare_next_turn_with_context: PrepareNextTurnFn | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport = "auto"
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode = "parallel"


class _PartialAgentState:
    """Subset of AgentState fields that callers may initialize.

    See :class:`AgentOptions.initial_state`.
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.thinking_level = thinking_level
        self.tools = tools
        self.messages = messages


PartialAgentState = _PartialAgentState


def create_mutable_agent_state(
    initial_state: _PartialAgentState | None = None,
) -> AgentState:
    """Factory creating a fresh :class:`AgentState` with copying accessors."""
    initial = initial_state or _PartialAgentState()
    return AgentState(
        system_prompt=initial.system_prompt,
        model=initial.model if initial.model is not None else _DEFAULT_MODEL,
        thinking_level=initial.thinking_level,
        tools=initial.tools,
        messages=initial.messages,
        is_streaming=False,
        streaming_message=None,
        pending_tool_calls=None,
        error_message=None,
    )


class _PendingMessageQueue:
    """Queue of pending AgentMessages drained per a :data:`QueueMode`."""

    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


@dataclass
class _ActiveRun:
    """Bookkeeping for an in-flight agent run."""

    task: asyncio.Future[None] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    aborted: bool = False


async def _maybe_await(value: Any) -> Any:
    """Await value if it's awaitable, otherwise return it directly."""
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


class Agent:
    """Stateful wrapper around the low-level agent loop.

    ``Agent`` owns the current transcript, emits lifecycle events, executes
    tools, and exposes queueing APIs for steering and follow-up messages.
    """

    def __init__(self, options: AgentOptions) -> None:
        opts = options or AgentOptions()
        self._state: AgentState = create_mutable_agent_state(opts.initial_state)
        self.convert_to_llm: ConvertToLlmFn = opts.convert_to_llm or _default_convert_to_llm
        self.transform_context: TransformContextFn | None = opts.transform_context
        self.stream_function: StreamFn = opts.stream_fn or get_default_stream_fn()
        self.get_api_key: GetApiKeyFn | None = opts.get_api_key
        self.on_payload: Callable[[Any, Any], Any] | None = opts.on_payload
        self.on_response: Callable[[Any, Any], None] | None = opts.on_response
        self.before_tool_call: BeforeToolCallFn | None = opts.before_tool_call
        self.after_tool_call: AfterToolCallFn | None = opts.after_tool_call
        self.prepare_next_turn: PrepareNextTurnFn | None = opts.prepare_next_turn
        self.prepare_next_turn_with_context: PrepareNextTurnFn | None = opts.prepare_next_turn_with_context
        self._steering_queue = _PendingMessageQueue(opts.steering_mode)
        self._follow_up_queue = _PendingMessageQueue(opts.follow_up_mode)
        self.session_id: str | None = opts.session_id
        self.thinking_budgets: ThinkingBudgets | None = opts.thinking_budgets
        self.transport: Transport = opts.transport
        self.max_retry_delay_ms: int | None = opts.max_retry_delay_ms
        self.tool_execution: ToolExecutionMode = opts.tool_execution
        self._listeners: list[Callable[[AgentEvent, Any], Awaitable[None] | None]] = []
        self._active_run: _ActiveRun | None = None

    # --- Subscription ---

    def on_event(self, listener: Callable[[AgentEvent, Any], Awaitable[None] | None]) -> Callable[()]:
        """Register a listener and return an unsubscribe function."""
        self._listeners.append(listener)
        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if not removed:
                removed = True
                with contextlib.suppress(ValueError):
                    self._listeners.remove(listener)

        return unsubscribe

    # Subscribe alias matching the TypeScript API
    subscribe = on_event

    # --- State access ---

    @property
    def state(self) -> AgentState:
        return self._state

    # --- Queue mode accessors ---

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    # --- Queue mutation ---

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn finishes."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self.clear_steering_queue()
        self.clear_follow_up_queue()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # --- Run lifecycle ---

    @property
    def is_active(self) -> bool:
        return self._active_run is not None

    @property
    def aborted(self) -> bool:
        return self._active_run.aborted if self._active_run else False

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_run is not None:
            self._active_run.aborted = True
            self._active_run.abort_event.set()

    async def wait_for_idle(self) -> None:
        """Resolve when the current run has finished."""
        if self._active_run is not None:
            await self._active_run.task

    def reset(self) -> None:
        """Clear transcript state, runtime state, and queued messages."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    # --- Public actions ---

    async def start(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages.

        Alias for :meth:`prompt`.
        """
        await self.prompt(message, images)

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_run(self) -> None:
        """Continue from the current transcript.

        The last message must be a user or tool-result message.
        """
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        if not self._state.messages:
            raise RuntimeError("No messages to continue from")

        last_message = self._state.messages[-1]
        if last_message.role == "assistant":
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(queued_steering, skip_initial_steering_poll=True)
                return
            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return
            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # Backwards-compatible alias matching TS naming
    continue_ = continue_run

    def set_model(self, model: Model) -> None:
        """Set the active model used for future turns."""
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set the requested reasoning level for future turns."""
        self._state.thinking_level = level

    # --- Internal helpers ---

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input
        if isinstance(input, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [UserMessage(content=content, timestamp=int(time.time() * 1000))]
        return [input]

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
                self._process_events,
                signal,
                self.stream_function,
            )
        )

    async def _run_continuation(self) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                self._process_events,
                signal,
                self.stream_function,
            )
        )

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _create_loop_config(self, *, skip_initial_steering_poll: bool = False) -> Any:
        from .types import AgentLoopConfig

        skip_steering = skip_initial_steering_poll

        async def _prepare_next_turn(context: PrepareNextTurnContext) -> AgentLoopTurnUpdate | None:
            if self.prepare_next_turn_with_context is not None:
                return await _maybe_await(self.prepare_next_turn_with_context(context, self._active_run))
            if self.prepare_next_turn is not None:
                return await _maybe_await(self.prepare_next_turn(self._active_run))
            return None

        async def _get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_steering
            if skip_steering:
                skip_steering = False
                return []
            return self._steering_queue.drain()

        async def _get_follow_up_messages() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        config = AgentLoopConfig(
            model=self._state.model,
            reasoning=None if self._state.thinking_level == "off" else self._state.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            transport=self.transport,
            thinking_budgets=self.thinking_budgets,
            max_retry_delay_ms=self.max_retry_delay_ms,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            prepare_next_turn=(
                _prepare_next_turn if (self.prepare_next_turn or self.prepare_next_turn_with_context) else None
            ),
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=_get_steering_messages,
            get_follow_up_messages=_get_follow_up_messages,
        )
        return config

    async def _run_with_lifecycle(self, executor: Callable[[Any], Awaitable[None]]) -> None:
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        active_run = _ActiveRun()
        self._active_run = active_run
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(active_run.abort_event)
        except Exception as error:
            await self._handle_run_failure(error, active_run.aborted)
        finally:
            self._finish_run()

    async def _handle_run_failure(self, error: BaseException, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.id,
            usage=EMPTY_USAGE,
            stop_reason=StopReason.ABORTED if aborted else StopReason.ERROR,
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        await self._process_events(MessageStartEvent(message=failure_message))
        await self._process_events(MessageEndEvent(message=failure_message))
        await self._process_events(TurnEndEvent(message=failure_message, tool_results=[]))
        await self._process_events(AgentEndEvent(messages=[failure_message]))

    def _finish_run(self) -> None:
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        if self._active_run is not None and not self._active_run.task.done():
            self._active_run.task.set_result(None)
        self._active_run = None

    async def _process_events(self, event: AgentEvent) -> None:
        """Reduce internal state for a loop event, then await listeners."""
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent)):
            self._state.streaming_message = event.message
        elif isinstance(event, MessageEndEvent):
            self._state.streaming_message = None
            if event.message is not None:
                self._state.messages.append(event.message)
        elif isinstance(event, ToolExecutionStartEvent):
            self._state.pending_tool_calls.add(event.tool_call_id)
        elif isinstance(event, ToolExecutionEndEvent):
            self._state.pending_tool_calls.discard(event.tool_call_id)
        elif isinstance(event, TurnEndEvent):
            if isinstance(event.message, AssistantMessage) and event.message.error_message is not None:
                self._state.error_message = event.message.error_message
        elif isinstance(event, AgentEndEvent):
            self._state.streaming_message = None

        if self._active_run is None:
            raise RuntimeError("Agent listener invoked outside active run")
        signal = self._active_run.abort_event
        for listener in list(self._listeners):
            result = listener(event, signal)
            if asyncio.iscoroutine(result):
                await result
