"""Agent loop that works with AgentMessage throughout.

Mirrors ``packages/agent/src/agent-loop.ts``.

Transforms to ``Message[]`` only at the LLM call boundary. The full streaming
and tool-execution machinery is intentionally simplified here: this module ports
the structural loop and event sequencing, using a pluggable ``StreamFn``. Tests
provide a faux stream function rather than a real LLM provider.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_ai import (
    AssistantMessage,
    Context,
    StopReason,
    TextContent,
    ToolResultMessage,
)

from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentMessage,
    AgentToolCall,
    AgentToolResult,
    AfterToolCallContext,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
    TurnStartEvent,
)

__all__ = ["run_agent_loop", "run_agent_loop_continue", "AgentEventSink"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


@dataclass
class _ExecutedToolBatch:
    messages: list[ToolResultMessage]
    terminate: bool


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """Start an agent loop with a new prompt message.

    The prompt is added to the context and events are emitted for it.
    """
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await _emit(emit, AgentEndEvent()) if False else None  # placeholder no-op
    from .types import AgentStartEvent

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for prompt in prompts:
        await _emit(emit, MessageStartEvent(message=prompt))
        await _emit(emit, MessageEndEvent(message=prompt))

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """Continue an agent loop from the current context without adding a new message.

    The last message must convert to a ``user`` or ``toolResult`` message via
    ``convert_to_llm``. If it doesn't, the LLM provider will reject the request.
    """
    if not context.messages:
        raise RuntimeError("Cannot continue: no messages in context")

    if context.messages[-1].role == "assistant":
        raise RuntimeError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    from .types import AgentStartEvent

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())

    await _run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: Any | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    """Main loop logic shared by run_agent_loop and run_agent_loop_continue."""
    current_context = initial_context
    config = initial_config
    first_turn = True
    # Check for steering messages at start (user may have typed while waiting)
    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages is not None:
        pending_messages = await config.get_steering_messages()

    # Outer loop: continues when queued follow-up messages arrive after agent would stop
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, TurnStartEvent())
            else:
                first_turn = False

            # Process pending messages (inject before next assistant response)
            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, MessageStartEvent(message=message))
                    await _emit(emit, MessageEndEvent(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(
                current_context, config, signal, emit, stream_function
            )
            new_messages.append(message)

            if message.stop_reason in (StopReason.ERROR, StopReason.ABORTED):
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if getattr(c, "type", None) == "toolCall"]

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                if message.stop_reason == StopReason.LENGTH:
                    executed_batch = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    executed_batch = await _execute_tool_calls(
                        current_context, message, config, signal, emit
                    )
                tool_results.extend(executed_batch.messages)
                has_more_tool_calls = not executed_batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))

            next_turn_context = PrepareNextTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )
            if config.prepare_next_turn is not None:
                next_turn_snapshot = await _maybe_await(config.prepare_next_turn(next_turn_context))
                if next_turn_snapshot is not None:
                    if next_turn_snapshot.context is not None:
                        current_context = next_turn_snapshot.context
                    config = _with_turn_update(config, next_turn_snapshot)

            if config.should_stop_after_turn is not None:
                should_stop = await _maybe_await(
                    config.should_stop_after_turn(
                        ShouldStopAfterTurnContext(
                            message=message,
                            tool_results=tool_results,
                            context=current_context,
                            new_messages=new_messages,
                        )
                    )
                )
                if should_stop:
                    await _emit(emit, AgentEndEvent(messages=new_messages))
                    return

            if config.get_steering_messages is not None:
                pending_messages = await config.get_steering_messages()

        # Agent would stop here. Check for follow-up messages.
        follow_up_messages: list[AgentMessage] = []
        if config.get_follow_up_messages is not None:
            follow_up_messages = await config.get_follow_up_messages()
        if follow_up_messages:
            # Set as pending so inner loop processes them
            pending_messages = follow_up_messages
            continue

        # No more messages, exit
        break

    await _emit(emit, AgentEndEvent(messages=new_messages))


def _with_turn_update(config: AgentLoopConfig, update: Any) -> AgentLoopConfig:
    """Return a new config with updated model/reasoning from a turn update."""
    import dataclasses

    kwargs: dict[str, Any] = {}
    if update.model is not None:
        kwargs["model"] = update.model
    if update.thinking_level is not None:
        kwargs["reasoning"] = None if update.thinking_level == "off" else update.thinking_level
    return dataclasses.replace(config, **kwargs)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    result = emit(event)
    if asyncio.iscoroutine(result):
        await result


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Any | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """Stream an assistant response from the LLM.

    Applies ``transform_context`` then ``convert_to_llm`` before calling the
    stream function. Handles both faux/real stream functions that return an
    async iterator of events.
    """
    messages = context.messages
    if config.transform_context is not None:
        messages = await config.transform_context(messages, signal)

    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    resolved_api_key = None
    if config.get_api_key is not None:
        resolved_api_key = await _maybe_await(config.get_api_key(config.model.provider))
    if not resolved_api_key:
        resolved_api_key = config.api_key

    response = await _maybe_await(stream_function(config.model, llm_context, config))

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)  # type: ignore[arg-type]
            added_partial = True
            await _emit(emit, MessageStartEvent(message=partial_message))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(message=partial_message, assistant_message_event=event),
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await _emit(emit, MessageStartEvent(message=final_message))
            await _emit(emit, MessageEndEvent(message=final_message))
            return final_message

    # Stream exhausted without an explicit done/error event: finalize.
    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, MessageStartEvent(message=final_message))
    await _emit(emit, MessageEndEvent(message=final_message))
    return final_message


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[AgentToolCall],
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )
        result = _create_error_tool_result(
            f'Tool call "{tool_call.name}" was not executed: the response hit the output token '
            "limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
        )
        await _emit(
            emit,
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                is_error=True,
            ),
        )
        tool_result_message = _create_tool_result_message(tool_call, result, is_error=True)
        await _emit(emit, MessageStartEvent(message=tool_result_message))
        await _emit(emit, MessageEndEvent(message=tool_result_message))
        messages.append(tool_result_message)
    return _ExecutedToolBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Any | None,
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
    """Execute tool calls sequentially.

    The TypeScript version supports parallel execution; this port uses the
    sequential path for all calls for simplicity. Tool execution mode parity
    can be layered in later without changing the event sequence.
    """
    tool_calls = [c for c in assistant_message.content if getattr(c, "type", None) == "toolCall"]
    return await _execute_tool_calls_sequential(
        current_context, assistant_message, tool_calls, config, signal, emit
    )


def _find_tool(tools: list[Any] | None, name: str) -> Any | None:
    if tools is None:
        return None
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _is_aborted(signal: Any | None) -> bool:
    if signal is None:
        return False
    is_set = getattr(signal, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if isinstance(is_set, bool):
        return is_set
    return False


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: Any | None,
    emit: AgentEventSink,
) -> _ExecutedToolBatch:
    finalized_calls: list[dict[str, Any]] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            ),
        )

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if preparation["kind"] == "immediate":
            finalized = {
                "tool_call": tool_call,
                "result": preparation["result"],
                "is_error": preparation["is_error"],
            }
        else:
            executed = await _execute_prepared_tool_call(preparation, signal, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )

        await _emit(
            emit,
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=finalized["result"],
                is_error=finalized["is_error"],
            ),
        )
        tool_result_message = _create_tool_result_message(
            finalized["tool_call"], finalized["result"], is_error=finalized["is_error"]
        )
        await _emit(emit, MessageStartEvent(message=tool_result_message))
        await _emit(emit, MessageEndEvent(message=tool_result_message))
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if _is_aborted(signal):
            break

    return _ExecutedToolBatch(
        messages=messages,
        terminate=_should_terminate_tool_batch(finalized_calls),
    )


def _should_terminate_tool_batch(finalized_calls: list[dict[str, Any]]) -> bool:
    if not finalized_calls:
        return False
    return all(finalized["result"].terminate is True for finalized in finalized_calls)


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: Any | None,
) -> dict[str, Any]:
    tool = _find_tool(current_context.tools, tool_call.name)
    if tool is None:
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(f"Tool {tool_call.name} not found"),
            "is_error": True,
        }

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = prepared_tool_call.arguments
        if config.before_tool_call is not None:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared_tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if _is_aborted(signal):
                return {
                    "kind": "immediate",
                    "result": _create_error_tool_result("Operation aborted"),
                    "is_error": True,
                }
            if before_result is not None and before_result.block:
                reason = before_result.reason or "Tool execution was blocked"
                return {
                    "kind": "immediate",
                    "result": _create_error_tool_result(reason),
                    "is_error": True,
                }
        if _is_aborted(signal):
            return {
                "kind": "immediate",
                "result": _create_error_tool_result("Operation aborted"),
                "is_error": True,
            }
        return {
            "kind": "prepared",
            "tool_call": prepared_tool_call,
            "tool": tool,
            "args": validated_args,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(str(error)),
            "is_error": True,
        }


def _prepare_tool_call_arguments(tool: Any, tool_call: AgentToolCall) -> AgentToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared = tool.prepare_arguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return AgentToolCall(id=tool_call.id, name=tool_call.name, arguments=prepared)


async def _execute_prepared_tool_call(
    prepared: dict[str, Any],
    signal: Any | None,
    emit: AgentEventSink,
) -> dict[str, Any]:
    tool = prepared["tool"]
    tool_call = prepared["tool_call"]
    args = prepared["args"]
    try:
        if tool.execute is None:
            return {
                "result": _create_error_tool_result("Tool has no execute callback"),
                "is_error": True,
            }
        result = await tool.execute(tool_call.id, args, signal, None)
        return {"result": result, "is_error": False}
    except Exception as error:  # noqa: BLE001
        return {
            "result": _create_error_tool_result(str(error)),
            "is_error": True,
        }


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: dict[str, Any],
    executed: dict[str, Any],
    config: AgentLoopConfig,
    signal: Any | None,
) -> dict[str, Any]:
    result = executed["result"]
    is_error = executed["is_error"]

    if config.after_tool_call is not None:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared["tool_call"],
                        args=prepared["args"],
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if after_result is not None:
                if after_result.content is not None:
                    result.content = after_result.content
                if after_result.details is not None:
                    result.details = after_result.details
                if after_result.usage is not None:
                    result.usage = after_result.usage
                if after_result.terminate is not None:
                    result.terminate = after_result.terminate
                if after_result.is_error is not None:
                    is_error = after_result.is_error
        except Exception as error:  # noqa: BLE001
            result = _create_error_tool_result(str(error))
            is_error = True

    return {
        "tool_call": prepared["tool_call"],
        "result": result,
        "is_error": is_error,
    }


def _create_tool_result_message(
    tool_call: AgentToolCall,
    result: AgentToolResult,
    *,
    is_error: bool,
) -> ToolResultMessage:
    msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content or [],
        details=result.details,
        usage=result.usage,
        is_error=is_error,
        timestamp=_now_ms(),
    )
    if result.added_tool_names:
        msg.added_tool_names = result.added_tool_names
    return msg
