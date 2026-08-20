"""Tests for pi_coding_agent.extensions.runner event emission."""

from __future__ import annotations

import asyncio
from typing import Any

from pi_ai import TextContent
from pi_coding_agent.extensions.events import (
    ExtensionContext,
    ToolCallEvent,
    ToolCallEventResult,
    ToolResultEvent,
    ToolResultEventResult,
)
from pi_coding_agent.extensions.runner import ExtensionRunner


def _runner_with_handlers(handlers: dict[str, list[Any]]) -> ExtensionRunner:
    """A runner pre-loaded with handlers as if one extension ("fake.py")
    had registered all of them, without touching the filesystem."""
    runner = ExtensionRunner("/tmp/does-not-matter")
    runner._handlers = {event_name: [("fake.py", h) for h in hs] for event_name, hs in handlers.items()}
    return runner


class TestEmitNotifyOnly:
    def test_every_handler_runs_in_order(self) -> None:
        calls: list[str] = []

        def h1(_event: Any, _ctx: Any) -> None:
            calls.append("h1")

        def h2(_event: Any, _ctx: Any) -> None:
            calls.append("h2")

        runner = _runner_with_handlers({"agent_start": [h1, h2]})
        asyncio.run(runner.emit("agent_start", object()))
        assert calls == ["h1", "h2"]

    def test_async_handler_is_awaited(self) -> None:
        calls: list[str] = []

        async def h1(_event: Any, _ctx: Any) -> None:
            calls.append("async")

        runner = _runner_with_handlers({"agent_start": [h1]})
        asyncio.run(runner.emit("agent_start", object()))
        assert calls == ["async"]

    def test_handler_exception_is_recorded_not_raised(self) -> None:
        def broken(_event: Any, _ctx: Any) -> None:
            raise RuntimeError("boom")

        called: list[str] = []

        def ok(_event: Any, _ctx: Any) -> None:
            called.append("ok")

        runner = _runner_with_handlers({"agent_start": [broken, ok]})
        asyncio.run(runner.emit("agent_start", object()))
        assert called == ["ok"]
        assert len(runner.get_extensions().errors) == 1
        assert "boom" in runner.get_extensions().errors[0].error

    def test_context_passed_to_handler_has_cwd(self) -> None:
        seen: list[ExtensionContext] = []

        def h(_event: Any, ctx: ExtensionContext) -> None:
            seen.append(ctx)

        runner = ExtensionRunner("/some/cwd")
        runner._handlers = {"agent_start": [("fake.py", h)]}
        asyncio.run(runner.emit("agent_start", object()))
        assert seen[0].cwd == "/some/cwd"

    def test_no_handlers_registered_is_a_noop(self) -> None:
        runner = ExtensionRunner("/tmp")
        asyncio.run(runner.emit("agent_start", object()))  # does not raise


class TestEmitToolCall:
    def test_no_handlers_returns_none(self) -> None:
        runner = ExtensionRunner("/tmp")
        result = asyncio.run(runner.emit_tool_call(ToolCallEvent(tool_call_id="t1", tool_name="bash")))
        assert result is None

    def test_block_short_circuits_remaining_handlers(self) -> None:
        called: list[str] = []

        def h1(_event: Any, _ctx: Any) -> ToolCallEventResult:
            called.append("h1")
            return ToolCallEventResult(block=True, reason="nope")

        def h2(_event: Any, _ctx: Any) -> None:
            called.append("h2")

        runner = _runner_with_handlers({"tool_call": [h1, h2]})
        result = asyncio.run(runner.emit_tool_call(ToolCallEvent(tool_call_id="t1", tool_name="bash")))
        assert result is not None
        assert result.block is True
        assert result.reason == "nope"
        assert called == ["h1"]  # h2 never ran

    def test_non_blocking_result_keeps_calling_later_handlers(self) -> None:
        def h1(_event: Any, _ctx: Any) -> ToolCallEventResult:
            return ToolCallEventResult(block=False)

        def h2(_event: Any, _ctx: Any) -> ToolCallEventResult:
            return ToolCallEventResult(block=False, reason="from h2")

        runner = _runner_with_handlers({"tool_call": [h1, h2]})
        result = asyncio.run(runner.emit_tool_call(ToolCallEvent(tool_call_id="t1", tool_name="bash")))
        assert result is not None
        assert result.reason == "from h2"  # last non-empty result wins when nothing blocks

    def test_handler_can_mutate_arguments_in_place(self) -> None:
        def h1(event: ToolCallEvent, _ctx: Any) -> None:
            event.arguments["patched"] = True

        runner = _runner_with_handlers({"tool_call": [h1]})
        event = ToolCallEvent(tool_call_id="t1", tool_name="bash", arguments={"command": "ls"})
        asyncio.run(runner.emit_tool_call(event))
        assert event.arguments == {"command": "ls", "patched": True}

    def test_exception_in_handler_is_recorded_and_other_handlers_still_run(self) -> None:
        def broken(_event: Any, _ctx: Any) -> None:
            raise RuntimeError("boom")

        def h2(_event: Any, _ctx: Any) -> ToolCallEventResult:
            return ToolCallEventResult(block=True, reason="ok")

        runner = _runner_with_handlers({"tool_call": [broken, h2]})
        result = asyncio.run(runner.emit_tool_call(ToolCallEvent(tool_call_id="t1", tool_name="bash")))
        assert result is not None
        assert result.reason == "ok"
        assert len(runner.get_extensions().errors) == 1


class TestEmitToolResult:
    def test_no_handlers_returns_none(self) -> None:
        runner = ExtensionRunner("/tmp")
        result = asyncio.run(runner.emit_tool_result(ToolResultEvent(tool_call_id="t1", tool_name="bash")))
        assert result is None

    def test_no_handler_modifies_anything_returns_none(self) -> None:
        def h(_event: Any, _ctx: Any) -> ToolResultEventResult:
            return ToolResultEventResult()  # both fields None

        runner = _runner_with_handlers({"tool_result": [h]})
        result = asyncio.run(runner.emit_tool_result(ToolResultEvent(tool_call_id="t1", tool_name="bash")))
        assert result is None

    def test_is_error_override_is_applied(self) -> None:
        def h(_event: Any, _ctx: Any) -> ToolResultEventResult:
            return ToolResultEventResult(is_error=True)

        runner = _runner_with_handlers({"tool_result": [h]})
        result = asyncio.run(
            runner.emit_tool_result(ToolResultEvent(tool_call_id="t1", tool_name="bash", is_error=False))
        )
        assert result is not None
        assert result.is_error is True

    def test_handlers_chain_seeing_previous_mutations(self) -> None:
        def h1(_event: Any, _ctx: Any) -> ToolResultEventResult:
            return ToolResultEventResult(content=[TextContent(text="from h1")])

        def h2(event: ToolResultEvent, _ctx: Any) -> ToolResultEventResult:
            assert event.content == [TextContent(text="from h1")]
            return ToolResultEventResult(is_error=True)

        runner = _runner_with_handlers({"tool_result": [h1, h2]})
        result = asyncio.run(runner.emit_tool_result(ToolResultEvent(tool_call_id="t1", tool_name="bash")))
        assert result is not None
        assert result.content == [TextContent(text="from h1")]
        assert result.is_error is True
