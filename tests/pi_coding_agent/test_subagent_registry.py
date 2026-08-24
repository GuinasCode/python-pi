"""Tests for pi_coding_agent.subagent.registry and .runner's process-driving
logic (_drive), using a fake asyncio.subprocess.Process so timing (steer
mid-run, stop before vs. after natural completion, kill-on-timeout) is
fully deterministic — no real OS process or network/model call involved.

A real end-to-end smoke test (actually spawning a child `pi --print --mode
rpc` process) lives in test_subagent_e2e.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pi_coding_agent.subagent.agent_def import AgentDef
from pi_coding_agent.subagent.registry import SubagentRegistry
from pi_coding_agent.subagent.runner import _drive


class FakeStdin:
    """Records every line written, and can be "closed" (simulating the
    child having already exited) to make further writes fail."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            raise BrokenPipeError()
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True

    @property
    def lines(self) -> list[dict]:
        return [json.loads(chunk.decode()) for chunk in self.written]


class FakeStream:
    """Async iterator over preset byte lines. Each line can carry a delay;
    if `killed_event` fires before that delay elapses, the stream ends
    right there instead (simulating the OS closing the pipe on kill).
    Calls `on_exhausted` exactly once, the moment iteration naturally ends
    (whether by running out of lines or by being killed mid-wait) — this
    is what a real process's `wait()` actually blocks on (stdout closing),
    not a second independent consumption of the same stream."""

    def __init__(
        self,
        lines: list[bytes],
        delays: list[float] | None = None,
        killed_event: asyncio.Event | None = None,
        on_exhausted: object = None,
    ):
        self._lines = lines
        self._delays = delays or [0.0] * len(lines)
        self._killed_event = killed_event or asyncio.Event()
        self._i = 0
        self._on_exhausted = on_exhausted
        self._fired = False

    def __aiter__(self) -> FakeStream:
        return self

    def _fire_exhausted(self) -> None:
        if not self._fired:
            self._fired = True
            if self._on_exhausted is not None:
                self._on_exhausted()

    async def __anext__(self) -> bytes:
        if self._i >= len(self._lines):
            self._fire_exhausted()
            raise StopAsyncIteration
        delay = self._delays[self._i]
        if delay:
            try:
                await asyncio.wait_for(self._killed_event.wait(), timeout=delay)
                self._fire_exhausted()
                raise StopAsyncIteration  # killed before this line was "sent"
            except TimeoutError:
                pass
        line = self._lines[self._i]
        self._i += 1
        return line


class FakeProcess:
    """Minimal stand-in for asyncio.subprocess.Process, driven manually.
    `wait()` blocks on the same "process actually exited" signal that
    stdout exhaustion (or a kill) fires — it never consumes stdout itself,
    since the real driving code (_drive) is the one iterating it."""

    def __init__(self, stdout_lines: list[bytes], delays: list[float] | None = None, returncode: int = 0):
        self.stdin = FakeStdin()
        self._killed_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self.stdout = FakeStream(stdout_lines, delays, self._killed_event, on_exhausted=self._exit_event.set)
        self.stderr = FakeStream([])
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.stdin.close()
        self._killed_event.set()
        self._final_returncode = -9
        self._exit_event.set()

    async def wait(self) -> int:
        await self._exit_event.wait()
        self.returncode = self._final_returncode
        return self._final_returncode


def _event(**kwargs: object) -> bytes:
    return (json.dumps(kwargs) + "\n").encode()


class TestSubagentRegistry:
    def test_list_live_only_includes_running_handles(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="done", stop_reason="stop")])
        handle = registry.register("scout", "look around", proc)
        assert registry.list_live() == [handle]

        asyncio.run(_drive(handle, timeout=5.0, on_progress=None))
        assert registry.list_live() == []
        assert registry.list_all() == [handle]

    def test_get_exact_id(self) -> None:
        registry = SubagentRegistry()
        handle = registry.register("scout", "task", FakeProcess([]))
        assert registry.get(handle.id) is handle

    def test_get_unambiguous_prefix(self) -> None:
        registry = SubagentRegistry()
        handle = registry.register("scout", "task", FakeProcess([]))
        assert registry.get(handle.id[:4]) is handle

    def test_get_unknown_returns_none(self) -> None:
        registry = SubagentRegistry()
        assert registry.get("nope") is None

    def test_get_ambiguous_prefix_returns_none(self) -> None:
        registry = SubagentRegistry()
        h1 = registry.register("a", "t", FakeProcess([]))
        # Force a matching prefix rather than relying on random uuid collision.
        h2 = registry.register("b", "t", FakeProcess([]))
        h2.id = h1.id[:4] + "zzzz"
        registry._handles[h2.id] = h2
        assert registry.get(h1.id[:4]) is None


class TestDrive:
    def test_normal_completion_marks_done_with_accumulated_text(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess(
            [
                _event(type="ready"),
                _event(type="text_delta", delta="hello "),
                _event(type="text_delta", delta="world"),
                _event(type="done", stop_reason="stop"),
            ]
        )
        handle = registry.register("scout", "task", proc)
        asyncio.run(_drive(handle, timeout=5.0, on_progress=None))

        assert handle.status == "done"
        assert handle.result is not None
        assert handle.result.output == "hello world"
        assert handle.result.succeeded

    def test_error_event_marks_status_error(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="error", message="boom")])
        handle = registry.register("scout", "task", proc)
        asyncio.run(_drive(handle, timeout=5.0, on_progress=None))

        assert handle.status == "error"
        assert handle.result is not None
        assert "boom" in handle.result.output
        assert not handle.result.succeeded

    def test_progress_callback_receives_text_deltas(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="text_delta", delta="chunk1"), _event(type="done", stop_reason="stop")])
        handle = registry.register("scout", "task", proc)
        seen: list[str] = []
        asyncio.run(_drive(handle, timeout=5.0, on_progress=seen.append))
        assert "chunk1" in seen

    def test_timeout_kills_process_and_marks_status_timeout(self) -> None:
        registry = SubagentRegistry()
        # A line that never arrives within the timeout.
        proc = FakeProcess([_event(type="text_delta", delta="stuck")], delays=[5.0])
        handle = registry.register("scout", "task", proc)
        asyncio.run(_drive(handle, timeout=0.05, on_progress=None))

        assert handle.status == "timeout"
        assert proc.killed
        assert handle.result is not None
        assert "timeout" in handle.result.output.lower()

    def test_aborted_done_event_marks_status_killed(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="done", stop_reason="aborted")])
        handle = registry.register("scout", "task", proc)
        asyncio.run(_drive(handle, timeout=5.0, on_progress=None))
        assert handle.status == "killed"


class TestHandleSteer:
    @pytest.mark.asyncio
    async def test_steer_writes_json_line_while_running(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="text_delta", delta="x")], delays=[5.0])
        handle = registry.register("scout", "task", proc)
        drive_task = asyncio.create_task(_drive(handle, timeout=5.0, on_progress=None))

        ok = await handle.steer("also check the tests")
        assert ok is True
        assert proc.stdin.lines[-1] == {"type": "steer", "text": "also check the tests"}

        proc.kill()
        await drive_task

    @pytest.mark.asyncio
    async def test_steer_is_a_noop_once_finished(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="done", stop_reason="stop")])
        handle = registry.register("scout", "task", proc)
        await _drive(handle, timeout=5.0, on_progress=None)

        ok = await handle.steer("too late")
        assert ok is False


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_before_graceful_timeout_uses_child_reported_result(self) -> None:
        """The child honors the RPC stop command and reports
        stop_reason=aborted on its own, well within graceful_timeout — no
        hard kill needed."""
        registry = SubagentRegistry()
        proc = FakeProcess(
            [_event(type="text_delta", delta="partial"), _event(type="done", stop_reason="aborted")],
            delays=[0.0, 0.02],
        )
        handle = registry.register("scout", "task", proc)
        drive_task = asyncio.create_task(_drive(handle, timeout=5.0, on_progress=None))
        await asyncio.sleep(0.005)  # let the "partial" text_delta land first

        result = await handle.stop(graceful_timeout=1.0)

        assert result.status == "killed"
        assert not proc.killed  # exited on its own — never force-killed
        assert proc.stdin.lines[-1] == {"type": "stop"}
        await drive_task

    @pytest.mark.asyncio
    async def test_stop_force_kills_after_graceful_timeout(self) -> None:
        """The child never acknowledges the stop command — stop() must
        force-kill it once graceful_timeout elapses."""
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="text_delta", delta="stuck")], delays=[10.0])
        handle = registry.register("scout", "task", proc)
        drive_task = asyncio.create_task(_drive(handle, timeout=10.0, on_progress=None))
        await asyncio.sleep(0.005)

        result = await handle.stop(graceful_timeout=0.05)

        assert result.status == "killed"
        assert proc.killed
        await drive_task

    @pytest.mark.asyncio
    async def test_stop_on_already_finished_handle_returns_existing_result(self) -> None:
        registry = SubagentRegistry()
        proc = FakeProcess([_event(type="done", stop_reason="stop")])
        handle = registry.register("scout", "task", proc)
        await _drive(handle, timeout=5.0, on_progress=None)
        first_result = handle.result

        result = await handle.stop()
        assert result is first_result
        assert not proc.killed


class TestAgentDefSanity:
    def test_agent_def_constructs(self) -> None:
        agent = AgentDef(name="scout", system_prompt="You are a scout.")
        assert agent.name == "scout"
