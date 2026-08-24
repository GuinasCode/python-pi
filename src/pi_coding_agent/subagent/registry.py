"""Live tracking of running subagent child processes.

Before this, ``run_subagent`` was a pure black box: the caller (the
``subagent`` tool) called it, blocked on ``asyncio.gather``, and got a
result back once every child process had exited — no visibility into what
was still running, no way to list live children, no way to stop or steer
one mid-run.

``SubagentRegistry`` sits between the tool and the runner: every spawn
goes through ``registry.spawn()``, which starts the child process and
immediately returns a ``SubagentHandle`` the caller can ``await .wait()``
on for the final result — but the handle is *also* registered and stays
visible (``list_live()``/``get()``) and controllable (``.stop()``,
``.steer()``) for as long as the process runs, from any other code path
(e.g. the ``/agents`` REPL command) concurrently with the tool call that
spawned it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class SubagentResult:
    """Result of a single subagent invocation."""

    output: str
    exit_code: int
    agent_name: str
    status: str = "done"  # done | error | timeout | killed

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.status == "done"


@dataclass
class SubagentHandle:
    """A running (or finished) subagent child process.

    ``proc`` and the internal event are populated by
    ``SubagentRegistry.spawn`` — this dataclass is intentionally a plain
    record plus the two RPC actions (``stop``/``steer``); the actual
    process-driving loop (reading stdout, deciding when it's done) lives
    in ``runner.py`` so ``registry.py`` never has to know about the RPC
    wire format.
    """

    id: str
    agent_name: str
    task: str
    proc: asyncio.subprocess.Process
    started_at: float = field(default_factory=time.time)
    status: str = "running"  # running | done | error | timeout | killed
    result: SubagentResult | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    async def wait(self) -> SubagentResult:
        """Block until the child has finished (however it finished) and
        return its result."""
        await self._done.wait()
        assert self.result is not None
        return self.result

    def mark_done(self, result: SubagentResult) -> None:
        """Called by the driving task in runner.py when the process
        actually exits, however that happened. Idempotent — a hard-kill
        from .stop()'s graceful_timeout path and the driving task's own
        proc.wait() unblocking shortly after can both race to call this;
        whichever gets here first wins, the second is a no-op."""
        if self._done.is_set():
            return
        self.status = result.status
        self.result = result
        self._done.set()

    async def steer(self, text: str) -> bool:
        """Send extra guidance to the running child, delivered at its next
        turn boundary (see AgentSession.queue_steer_message). Returns
        False (no-op) if the child has already finished or its stdin is
        unavailable — steering a finished subagent isn't an error, just
        pointless."""
        if self.status != "running" or self.proc.stdin is None or self.proc.stdin.is_closing():
            return False
        try:
            self.proc.stdin.write((json.dumps({"type": "steer", "text": text}) + "\n").encode())
            await self.proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

    async def stop(self, *, graceful_timeout: float = 3.0) -> SubagentResult:
        """Ask the child to stop gracefully (RPC ``stop`` command, honored
        at its next turn boundary); force-kill it if it hasn't exited
        within ``graceful_timeout``. Always returns a result — a hard kill
        still produces one (status="killed", whatever output was captured
        before the kill)."""
        if self.status != "running":
            assert self.result is not None
            return self.result
        if self.proc.stdin is not None and not self.proc.stdin.is_closing():
            try:
                self.proc.stdin.write((json.dumps({"type": "stop"}) + "\n").encode())
                await self.proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(self._done.wait(), timeout=graceful_timeout)
        except TimeoutError:
            self.proc.kill()
            await self.proc.wait()
            # The driving task's own proc.wait() will also unblock and call
            # mark_done() shortly after this, from a fresh SubagentResult it
            # builds — but the caller of stop() shouldn't have to wait for
            # that second event loop tick, so also mark it here directly if
            # the driving task hasn't beaten us to it.
            if self.status == "running":
                self.mark_done(
                    SubagentResult(output="[stopped]", exit_code=-1, agent_name=self.agent_name, status="killed")
                )
        assert self.result is not None
        return self.result


class SubagentRegistry:
    """Registry of subagent handles for one AgentSession's lifetime."""

    def __init__(self) -> None:
        self._handles: dict[str, SubagentHandle] = {}

    def register(self, agent_name: str, task: str, proc: asyncio.subprocess.Process) -> SubagentHandle:
        handle = SubagentHandle(id=uuid.uuid4().hex[:8], agent_name=agent_name, task=task, proc=proc)
        self._handles[handle.id] = handle
        return handle

    def list_live(self) -> list[SubagentHandle]:
        return [h for h in self._handles.values() if h.status == "running"]

    def list_all(self) -> list[SubagentHandle]:
        return sorted(self._handles.values(), key=lambda h: h.started_at, reverse=True)

    def get(self, ref: str) -> SubagentHandle | None:
        """Exact id or unambiguous id-prefix — same resolution shape as
        SessionManager.resolve_session_ref, for a consistent /command UX."""
        if ref in self._handles:
            return self._handles[ref]
        matches = [h for h in self._handles.values() if h.id.startswith(ref)]
        return matches[0] if len(matches) == 1 else None


__all__ = ["SubagentHandle", "SubagentRegistry", "SubagentResult"]
