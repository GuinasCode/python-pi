"""Spawn a pi subprocess for a single subagent invocation.

Each subagent runs as a child pi process in RPC print mode
(``--mode rpc``): a structured JSON-Lines event stream on stdout (same
event shapes as ``--mode json``), plus a command channel on stdin that
lets the parent steer or stop the child while it's still running —
``{"type": "steer", "text": "..."}`` queues extra guidance, delivered at
the child's next turn boundary; ``{"type": "stop"}`` asks it to abort at
its next turn boundary. See ``pi_coding_agent.print_mode._run_rpc_mode``
for the child side of this protocol.

Every spawn goes through a ``SubagentRegistry`` (see ``registry.py``), so
the process stays visible/listable/stoppable/steerable from any other code
path — not just the caller that spawned it — for as long as it runs, not
only once it's done. ``run_subagent`` is a blocking convenience wrapper
(spawn, then wait) for callers that don't need that live visibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Callable

from pi_coding_agent.subagent.agent_def import AgentDef
from pi_coding_agent.subagent.registry import SubagentHandle, SubagentRegistry, SubagentResult

_DEFAULT_TIMEOUT = 300.0

# asyncio only holds a weak reference to a task started via create_task —
# without also keeping a strong reference somewhere, the task can be
# garbage-collected before it finishes. This set is that strong reference,
# for the driving tasks spawn_subagent() fires off; each removes itself
# once done.
_background_drives: set[asyncio.Task[None]] = set()


def _build_args(agent: AgentDef, task: str) -> list[str]:
    """Same Python interpreter that's currently running, so the child
    shares this process's virtualenv without needing ``pi`` on PATH."""
    args: list[str] = [
        sys.executable,
        "-m",
        "pi_coding_agent.cli",
        "--print",
        "--mode",
        "rpc",
        "--no-session",
    ]
    if agent.system_prompt:
        args += ["--system-prompt", agent.system_prompt]
    if agent.tools:
        args += ["--tools", ",".join(agent.tools)]
    if agent.model:
        args += ["--model", agent.model]
    if agent.temperature is not None:
        args += ["--temperature", str(agent.temperature)]
    args.append(task)
    return args


async def spawn_subagent(
    registry: SubagentRegistry,
    agent: AgentDef,
    task: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    on_progress: Callable[[str], None] | None = None,
    cwd: str | None = None,
) -> SubagentHandle:
    """Start *agent* on *task* as a child process and return immediately.

    The returned handle is already registered in *registry* — visible via
    ``list_live()``/``get()`` and controllable via ``.stop()``/``.steer()``
    — before this coroutine's caller has done anything else. Await
    ``handle.wait()`` for the final result; a background task (started
    here) drives the process to completion and calls
    ``handle.mark_done()`` regardless of how it ends (normal exit,
    timeout, or an explicit ``.stop()``).

    *cwd* defaults to inheriting this process's own working directory
    (unset) — real subagent spawns want that, since it's what makes the
    child's ``.env``/project settings resolve the same way the parent
    session's did. Only tests that need a child isolated from the
    parent's real ``.env`` (deterministic faux-provider tests) pass an
    explicit tmp directory.
    """
    proc = await asyncio.create_subprocess_exec(
        *_build_args(agent, task),
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    handle = registry.register(agent.name, task, proc)
    drive_task = asyncio.create_task(_drive(handle, timeout=timeout, on_progress=on_progress))
    _background_drives.add(drive_task)
    drive_task.add_done_callback(_background_drives.discard)
    return handle


async def run_subagent(
    agent: AgentDef,
    task: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    on_progress: Callable[[str], None] | None = None,
    registry: SubagentRegistry | None = None,
    cwd: str | None = None,
) -> SubagentResult:
    """Spawn + wait, for callers that just want the final result and don't
    need live orchestration. When *registry* is omitted, a throwaway
    one-off registry backs the single handle for the duration of this
    call — it's simply not reachable from any other code path (nothing
    outside this function ever gets a reference to it)."""
    handle = await spawn_subagent(
        registry or SubagentRegistry(), agent, task, timeout=timeout, on_progress=on_progress, cwd=cwd
    )
    return await handle.wait()


async def _drive(
    handle: SubagentHandle,
    *,
    timeout: float,
    on_progress: Callable[[str], None] | None,
) -> None:
    """Pump the child's stdout (JSON events) and stderr (drained, not
    surfaced — same as before, just now actually read so a full stderr
    pipe can't deadlock the child) until it exits or *timeout* elapses,
    then resolve the handle exactly once."""
    proc = handle.proc
    text_parts: list[str] = []
    final_status = "done"
    error_message: str | None = None

    async def _pump_stdout() -> None:
        nonlocal final_status, error_message
        assert proc.stdout
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a stray non-JSON line rather than dropping it —
                # still counts as progress/output.
                text_parts.append(line)
                if on_progress:
                    on_progress(line)
                continue
            etype = event.get("type")
            if etype == "text_delta":
                delta = event.get("delta", "")
                text_parts.append(delta)
                if on_progress:
                    on_progress(delta)
            elif etype == "tool_call_start" and on_progress:
                on_progress(f"[tool: {event.get('name', '')}]")
            elif etype == "error":
                final_status = "error"
                error_message = event.get("message", "")
            elif etype == "done" and event.get("stop_reason") == "aborted":
                final_status = "killed"

    async def _drain_stderr() -> None:
        assert proc.stderr
        async for _ in proc.stderr:
            pass

    try:
        await asyncio.wait_for(
            asyncio.gather(_pump_stdout(), _drain_stderr(), proc.wait()),
            timeout=timeout,
        )
        output = "".join(text_parts).strip()
        if final_status == "error":
            output = f"[error] {error_message}\n{output}".strip()
        elif (proc.returncode or 0) != 0 and final_status == "done":
            final_status = "error"
        result = SubagentResult(
            output=output, exit_code=proc.returncode or 0, agent_name=handle.agent_name, status=final_status
        )
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        result = SubagentResult(
            output=f"[timeout after {timeout:.0f}s]\n{''.join(text_parts)}".strip(),
            exit_code=-1,
            agent_name=handle.agent_name,
            status="timeout",
        )

    handle.mark_done(result)


__all__ = ["SubagentHandle", "SubagentRegistry", "SubagentResult", "run_subagent", "spawn_subagent"]
