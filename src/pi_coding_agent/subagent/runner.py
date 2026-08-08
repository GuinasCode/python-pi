"""Spawn a pi subprocess for a single subagent invocation.

Each subagent runs as a child pi process in print mode (``--print``),
capturing stdout as the result.  stderr is forwarded to the caller for
visibility but not included in the tool result.

The runner opens a second *lane* on the parent session (via SessionLanes)
so the subagent can run concurrently with other operations in the parent
session without blocking the main lane.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass

from pi_coding_agent.subagent.agent_def import AgentDef


@dataclass
class SubagentResult:
    """Result of a single subagent invocation."""

    output: str
    exit_code: int
    agent_name: str

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


async def run_subagent(
    agent: AgentDef,
    task: str,
    *,
    timeout: float = 300.0,
    on_progress: Callable[[str], None] | None = None,
) -> SubagentResult:
    """Spawn a pi subprocess for *agent* on *task*.

    Uses the same Python interpreter that is currently running so that the
    child process shares the same virtualenv without requiring ``pi`` to be
    on PATH.

    *on_progress* is called with each stdout line as it arrives (streaming
    progress back to the caller).
    """
    args: list[str] = [
        sys.executable,
        "-m",
        "pi_coding_agent.cli",
        "--print",
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

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    chunks: list[str] = []

    async def _read_stdout() -> None:
        assert proc.stdout
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace")
            chunks.append(line)
            if on_progress:
                on_progress(line)

    try:
        await asyncio.wait_for(
            asyncio.gather(_read_stdout(), proc.wait()),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return SubagentResult(
            output=f"[timeout after {timeout:.0f}s]\n{''.join(chunks)}",
            exit_code=-1,
            agent_name=agent.name,
        )

    return SubagentResult(
        output="".join(chunks),
        exit_code=proc.returncode or 0,
        agent_name=agent.name,
    )


__all__ = ["SubagentResult", "run_subagent"]
