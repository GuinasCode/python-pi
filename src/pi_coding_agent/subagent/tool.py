"""Subagent AgentTool — lets the model delegate to specialised subagents.

The model parent invokes this tool like any other tool.  Underneath, it
spawns child pi processes, one per subagent, and returns their collected
output as the tool result.

Three modes
-----------
single
    ``{ "agent": "<name>", "task": "<description>" }``
    One subagent, one task.

parallel
    ``{ "tasks": [{ "agent": "<name>", "task": "..." }, ...] }``
    Up to 8 subagents, at most 4 running concurrently.  Each runs on its
    own lane via ``asyncio.gather`` + a semaphore.

chain
    ``{ "chain": [{ "agent": "<name>", "task": "..." }, ...] }``
    Sequential.  Use ``{previous}`` inside a task string to splice in the
    previous step's output (compacted to the first 4 000 chars).

Agent definitions
-----------------
Agents are Markdown files with YAML frontmatter discovered by
:func:`~pi_coding_agent.subagent.agent_def.discover_agents`.  See
``agent_def.py`` for the format.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent
from pi_coding_agent.subagent.agent_def import AgentDef, discover_agents
from pi_coding_agent.subagent.runner import SubagentResult, run_subagent

_PARALLEL_MAX_TASKS = 8
_PARALLEL_CONCURRENCY = 4
_CHAIN_PREV_MAX_CHARS = 4_000


def _ok(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


def _summarize(result: SubagentResult) -> str:
    header = f"[{result.agent_name}]" + ("" if result.succeeded else f" exit={result.exit_code}")
    body = result.output.strip() or "(no output)"
    return f"{header}\n{body}"


async def _run_single(
    agents: dict[str, AgentDef],
    args: dict[str, Any],
    on_progress: Callable[[str], None] | None,
) -> AgentToolResult:
    agent_name = args.get("agent", "")
    task = args.get("task", "")
    agent = agents.get(agent_name)
    if agent is None:
        available = ", ".join(agents) if agents else "(none discovered)"
        return _ok(f"Unknown agent: {agent_name!r}. Available: {available}")
    result = await run_subagent(agent, task, on_progress=on_progress)
    return _ok(_summarize(result))


async def _run_parallel(
    agents: dict[str, AgentDef],
    args: dict[str, Any],
    on_progress: Callable[[str], None] | None,
) -> AgentToolResult:
    tasks_spec = (args.get("tasks") or [])[:_PARALLEL_MAX_TASKS]
    sem = asyncio.Semaphore(_PARALLEL_CONCURRENCY)

    async def _bounded(spec: dict[str, Any]) -> SubagentResult:
        agent_name = spec.get("agent", "")
        task = spec.get("task", "")
        agent = agents.get(agent_name) or AgentDef(name=agent_name, system_prompt=f"You are {agent_name}.")
        async with sem:
            return await run_subagent(agent, task, on_progress=on_progress)

    outcomes = await asyncio.gather(*[_bounded(s) for s in tasks_spec], return_exceptions=True)

    parts: list[str] = []
    for spec, outcome in zip(tasks_spec, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            parts.append(f"[{spec.get('agent', '?')}] error: {outcome}")
        else:
            parts.append(_summarize(outcome))

    return _ok("\n\n---\n\n".join(parts))


async def _run_chain(
    agents: dict[str, AgentDef],
    args: dict[str, Any],
    on_progress: Callable[[str], None] | None,
) -> AgentToolResult:
    chain_spec = args.get("chain") or []
    previous_output = ""
    parts: list[str] = []

    for step in chain_spec:
        agent_name = step.get("agent", "")
        raw_task = step.get("task", "")
        # Inject previous output (truncated so context stays manageable)
        task = raw_task.replace("{previous}", previous_output[:_CHAIN_PREV_MAX_CHARS])

        agent = agents.get(agent_name)
        if agent is None:
            err = f"[{agent_name}] unknown agent — skipping step"
            parts.append(err)
            previous_output = err
            continue

        result = await run_subagent(agent, task, on_progress=on_progress)
        summary = _summarize(result)
        parts.append(summary)
        previous_output = result.output.strip()

    return _ok("\n\n---\n\n".join(parts))


def create_subagent_tool(
    *,
    cwd: str,
    config_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> AgentTool:
    """Return an :class:`AgentTool` that lets the model invoke subagents.

    *cwd* and *config_dir* are used to discover agent markdown definitions.
    *on_progress* receives each stdout line from running subagents, useful for
    streaming live progress to the interactive display.
    """
    agents = discover_agents(cwd, config_dir)
    agent_names = list(agents) if agents else []
    agent_list_str = ", ".join(agent_names) if agent_names else "(none discovered — add .pi/agents/<name>.md)"

    schema: dict[str, Any] = {
        "type": "object",
        "oneOf": [
            {
                "title": "single",
                "description": f"Run one subagent on one task. Available agents: {agent_list_str}",
                "required": ["agent", "task"],
                "properties": {
                    "agent": {"type": "string", "description": "Agent name"},
                    "task": {"type": "string", "description": "Task description for this agent"},
                },
                "additionalProperties": False,
            },
            {
                "title": "parallel",
                "description": f"Run multiple agents concurrently (max {_PARALLEL_MAX_TASKS}, "
                               f"{_PARALLEL_CONCURRENCY} at a time). Available: {agent_list_str}",
                "required": ["tasks"],
                "properties": {
                    "tasks": {
                        "type": "array",
                        "maxItems": _PARALLEL_MAX_TASKS,
                        "items": {
                            "type": "object",
                            "required": ["agent", "task"],
                            "properties": {
                                "agent": {"type": "string"},
                                "task": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "additionalProperties": False,
            },
            {
                "title": "chain",
                "description": "Run agents sequentially. Use {previous} to pass prior output.",
                "required": ["chain"],
                "properties": {
                    "chain": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["agent", "task"],
                            "properties": {
                                "agent": {"type": "string"},
                                "task": {
                                    "type": "string",
                                    "description": "Use {previous} to reference the previous step's output",
                                },
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "additionalProperties": False,
            },
        ],
    }

    async def _execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        update: Any,
    ) -> AgentToolResult:
        if "chain" in args:
            return await _run_chain(agents, args, on_progress)
        if "tasks" in args:
            return await _run_parallel(agents, args, on_progress)
        return await _run_single(agents, args, on_progress)

    return AgentTool(
        name="subagent",
        description=(
            "Delegate tasks to specialised subagent processes. "
            "Modes: single {agent,task} | parallel {tasks:[...]} | chain {chain:[...]}. "
            f"Available agents: {agent_list_str}."
        ),
        parameters=schema,
        execute=_execute,
    )


__all__ = ["create_subagent_tool"]
