"""Example Pi extension: exposes execute_code as a real tool the model
can call, via the official extension mechanism (pi.register_tool) — not
a second tool-registration path (spec section 43: "o sistema de
extensions já existe... não crie uma nova extension mechanism").

Copy this into <project>/.pi/extensions/ (or ~/.pi/extensions/) to try
it. See docs/extensions.md for the authoring guide and
docs/execute-code.md for what execute_code actually does and doesn't
guarantee (in particular: mode="strict" is best-effort exposure
reduction, not a sandbox).
"""

from __future__ import annotations

from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent
from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.runner import CodeExecutor

_executor = CodeExecutor()


async def _execute_code(
    _tool_call_id: str,
    args: dict[str, Any],
    _context: Any,
    _on_update: Any,
) -> AgentToolResult:
    code = args.get("code", "")
    timeout = float(args.get("timeout", 60.0))
    mode = args.get("mode", "strict")

    result = await _executor.execute(
        code,
        timeout=timeout,
        mode=mode,
        rpc_handlers=DEFAULT_HANDLERS,
    )

    summary = (
        f"status: {result.status.value}\n"
        f"exit_code: {result.exit_code}\n"
        f"duration_ms: {result.duration_ms:.0f}\n"
        f"rpc_call_count: {result.rpc_call_count}\n"
        f"--- stdout ---\n{result.stdout.preview}\n"
        f"--- stderr ---\n{result.stderr.preview}"
    )
    return AgentToolResult(content=[TextContent(text=summary)])


def extension(pi: Any) -> None:
    pi.register_tool(
        AgentTool(
            name="execute_code",
            description=(
                "Run a Python script to process large tool outputs programmatically "
                "(filter logs, summarize results, loop/branch) instead of pulling raw "
                "data into the conversation. The script can call a small allowlisted "
                "set of tools via `from pi_tools import read_file, search_files, "
                "list_files, terminal, fetch_url` — never itself, and never "
                "delegate_task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python script to run"},
                    "timeout": {"type": "number", "description": "Seconds before the script is killed"},
                    "mode": {
                        "type": "string",
                        "enum": ["strict", "project"],
                        "description": (
                            "'strict' (default): throwaway cwd, minimal environment. "
                            "'project': real project cwd and full environment — only "
                            "use when the script genuinely needs project access."
                        ),
                    },
                },
                "required": ["code"],
            },
            label="Execute Code",
            execute=_execute_code,
        )
    )
