"""Example Pi extension: a `hello` tool that greets someone by name.

Copy this into <project>/.pi/extensions/ (or ~/.pi/extensions/ to make it
available everywhere) to try it out. See docs/extensions.md for the full
authoring guide.
"""

from __future__ import annotations

from typing import Any

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent


async def _hello(
    _tool_call_id: str,
    args: dict[str, Any],
    _context: Any,
    _on_update: Any,
) -> AgentToolResult:
    name = args.get("name", "")
    return AgentToolResult(content=[TextContent(text=f"Hello, {name}!")])


def extension(pi: Any) -> None:
    pi.register_tool(
        AgentTool(
            name="hello",
            description="Greets someone by name.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Who to greet"}},
                "required": ["name"],
            },
            label="Hello",
            execute=_hello,
        )
    )
