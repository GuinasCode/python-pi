"""RPC handlers — the parent-side functions an RpcServer actually calls.

Each handler wraps a real, existing pi_coding_agent tool function
(never reimplements one) and adapts its ToolResult into the plain,
serializable value pi_tools' wrappers promise. Handlers run in the
parent's own event loop (via run_in_executor for the underlying sync
tool call, same pattern pi_coding_agent.agent_session._execute_tool_call
already uses for builtin tools) — so a call here goes through the exact
same code path a direct `read` tool call from the model would.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pi_coding_agent.tools import read_file
from pi_runtime.execute_code.rpc import RpcError


async def read_file_handler(_tool: str, arguments: dict[str, Any]) -> str:
    path = arguments.get("path")
    if not isinstance(path, str):
        raise RpcError("missing or invalid 'path' argument", error_type="malformed_request")
    offset = int(arguments.get("offset", 0))
    limit = int(arguments.get("limit", 2000))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: read_file(path, offset=offset, limit=limit))

    if result.is_error:
        text = result.content[0].get("text", "read_file failed") if result.content else "read_file failed"
        raise RpcError(text, error_type="tool_error")

    if not result.content or result.content[0].get("type") != "text":
        raise RpcError(
            "read_file returned a non-text result (e.g. an image) — not representable over this RPC wrapper",
            error_type="tool_error",
        )
    return str(result.content[0].get("text", ""))


DEFAULT_HANDLERS = {"read_file": read_file_handler}

__all__ = ["DEFAULT_HANDLERS", "read_file_handler"]
