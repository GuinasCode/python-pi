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

from pi_coding_agent.tools import ToolResult, execute_bash, fetch_url, grep_search, list_files, read_file
from pi_runtime.execute_code.rpc import RpcError

# terminal is capped well below execute_bash's own 120s default so a
# runaway command inside execute_code can't tie up the parent's RPC
# server for as long as a directly-invoked bash tool call could.
_MAX_TERMINAL_TIMEOUT_SECONDS = 60


def _text_or_raise(result: ToolResult, *, tool_failed_message: str) -> str:
    if result.is_error:
        text = result.content[0].get("text", tool_failed_message) if result.content else tool_failed_message
        raise RpcError(text, error_type="tool_error")
    if not result.content or result.content[0].get("type") != "text":
        raise RpcError(
            "tool returned a non-text result (e.g. an image) — not representable over this RPC wrapper",
            error_type="tool_error",
        )
    return str(result.content[0].get("text", ""))


async def read_file_handler(_tool: str, arguments: dict[str, Any]) -> str:
    path = arguments.get("path")
    if not isinstance(path, str):
        raise RpcError("missing or invalid 'path' argument", error_type="malformed_request")
    offset = int(arguments.get("offset", 0))
    limit = int(arguments.get("limit", 2000))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: read_file(path, offset=offset, limit=limit))
    return _text_or_raise(result, tool_failed_message="read_file failed")


async def search_files_handler(_tool: str, arguments: dict[str, Any]) -> str:
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str):
        raise RpcError("missing or invalid 'pattern' argument", error_type="malformed_request")
    path = arguments.get("path", ".")
    include = arguments.get("include", "*")
    ignore_case = bool(arguments.get("ignore_case", False))
    max_results = int(arguments.get("max_results", 50))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: grep_search(pattern, path=path, include=include, ignore_case=ignore_case, max_results=max_results),
    )
    return _text_or_raise(result, tool_failed_message="search_files failed")


async def list_files_handler(_tool: str, arguments: dict[str, Any]) -> str:
    path = arguments.get("path", ".")
    max_depth = int(arguments.get("max_depth", 3))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: list_files(path, max_depth=max_depth))
    return _text_or_raise(result, tool_failed_message="list_files failed")


async def terminal_handler(_tool: str, arguments: dict[str, Any]) -> str:
    command = arguments.get("command")
    if not isinstance(command, str):
        raise RpcError("missing or invalid 'command' argument", error_type="malformed_request")
    cwd = arguments.get("cwd")
    timeout = min(int(arguments.get("timeout", _MAX_TERMINAL_TIMEOUT_SECONDS)), _MAX_TERMINAL_TIMEOUT_SECONDS)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: execute_bash(command, cwd=cwd, timeout=timeout))
    return _text_or_raise(result, tool_failed_message="terminal command failed")


async def fetch_url_handler(_tool: str, arguments: dict[str, Any]) -> str:
    url = arguments.get("url")
    if not isinstance(url, str):
        raise RpcError("missing or invalid 'url' argument", error_type="malformed_request")
    timeout = float(arguments.get("timeout", 30.0))

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: fetch_url(url, timeout=timeout))
    return _text_or_raise(result, tool_failed_message="fetch_url failed")


# execute_code and delegate_task are deliberately never registered here
# (spec section 6/8): a script can only reach the tools this dict names,
# so recursion into execute_code and unconstrained subagent spawning are
# blocked structurally, not by a runtime check that could be bypassed.
DEFAULT_HANDLERS = {
    "read_file": read_file_handler,
    "search_files": search_files_handler,
    "list_files": list_files_handler,
    "terminal": terminal_handler,
    "fetch_url": fetch_url_handler,
}

__all__ = [
    "DEFAULT_HANDLERS",
    "fetch_url_handler",
    "list_files_handler",
    "read_file_handler",
    "search_files_handler",
    "terminal_handler",
]
