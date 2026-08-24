"""pi_tools — the programmatic tool-calling library available inside a
running execute_code script.

Spec section 7's exact requirement: small, serializable, RPC-based,
never imports the real AgentTool/Tool object, never shares mutable
state with the parent directly, returns plain data. This module is
literally that — every function here is a thin RPC call over a loopback
socket to the parent process; nothing here executes a tool locally.

Usage inside a script run via execute_code:

    from pi_tools import read_file
    text = read_file("logs/app.log")
    errors = [line for line in text.splitlines() if "ERROR" in line]
    print("\n".join(errors[-100:]))

Connection parameters (host/port/token) are read from environment
variables the parent (pi_runtime.execute_code.runner.CodeExecutor) sets
when spawning this child — never hardcoded, never passed as a CLI
argument (which would leak the token into `ps`/task-manager output on
some platforms).
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from typing import Any

_ENV_HOST = "PI_RPC_HOST"
_ENV_PORT = "PI_RPC_PORT"
_ENV_TOKEN = "PI_RPC_TOKEN"


class PiToolsUnavailable(RuntimeError):
    """Raised when this script isn't actually running under
    execute_code's RPC harness (the env vars are absent) — calling a
    pi_tools function outside that context must fail loudly, not hang
    on a connection to nothing."""


class RpcCallError(RuntimeError):
    """Raised when the parent's RPC server reports an error response —
    carries the structured error type (spec section 41: never collapse
    to indistinguishable text) so calling code can branch on it."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"[{error_type}] {message}")
        self.error_type = error_type


def _connection_params() -> tuple[str, int, str]:
    host = os.environ.get(_ENV_HOST)
    port = os.environ.get(_ENV_PORT)
    token = os.environ.get(_ENV_TOKEN)
    if not host or not port or not token:
        raise PiToolsUnavailable(
            "pi_tools requires running inside execute_code's RPC harness "
            f"({_ENV_HOST}/{_ENV_PORT}/{_ENV_TOKEN} are not set)"
        )
    return host, int(port), token


def call_tool(tool: str, **arguments: Any) -> Any:
    """The generic RPC call every specific wrapper below is built on —
    also usable directly for any allowlisted tool that doesn't have its
    own typed wrapper yet."""
    host, port, token = _connection_params()
    request_id = uuid.uuid4().hex
    request = {"request_id": request_id, "token": token, "tool": tool, "arguments": arguments}

    with socket.create_connection((host, port), timeout=30) as sock:
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(65536)
            if not chunk:
                raise RpcCallError("rpc_error", "connection closed before a response was received")
            buffer += chunk
        line, _, _ = buffer.partition(b"\n")

    response = json.loads(line)
    if response.get("status") == "error":
        error = response.get("error") or {}
        raise RpcCallError(error.get("type", "rpc_error"), error.get("message", "unknown RPC error"))
    return response.get("result")


def read_file(path: str, *, offset: int = 0, limit: int = 2000) -> str:
    """RPC wrapper around the parent's `read` tool — text content only
    (an image result from the real tool is not representable as the
    plain string this wrapper promises; the parent raises a clear
    rpc_error for that case rather than silently mangling it)."""
    return str(call_tool("read_file", path=path, offset=offset, limit=limit))


__all__ = ["PiToolsUnavailable", "RpcCallError", "call_tool", "read_file"]
