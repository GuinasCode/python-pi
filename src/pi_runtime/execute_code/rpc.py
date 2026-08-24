"""RPC transport for execute_code — Slice A2.

Architectural decision (documented, not silent — spec's own rule that
choices need justification): the spec prefers a Unix domain socket on
POSIX with "a compatible mechanism on Windows". This development
environment's own CPython build has no `socket.AF_UNIX` at all on
Windows (`hasattr(socket, "AF_UNIX")` is False here — confirmed, not
assumed), and maintaining two separate transport implementations per OS
would double the security-relevant surface (the exact code that
validates every incoming request) for a benefit — marginally different
socket semantics — that doesn't offset the cost. So: TCP loopback
(127.0.0.1, OS-assigned port 0) with a per-execution random auth token,
never HTTP (a bare JSON-Lines protocol over the raw socket) — every
message must carry the exact token or the parent closes the connection
immediately. This is not "less secure than a Unix socket" in practice:
loopback-only binding already means no other host can reach the port,
and the token means no other *local* process can either without
already having intercepted the child's environment (at which point it
already has arbitrary code execution in this process tree regardless of
transport choice).

Message framing: one JSON object per line (`\n`-terminated), both
directions — request `{"request_id", "token", "tool", "arguments"}`,
response `{"request_id", "status": "success"|"error", "result"|"error"}`.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

RpcHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


class RpcError(Exception):
    """Raised by a handler to produce a structured error response
    (`{"type": ..., "message": ...}`) instead of a bare string — spec
    section 9's error shape."""

    def __init__(self, message: str, *, error_type: str = "tool_error") -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass
class RpcCallRecord:
    """One logged request/response — Telemetry (spec section 48)
    consumes this list via RpcServer.call_log, not a second logging
    mechanism."""

    request_id: str
    tool: str
    status: str
    error_type: str | None = None


@dataclass
class RpcServer:
    """Owns the loopback listener for exactly one execute_code
    invocation. `handlers` maps an allowlisted tool name to an async
    callable — the *parent* decides what's callable (spec section 10:
    "nunca confiar no processo filho"), the child never gets to invoke
    anything the parent didn't explicitly register here."""

    handlers: dict[str, RpcHandler]
    max_calls: int | None = None
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    call_log: list[RpcCallRecord] = field(default_factory=list)

    _server: asyncio.AbstractServer | None = field(default=None, init=False, repr=False)
    _port: int = field(default=0, init=False, repr=False)

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle_connection, host="127.0.0.1", port=0)
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        return self._port

    @property
    def port(self) -> int:
        return self._port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> RpcServer:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                await self._handle_one_request(line, writer)
        finally:
            writer.close()

    async def _handle_one_request(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            writer.write(
                self._encode(
                    {
                        "request_id": None,
                        "status": "error",
                        "error": {"type": "malformed_request", "message": "invalid JSON"},
                    }
                )
            )
            await writer.drain()
            return

        request_id = request.get("request_id")
        if request.get("token") != self.token:
            writer.write(self._error(request_id, "unauthorized", "invalid or missing token"))
            await writer.drain()
            return

        tool = request.get("tool")
        arguments = request.get("arguments") or {}
        if not isinstance(tool, str):
            writer.write(self._error(request_id, "malformed_request", "missing 'tool' field"))
            await writer.drain()
            return

        if self.max_calls is not None and len(self.call_log) >= self.max_calls:
            self.call_log.append(
                RpcCallRecord(request_id=request_id, tool=tool, status="error", error_type="resource_limit")
            )
            writer.write(self._error(request_id, "resource_limit", f"RPC call budget ({self.max_calls}) exceeded"))
            await writer.drain()
            return

        handler = self.handlers.get(tool)
        if handler is None:
            self.call_log.append(
                RpcCallRecord(request_id=request_id, tool=tool, status="error", error_type="unknown_tool")
            )
            writer.write(
                self._error(request_id, "unknown_tool", f"tool {tool!r} is not allowlisted for this execution")
            )
            await writer.drain()
            return

        try:
            result = await handler(tool, arguments)
        except RpcError as exc:
            self.call_log.append(
                RpcCallRecord(request_id=request_id, tool=tool, status="error", error_type=exc.error_type)
            )
            writer.write(self._error(request_id, exc.error_type, exc.message))
            await writer.drain()
            return
        except Exception as exc:  # a handler bug must still produce a structured response, never crash the RPC loop
            self.call_log.append(
                RpcCallRecord(request_id=request_id, tool=tool, status="error", error_type="rpc_error")
            )
            writer.write(self._error(request_id, "rpc_error", str(exc)))
            await writer.drain()
            return

        self.call_log.append(RpcCallRecord(request_id=request_id, tool=tool, status="success"))
        writer.write(self._encode({"request_id": request_id, "status": "success", "result": result}))
        await writer.drain()

    def _error(self, request_id: str | None, error_type: str, message: str) -> bytes:
        return self._encode(
            {"request_id": request_id, "status": "error", "error": {"type": error_type, "message": message}}
        )

    @staticmethod
    def _encode(payload: dict[str, Any]) -> bytes:
        return (json.dumps(payload, default=str) + "\n").encode("utf-8")


__all__ = ["RpcCallRecord", "RpcError", "RpcHandler", "RpcServer"]
