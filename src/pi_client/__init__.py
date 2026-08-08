"""Pi client - connection management, session handles, and transport.

Mirrors packages/client/src/.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ClientError(Exception):
    """Base error for pi_client."""


class ServerError(ClientError):
    """Error from the server."""

    def __init__(self, code: str, message: str, details: Any = None) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


class DisconnectedError(ClientError):
    """Client is not connected."""


class ClientDisposedError(ClientError):
    """Client has been disposed."""


class SessionOwnershipError(ClientError):
    """Session is not owned by this client."""


class SessionDetachedError(ClientError):
    """Session was detached."""


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


class ByteTransport(Protocol):
    """Transport interface for binary frame communication."""

    def write(self, data: bytes) -> None: ...
    def on_data(self, callback: Callable[[bytes], None] | None) -> None: ...
    def on_close(self, callback: Callable[[], None] | None) -> None: ...
    def close(self) -> None: ...


class ByteTransportFactory(Protocol):
    """Factory for creating transports."""

    def connect(self) -> ByteTransport: ...


@dataclass
class ClientOptions:
    """Options for creating a PiClient."""

    transport_factory: ByteTransportFactory | None = None
    token: str = "default"
    auto_connect: bool = True


@dataclass
class CreateSessionOptions:
    """Options for creating a session."""

    cwd: str | None = None
    name: str | None = None
    model: dict[str, str] | None = None
    thinking_level: str | None = None


@dataclass
class SessionSummary:
    """Summary of a session."""

    id: str
    name: str | None = None
    cwd: str = ""
    created_at: int = 0
    updated_at: int = 0
    phase: str = "idle"
    model: dict[str, str] = field(default_factory=lambda: {"provider": "", "id": ""})
    thinking_level: str = "off"
    attached: bool = False
    locked: bool = False


@dataclass
class SessionHandle:
    """Handle to a session for prompt/steer/abort operations."""

    session_id: str
    _client: Any = None  # Reference to parent client

    async def prompt(self, text: str) -> Any:
        """Send a prompt to the session."""
        if self._client is None:
            raise ClientDisposedError("Client disposed")
        return await self._client._send_prompt(self.session_id, text)

    async def steer(self, text: str) -> Any:
        """Send a steering message to the session."""
        if self._client is None:
            raise ClientDisposedError("Client disposed")
        return await self._client._send_steer(self.session_id, text)

    async def abort(self) -> Any:
        """Abort the current operation."""
        if self._client is None:
            raise ClientDisposedError("Client disposed")
        return await self._client._send_abort(self.session_id)

    async def set_model(self, model: dict[str, str]) -> Any:
        """Set the model for the session."""
        if self._client is None:
            raise ClientDisposedError("Client disposed")
        return await self._client._send_set_model(self.session_id, model)

    async def set_thinking(self, level: str) -> Any:
        """Set the thinking level for the session."""
        if self._client is None:
            raise ClientDisposedError("Client disposed")
        return await self._client._send_set_thinking(self.session_id, level)


class ClientState:
    """Tracks client state including sessions and snapshots."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionSummary] = {}
        self.connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self.server_id: str = ""
        self.protocol_version: int = 0

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Update state from a server snapshot."""
        self.server_id = snapshot.get("serverId", "")
        self.protocol_version = snapshot.get("protocolVersion", 0)
        self.sessions.clear()
        for session_data in snapshot.get("sessions", []):
            summary = SessionSummary(
                id=session_data.get("id", ""),
                name=session_data.get("name"),
                cwd=session_data.get("cwd", ""),
                created_at=session_data.get("createdAt", 0),
                updated_at=session_data.get("updatedAt", 0),
                phase=session_data.get("phase", "idle"),
                model=session_data.get("model", {"provider": "", "id": ""}),
                thinking_level=session_data.get("thinkingLevel", "off"),
                attached=session_data.get("attached", False),
                locked=session_data.get("locked", False),
            )
            self.sessions[summary.id] = summary


class PiClient:
    """Pi client for connecting to a Pi server and managing sessions."""

    def __init__(self, options: ClientOptions) -> None:
        self._options = options
        self._state = ClientState()
        self._transport: ByteTransport | None = None
        self._disposed = False
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._request_futures: dict[str, asyncio.Future[Any]] = {}
        self._session_handles: dict[str, SessionHandle] = {}

    @property
    def state(self) -> ClientState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state.connection_state == ConnectionState.CONNECTED

    @property
    def is_disposed(self) -> bool:
        return self._disposed

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register an event listener and return an unsubscribe function."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an event to all listeners."""
        for listener in list(self._event_listeners):
            listener(event)

    def _check_disposed(self) -> None:
        if self._disposed:
            raise ClientDisposedError("Client has been disposed")

    def dispose(self) -> None:
        """Dispose the client and close the connection."""
        if self._disposed:
            return
        self._disposed = True
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._state.connection_state = ConnectionState.DISCONNECTED
        self._session_handles.clear()
        self._event_listeners.clear()

    def get_sessions(self) -> list[SessionSummary]:
        """Get all known sessions."""
        return list(self._state.sessions.values())

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Get a session by ID."""
        return self._state.sessions.get(session_id)

    def get_session_handle(self, session_id: str) -> SessionHandle:
        """Get or create a session handle."""
        if session_id not in self._session_handles:
            self._session_handles[session_id] = SessionHandle(session_id=session_id, _client=self)
        return self._session_handles[session_id]

    async def _send_prompt(self, session_id: str, text: str) -> Any:
        """Send a prompt command. Placeholder for full implementation."""
        self._check_disposed()
        return {"command": "prompt", "sessionId": session_id, "text": text}

    async def _send_steer(self, session_id: str, text: str) -> Any:
        """Send a steer command."""
        self._check_disposed()
        return {"command": "steer", "sessionId": session_id, "text": text}

    async def _send_abort(self, session_id: str) -> Any:
        """Send an abort command."""
        self._check_disposed()
        return {"command": "abort", "sessionId": session_id}

    async def _send_set_model(self, session_id: str, model: dict[str, str]) -> Any:
        """Send a set_model command."""
        self._check_disposed()
        return {"command": "set_model", "sessionId": session_id, "model": model}

    async def _send_set_thinking(self, session_id: str, level: str) -> Any:
        """Send a set_thinking command."""
        self._check_disposed()
        return {"command": "set_thinking", "sessionId": session_id, "thinkingLevel": level}


__all__ = [
    "ByteTransport",
    "ByteTransportFactory",
    "ClientDisposedError",
    "ClientError",
    "ClientOptions",
    "ClientState",
    "ConnectionState",
    "CreateSessionOptions",
    "DisconnectedError",
    "PiClient",
    "ServerError",
    "SessionDetachedError",
    "SessionHandle",
    "SessionOwnershipError",
    "SessionSummary",
]
