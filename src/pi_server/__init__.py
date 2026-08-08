"""Pi server - session management and protocol handling.

Mirrors packages/server/src/.
"""

from __future__ import annotations

import hmac
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ServerError(Exception):
    """Base error for pi_server."""

    def __init__(self, code: str, message: str, details: Any = None) -> None:
        self.code = code
        self.details = details
        super().__init__(message)


class SessionBusyError(ServerError):
    def __init__(self, session_id: str) -> None:
        super().__init__("busy", f"Session {session_id} is busy")


class SessionLockedError(ServerError):
    def __init__(self, session_id: str) -> None:
        super().__init__("session_locked", f"Session {session_id} is locked by another connection")


class SessionNotFoundError(ServerError):
    def __init__(self, session_id: str) -> None:
        super().__init__("not_found", f"Session {session_id} not found")


class AuthError(ServerError):
    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__("auth", message)


class VersionError(ServerError):
    def __init__(self, version: int) -> None:
        super().__init__("version", f"Unsupported protocol version: {version}")


@dataclass
class ServerOptions:
    """Options for creating a PiServer."""

    server_id: str = "pi-server"
    token: str | None = None
    auto_approve: bool = False


@dataclass
class SessionInfo:
    """Info about a live session."""

    id: str
    name: str | None = None
    cwd: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    phase: str = "idle"
    model: dict[str, str] = field(default_factory=lambda: {"provider": "", "id": ""})
    thinking_level: str = "off"
    attached: bool = False
    locked_by: str | None = None
    revision: int = 0


class SessionManager:
    """Manages live sessions on the server."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionInfo] = {}

    def create_session(
        self,
        *,
        cwd: str | None = None,
        name: str | None = None,
        model: dict[str, str] | None = None,
        thinking_level: str | None = None,
    ) -> SessionInfo:
        """Create a new session."""
        import uuid as _uuid

        session_id = _uuid.uuid4().hex[:12]
        session = SessionInfo(
            id=session_id,
            name=name,
            cwd=cwd or "",
            model=model or {"provider": "", "id": ""},
            thinking_level=thinking_level or "off",
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionInfo:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        if self._sessions[session_id].locked_by is not None:
            raise SessionLockedError(session_id)
        del self._sessions[session_id]

    def attach_session(self, session_id: str, connection_id: str) -> SessionInfo:
        """Attach a connection to a session."""
        session = self.get_session(session_id)
        if session.locked_by is not None and session.locked_by != connection_id:
            raise SessionLockedError(session_id)
        session.locked_by = connection_id
        session.attached = True
        session.updated_at = int(time.time() * 1000)
        return session

    def detach_session(self, session_id: str, connection_id: str) -> None:
        """Detach a connection from a session."""
        session = self.get_session(session_id)
        if session.locked_by == connection_id:
            session.locked_by = None
            session.attached = False
            session.updated_at = int(time.time() * 1000)

    def set_model(self, session_id: str, model: dict[str, str]) -> SessionInfo:
        """Set the model for a session."""
        session = self.get_session(session_id)
        session.model = model
        session.updated_at = int(time.time() * 1000)
        return session

    def set_thinking_level(self, session_id: str, level: str) -> SessionInfo:
        """Set the thinking level for a session."""
        session = self.get_session(session_id)
        session.thinking_level = level
        session.updated_at = int(time.time() * 1000)
        return session


class PiServer:
    """Pi server for managing sessions and handling client connections."""

    def __init__(self, options: ServerOptions) -> None:
        self._options = options
        self._sessions = SessionManager()
        self._connections: dict[str, Any] = {}
        self._event_listeners: list[Callable[[dict[str, Any]], None]] = []

    @property
    def server_id(self) -> str:
        return self._options.server_id

    @property
    def sessions(self) -> SessionManager:
        return self._sessions

    def verify_token(self, token: str) -> bool:
        """Verify a client token against the server's expected token."""
        if self._options.token is None:
            return True
        return hmac.compare_digest(token, self._options.token)

    def create_snapshot(self) -> dict[str, Any]:
        """Create a server snapshot."""
        return {
            "serverId": self._options.server_id,
            "protocolVersion": 2,
            "revision": 0,
            "sessions": [
                {
                    "id": s.id,
                    "name": s.name,
                    "cwd": s.cwd,
                    "createdAt": s.created_at,
                    "updatedAt": s.updated_at,
                    "phase": s.phase,
                    "model": s.model,
                    "thinkingLevel": s.thinking_level,
                    "attached": s.attached,
                    "locked": s.locked_by is not None,
                }
                for s in self._sessions.list_sessions()
            ],
            "models": [],
        }

    def on_event(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register an event listener."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _emit_event(self, event: dict[str, Any]) -> None:
        for listener in list(self._event_listeners):
            listener(event)


__all__ = [
    "AuthError",
    "PiServer",
    "ServerError",
    "ServerOptions",
    "SessionBusyError",
    "SessionInfo",
    "SessionLockedError",
    "SessionManager",
    "SessionNotFoundError",
    "VersionError",
]
