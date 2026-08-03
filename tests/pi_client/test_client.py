"""Tests for pi_client."""

from __future__ import annotations

import pytest

from pi_client import (
    ClientDisposedError,
    ClientOptions,
    ClientState,
    ConnectionState,
    PiClient,
    ServerError,
)


class TestClientState:
    def test_default_state(self) -> None:
        state = ClientState()
        assert state.sessions == {}
        assert state.connection_state == ConnectionState.DISCONNECTED
        assert state.server_id == ""
        assert state.protocol_version == 0

    def test_update_snapshot(self) -> None:
        state = ClientState()
        state.update_snapshot({
            "serverId": "server-1",
            "protocolVersion": 2,
            "sessions": [
                {
                    "id": "s1",
                    "name": "Session 1",
                    "cwd": "/tmp",
                    "createdAt": 100,
                    "updatedAt": 200,
                    "phase": "idle",
                    "model": {"provider": "test", "id": "model"},
                    "thinkingLevel": "off",
                    "attached": False,
                    "locked": False,
                }
            ],
        })
        assert state.server_id == "server-1"
        assert state.protocol_version == 2
        assert "s1" in state.sessions
        assert state.sessions["s1"].name == "Session 1"


class TestPiClient:
    def test_creation(self) -> None:
        client = PiClient(ClientOptions())
        assert client.is_disposed is False
        assert client.is_connected is False
        assert client.get_sessions() == []

    def test_dispose(self) -> None:
        client = PiClient(ClientOptions())
        client.dispose()
        assert client.is_disposed is True

    def test_disposed_client_raises(self) -> None:
        client = PiClient(ClientOptions())
        client.dispose()
        with pytest.raises(ClientDisposedError):
            import asyncio
            asyncio.run(client._send_prompt("s1", "test"))

    def test_event_listener(self) -> None:
        client = PiClient(ClientOptions())
        events: list[dict] = []
        unsub = client.on_event(lambda e: events.append(e))
        client._emit_event({"type": "test"})
        assert len(events) == 1
        unsub()
        client._emit_event({"type": "test2"})
        assert len(events) == 1

    def test_session_handle(self) -> None:
        client = PiClient(ClientOptions())
        handle = client.get_session_handle("s1")
        assert handle.session_id == "s1"

    def test_double_dispose_safe(self) -> None:
        client = PiClient(ClientOptions())
        client.dispose()
        client.dispose()
        assert client.is_disposed is True


class TestErrors:
    def test_server_error(self) -> None:
        err = ServerError("auth", "Invalid token")
        assert err.code == "auth"
        assert "Invalid token" in str(err)

    def test_server_error_with_details(self) -> None:
        err = ServerError("invalid_request", "Bad request", {"field": "name"})
        assert err.code == "invalid_request"
        assert err.details == {"field": "name"}
