"""Tests for pi_server."""

from __future__ import annotations

import pytest

from pi_server import (
    PiServer,
    ServerOptions,
    SessionLockedError,
    SessionManager,
    SessionNotFoundError,
)


class TestSessionManager:
    def test_create_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session(cwd="/tmp", name="test")
        assert session.id != ""
        assert session.cwd == "/tmp"
        assert session.name == "test"

    def test_get_session(self) -> None:
        mgr = SessionManager()
        created = mgr.create_session()
        session = mgr.get_session(created.id)
        assert session.id == created.id

    def test_get_nonexistent_session(self) -> None:
        mgr = SessionManager()
        with pytest.raises(SessionNotFoundError):
            mgr.get_session("nonexistent")

    def test_list_sessions(self) -> None:
        mgr = SessionManager()
        mgr.create_session(name="s1")
        mgr.create_session(name="s2")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_delete_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.delete_session(session.id)
        with pytest.raises(SessionNotFoundError):
            mgr.get_session(session.id)

    def test_delete_nonexistent_raises(self) -> None:
        mgr = SessionManager()
        with pytest.raises(SessionNotFoundError):
            mgr.delete_session("nonexistent")

    def test_attach_detach(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.attach_session(session.id, "conn-1")
        assert mgr.get_session(session.id).attached is True
        assert mgr.get_session(session.id).locked_by == "conn-1"
        mgr.detach_session(session.id, "conn-1")
        assert mgr.get_session(session.id).attached is False

    def test_attach_locked_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.attach_session(session.id, "conn-1")
        with pytest.raises(SessionLockedError):
            mgr.attach_session(session.id, "conn-2")

    def test_set_model(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.set_model(session.id, {"provider": "anthropic", "id": "claude"})
        assert mgr.get_session(session.id).model == {"provider": "anthropic", "id": "claude"}

    def test_set_thinking_level(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        mgr.set_thinking_level(session.id, "high")
        assert mgr.get_session(session.id).thinking_level == "high"


class TestPiServer:
    def test_creation(self) -> None:
        server = PiServer(ServerOptions(server_id="test-server"))
        assert server.server_id == "test-server"

    def test_verify_token_no_token(self) -> None:
        server = PiServer(ServerOptions(token=None))
        assert server.verify_token("anything") is True

    def test_verify_token_match(self) -> None:
        server = PiServer(ServerOptions(token="secret"))
        assert server.verify_token("secret") is True

    def test_verify_token_mismatch(self) -> None:
        server = PiServer(ServerOptions(token="secret"))
        assert server.verify_token("wrong") is False

    def test_create_snapshot(self) -> None:
        server = PiServer(ServerOptions(server_id="test"))
        server.sessions.create_session(name="s1")
        snapshot = server.create_snapshot()
        assert snapshot["serverId"] == "test"
        assert snapshot["protocolVersion"] == 2
        assert len(snapshot["sessions"]) == 1
        assert snapshot["sessions"][0]["name"] == "s1"

    def test_event_listener(self) -> None:
        server = PiServer(ServerOptions())
        events: list[dict] = []
        unsub = server.on_event(lambda e: events.append(e))
        server._emit_event({"type": "test"})
        assert len(events) == 1
        unsub()
        server._emit_event({"type": "test2"})
        assert len(events) == 1
