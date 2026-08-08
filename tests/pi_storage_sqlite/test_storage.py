"""Tests for pi_storage_sqlite."""

from __future__ import annotations

from pi_storage_sqlite import SQLiteSessionRepository


class TestSQLiteSessionRepository:
    def test_create_session(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session(cwd="/tmp", name="test")
        assert session.id != ""
        assert session.cwd == "/tmp"
        assert session.name == "test"
        repo.close()

    def test_open_session(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        created = repo.create_session(name="test")
        opened = repo.open_session(created.id)
        assert opened is not None
        assert opened.id == created.id
        assert opened.name == "test"
        repo.close()

    def test_open_nonexistent(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        assert repo.open_session("nonexistent") is None
        repo.close()

    def test_list_sessions(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        repo.create_session(name="s1")
        repo.create_session(name="s2")
        sessions = repo.list_sessions()
        assert len(sessions) == 2
        repo.close()

    def test_delete_session(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        created = repo.create_session()
        assert repo.delete_session(created.id) is True
        assert repo.open_session(created.id) is None
        repo.close()

    def test_delete_nonexistent(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        assert repo.delete_session("nonexistent") is False
        repo.close()

    def test_append_and_get_entries(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session()
        seq1 = repo.append_entry(session.id, {"type": "user", "text": "hello"})
        seq2 = repo.append_entry(session.id, {"type": "assistant", "text": "hi"})
        assert seq1 == 0
        assert seq2 == 1
        entries = repo.get_entries(session.id)
        assert len(entries) == 2
        assert entries[0]["data"]["text"] == "hello"
        assert entries[1]["data"]["text"] == "hi"
        repo.close()

    def test_get_entries_with_limit(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session()
        for i in range(10):
            repo.append_entry(session.id, {"index": i})
        entries = repo.get_entries(session.id, limit=5)
        assert len(entries) == 5
        assert entries[0]["data"]["index"] == 0
        repo.close()

    def test_get_entries_with_offset(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session()
        for i in range(10):
            repo.append_entry(session.id, {"index": i})
        entries = repo.get_entries(session.id, from_seq=5)
        assert len(entries) == 5
        assert entries[0]["data"]["index"] == 5
        repo.close()

    def test_revision_increments(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session()
        assert session.revision == 0
        repo.append_entry(session.id, {"data": 1})
        opened = repo.open_session(session.id)
        assert opened is not None
        assert opened.revision == 1
        repo.close()

    def test_context_manager_closes_on_exit(self) -> None:
        with SQLiteSessionRepository(":memory:") as repo:
            repo.create_session()
        assert repo._closed is True

    def test_close_is_idempotent(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        repo.close()
        repo.close()  # must not raise sqlite3.ProgrammingError

    def test_search_matches_literal_percent(self) -> None:
        repo = SQLiteSessionRepository(":memory:")
        session = repo.create_session()
        repo.append_entry(session.id, {"text": "discount: 50% off"})
        repo._conn.execute(
            "INSERT INTO session_fts (session_id, seq, content) VALUES (?, 0, ?)",
            (session.id, "discount: 50% off"),
        )
        repo._conn.commit()

        results = repo.search("50%")
        assert len(results) == 1

        # A literal query with no % in the data must not match everything.
        no_match = repo.search("90%")
        assert no_match == []
        repo.close()
