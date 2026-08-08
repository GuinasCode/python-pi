"""SQLite session storage for Pi.

Mirrors packages/storage/sqlite-node/src/.
Uses Python stdlib sqlite3.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionMetadata:
    """Metadata for a stored session."""

    id: str
    name: str | None = None
    cwd: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = field(default_factory=lambda: int(time.time() * 1000))
    revision: int = 0
    size: int = 0


class SQLiteSessionRepository:
    """Repository for session storage using SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                cwd TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 0,
                size INTEGER NOT NULL DEFAULT 0,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS session_entries (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                parent_seq INTEGER,
                data TEXT NOT NULL,
                PRIMARY KEY (session_id, seq),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS session_fts (
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_session_entries_session ON session_entries(session_id);
            CREATE INDEX IF NOT EXISTS idx_session_fts_session ON session_fts(session_id);
            """
        )
        self._conn.commit()

    def create_session(
        self,
        *,
        cwd: str = "",
        name: str | None = None,
    ) -> SessionMetadata:
        """Create a new session."""
        import uuid as _uuid

        session_id = _uuid.uuid4().hex[:12]
        now = int(time.time() * 1000)
        metadata = SessionMetadata(
            id=session_id,
            name=name,
            cwd=cwd,
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            "INSERT INTO sessions (id, name, cwd, created_at, updated_at, revision, size, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, name, cwd, now, now, 0, 0, "{}"),
        )
        self._conn.commit()
        return metadata

    def open_session(self, session_id: str) -> SessionMetadata | None:
        """Open a session by ID."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return SessionMetadata(
            id=row["id"],
            name=row["name"],
            cwd=row["cwd"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revision=row["revision"],
            size=row["size"],
        )

    def list_sessions(self) -> list[SessionMetadata]:
        """List all sessions."""
        rows = self._conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [
            SessionMetadata(
                id=row["id"],
                name=row["name"],
                cwd=row["cwd"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                revision=row["revision"],
                size=row["size"],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        cursor = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def append_entry(
        self,
        session_id: str,
        data: dict[str, Any],
        *,
        parent_seq: int | None = None,
    ) -> int:
        """Append an entry to a session. Returns the sequence number."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 as next_seq FROM session_entries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row["next_seq"])
        self._conn.execute(
            "INSERT INTO session_entries (session_id, seq, parent_seq, data) VALUES (?, ?, ?, ?)",
            (session_id, seq, parent_seq, json.dumps(data)),
        )
        self._conn.execute(
            "UPDATE sessions SET updated_at = ?, revision = revision + 1, size = size + 1 WHERE id = ?",
            (int(time.time() * 1000), session_id),
        )
        self._conn.commit()
        return seq

    def get_entries(
        self,
        session_id: str,
        *,
        from_seq: int = 0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get entries for a session."""
        rows = self._conn.execute(
            "SELECT seq, parent_seq, data FROM session_entries WHERE session_id = ? AND seq >= ? ORDER BY seq LIMIT ?",
            (session_id, from_seq, limit),
        ).fetchall()
        return [{"seq": row["seq"], "parent_seq": row["parent_seq"], "data": json.loads(row["data"])} for row in rows]

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search session entries."""
        sql = "SELECT session_id, seq, content FROM session_fts WHERE content LIKE ?"
        params: list[Any] = [f"%{query}%"]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [{"session_id": row["session_id"], "seq": row["seq"], "content": row["content"]} for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


__all__ = [
    "SQLiteSessionRepository",
    "SessionMetadata",
]
