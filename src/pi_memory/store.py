"""Persistent memory store: SQLite + FTS5 + sqlite-vec hybrid search.

Separate from ``pi_storage_sqlite`` (conversation session storage) and
``pi_agent_core.session`` (session fork/branch/compaction) — this module
persists a small, curated set of facts *across* sessions (user
preferences, decisions, style), not conversation history.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pi_memory.embeddings import EMBEDDING_DIM, EmbeddingManager
from pi_memory.rrf import reciprocal_rank_fusion

try:
    import sqlite_vec

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK (type IN ('decision','style','user','feedback','project','reference')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    project_cwd TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title, content, content='memories', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;
"""

_FTS_CANDIDATES = 10
_VEC_CANDIDATES = 10

# L2 distance on the unit-normalized embeddings produced by EmbeddingManager.
# For normalized vectors, L2^2 = 2 - 2*cos_sim, so distance <= 0.5 corresponds
# to cosine similarity >= ~0.875 — close enough to flag as a likely duplicate
# without the LLM asking, while still leaving room for genuinely distinct
# memories that merely share vocabulary.
_DUPLICATE_DISTANCE_THRESHOLD = 0.5


class MemoryType(str, Enum):
    DECISION = "decision"
    STYLE = "style"
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass
class MemoryRecord:
    id: int
    type: MemoryType
    title: str
    content: str
    project_cwd: str | None
    created_at: int
    updated_at: int
    source: str


class MemoryStore:
    """SQLite-backed store for curated, persistent facts with hybrid search."""

    def __init__(self, db_path: str | Path = ":memory:", *, embeddings: EmbeddingManager | None = None) -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or EmbeddingManager.get_instance()
        # check_same_thread=False: callers (AgentTool.execute, AgentSession._recall_memories)
        # run write()/search() via run_in_executor, which may use a different worker thread
        # per call. Access is still effectively serialized — one blocking call at a time.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._vec_enabled = self._load_sqlite_vec()
        self._init_schema()

    @property
    def embeddings(self) -> EmbeddingManager:
        return self._embeddings

    def _load_sqlite_vec(self) -> bool:
        if not _SQLITE_VEC_AVAILABLE:
            return False
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception:
            return False

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        if self._vec_enabled:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                f"memory_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}])"
            )
        self._conn.commit()

    def write(
        self,
        *,
        type: MemoryType,
        title: str,
        content: str,
        project_cwd: str | None = None,
        source: str = "auto",
    ) -> MemoryRecord:
        now = int(time.time() * 1000)
        cursor = self._conn.execute(
            "INSERT INTO memories (type, title, content, project_cwd, created_at, updated_at, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (type.value, title, content, project_cwd, now, now, source),
        )
        memory_id = cursor.lastrowid
        assert memory_id is not None

        if self._vec_enabled and self._embeddings.is_available():
            try:
                vector = self._embeddings.embed(f"{title}: {content}", task="document")
                self._conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(vector)),
                )
            except Exception:
                pass

        self._conn.commit()
        return MemoryRecord(
            id=memory_id,
            type=type,
            title=title,
            content=content,
            project_cwd=project_cwd,
            created_at=now,
            updated_at=now,
            source=source,
        )

    def find_similar(
        self,
        title: str,
        content: str,
        *,
        project_cwd: str | None = None,
    ) -> tuple[MemoryRecord, float] | None:
        """Return the closest existing memory to (title, content) and its distance,
        if it looks like a likely duplicate (distance <= _DUPLICATE_DISTANCE_THRESHOLD).

        Semantic-only: requires sqlite-vec and a loaded embedding model. Returns
        None when unavailable or when nothing is close enough — this is a
        best-effort proactive nudge, not a guarantee of no duplicates.
        """
        if not (self._vec_enabled and self._embeddings.is_available()):
            return None
        try:
            vector = self._embeddings.embed(f"{title}: {content}", task="document")
        except Exception:
            return None
        rows = self._conn.execute(
            "SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (json.dumps(vector), 1),
        ).fetchall()
        if not rows:
            return None
        memory_id, distance = rows[0]["memory_id"], float(rows[0]["distance"])
        if distance > _DUPLICATE_DISTANCE_THRESHOLD:
            return None
        record = self._get(memory_id)
        if record is None:
            return None
        if project_cwd is not None and record.project_cwd not in (None, project_cwd):
            return None
        return record, distance

    def update(self, memory_id: int, *, title: str, content: str) -> MemoryRecord | None:
        """Overwrite an existing memory's title/content in place (used to merge
        a would-be duplicate into the record the user chose to keep)."""
        if self._get(memory_id) is None:
            return None
        now = int(time.time() * 1000)
        self._conn.execute(
            "UPDATE memories SET title = ?, content = ?, updated_at = ? WHERE id = ?",
            (title, content, now, memory_id),
        )
        if self._vec_enabled and self._embeddings.is_available():
            try:
                vector = self._embeddings.embed(f"{title}: {content}", task="document")
                self._conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
                self._conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(vector)),
                )
            except Exception:
                pass
        self._conn.commit()
        return self._get(memory_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        project_cwd: str | None = None,
    ) -> list[MemoryRecord]:
        if not query.strip():
            return []

        ranked_lists: list[list[int]] = [self._fts_search(query)]
        if self._vec_enabled and self._embeddings.is_available():
            try:
                vector = self._embeddings.embed(query, task="query")
                ranked_lists.append(self._vec_search(vector))
            except Exception:
                pass

        fused = reciprocal_rank_fusion(ranked_lists)
        records: list[MemoryRecord] = []
        for memory_id, _score in fused:
            record = self._get(memory_id)
            if record is None:
                continue
            if project_cwd is not None and record.project_cwd not in (None, project_cwd):
                continue
            records.append(record)
            if len(records) >= top_k:
                break
        return records

    def _fts_search(self, query: str) -> list[int]:
        try:
            rows = self._conn.execute(
                "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT ?",
                (_fts_escape(query), _FTS_CANDIDATES),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [row["rowid"] for row in rows]

    def _vec_search(self, vector: list[float]) -> list[int]:
        rows = self._conn.execute(
            "SELECT memory_id FROM memories_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (json.dumps(vector), _VEC_CANDIDATES),
        ).fetchall()
        return [row["memory_id"] for row in rows]

    def _get(self, memory_id: int) -> MemoryRecord | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        type=MemoryType(row["type"]),
        title=row["title"],
        content=row["content"],
        project_cwd=row["project_cwd"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source=row["source"],
    )


def _fts_escape(query: str) -> str:
    """Quote the raw query as a single FTS5 phrase to avoid syntax errors on
    user text containing FTS operators (``-``, ``"``, ``*``, ...)."""
    escaped = query.replace('"', '""')
    return f'"{escaped}"'
