"""Persistent memory store: SQLite + FTS5 + sqlite-vec hybrid search.

Separate from ``pi_storage_sqlite`` (conversation session storage) and
``pi_agent_core.session`` (session fork/branch/compaction) — this module
persists a small, curated set of facts *across* sessions (user
preferences, decisions, style), not conversation history.
"""

from __future__ import annotations

import json
import re
import shutil
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

_LEGACY_TYPES = ("decision", "style", "user", "feedback", "project", "reference")
_ALL_TYPES = (*_LEGACY_TYPES, "soul")

_FTS_TRIGGERS_SQL = """
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

# Same 3 triggers as individual single statements — conn.executescript()
# issues an implicit COMMIT before running, which would break out of the
# explicit transaction _migrate_add_soul_type() needs to stay inside, so
# the migration recreates them via conn.execute() one at a time instead.
_FTS_TRIGGER_STATEMENTS = [
    "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN "
    "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END",
    "CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN "
    "INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content); END",
    "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN "
    "INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content); "
    "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END",
]


def _memories_table_ddl(table_name: str, types: tuple[str, ...]) -> str:
    type_list = ",".join(f"'{t}'" for t in types)
    return (
        f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        f"    type TEXT NOT NULL CHECK (type IN ({type_list})),\n"
        "    title TEXT NOT NULL,\n"
        "    content TEXT NOT NULL,\n"
        "    project_cwd TEXT,\n"
        "    created_at INTEGER NOT NULL,\n"
        "    updated_at INTEGER NOT NULL,\n"
        "    source TEXT NOT NULL DEFAULT 'auto'\n"
        ");\n"
    )


# Schema created fresh (new db) always includes 'soul' from the start — the
# lazy-migration path (see MemoryStore._migrate_add_soul_type) only matters
# for pre-existing databases whose CHECK constraint predates MemoryType.SOUL.
_SCHEMA = (
    _memories_table_ddl("memories", _ALL_TYPES)
    + "\nCREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(\n"
    "    title, content, content='memories', content_rowid='id'\n"
    ");\n"
    + _FTS_TRIGGERS_SQL
)

_FTS_CANDIDATES = 10
_VEC_CANDIDATES = 10

# Best-effort, non-exhaustive patterns for obvious credential material. This
# is a minimal guard against accidental secret capture, not a DLP system —
# applies to every memory type (the gap predates Soul; Soul just made it
# worth closing).
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
]


def _looks_like_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)

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
    # Stable, high-priority, low-churn principles — see write()'s lazy
    # schema migration and MemoryStore.list_by_type() for how this differs
    # from the other, episodic types.
    SOUL = "soul"


class SecretDetectedError(ValueError):
    """Raised by write() when content/title looks like a credential."""


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
        # Cached result of the lazy 'soul' CHECK-constraint migration check
        # (None = not checked yet). See _ensure_soul_type_supported().
        self._soul_supported: bool | None = None

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

    def _ensure_soul_type_supported(self) -> None:
        """Lazily migrate a pre-existing database whose ``memories.type``
        CHECK constraint predates MemoryType.SOUL. A no-op for databases
        created fresh by this version (already have 'soul' in _SCHEMA), and
        a no-op on every call after the first successful check — this only
        does real work once, the first time a caller ever writes type=soul,
        not on every startup (a rebuild of a large real memories table is
        too expensive to risk paying unconditionally in _init_schema)."""
        if self._soul_supported:
            return
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if row is not None and row["sql"] and "'soul'" in row["sql"]:
            self._soul_supported = True
            return
        self._migrate_add_soul_type()
        self._soul_supported = True

    def _migrate_add_soul_type(self) -> None:
        """Rebuild the 'memories' table with a CHECK constraint that
        includes 'soul', preserving ids (so memories_fts and memories_vec —
        both linked by id value, not by schema reference — stay valid),
        recreating the FTS triggers (DROP TABLE auto-drops triggers defined
        on the dropped table), and validating the result before committing.
        Backs up the on-disk file first (skipped for :memory:) and restores
        it if anything looks wrong post-migration."""
        backup_path: Path | None = None
        if self._db_path != ":memory:":
            self._conn.commit()
            backup_path = Path(f"{self._db_path}.bak-pre-soul")
            shutil.copy2(self._db_path, backup_path)

        pre_count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(_memories_table_ddl("memories_new", _ALL_TYPES))
            self._conn.execute(
                "INSERT INTO memories_new (id, type, title, content, project_cwd, created_at, updated_at, source) "
                "SELECT id, type, title, content, project_cwd, created_at, updated_at, source FROM memories"
            )
            self._conn.execute("DROP TABLE memories")
            self._conn.execute("ALTER TABLE memories_new RENAME TO memories")
            for statement in _FTS_TRIGGER_STATEMENTS:
                self._conn.execute(statement)
            self._validate_soul_migration(pre_count)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            if backup_path is not None and backup_path.exists():
                self._conn.close()
                shutil.copy2(backup_path, self._db_path)
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._load_sqlite_vec()
            raise

    def _validate_soul_migration(self, pre_count: int) -> None:
        post_count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if post_count != pre_count:
            raise RuntimeError(f"soul migration row count mismatch: had {pre_count}, now {post_count}")
        integrity = self._conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"soul migration failed integrity_check: {integrity}")
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if row is None or row["sql"] is None or "'soul'" not in row["sql"]:
            raise RuntimeError("soul migration did not produce an updated CHECK constraint")
        if pre_count > 0:
            sample_id = self._conn.execute("SELECT id FROM memories LIMIT 1").fetchone()["id"]
            if self._conn.execute("SELECT id FROM memories WHERE id = ?", (sample_id,)).fetchone() is None:
                raise RuntimeError("soul migration smoke-test read failed")

    def write(
        self,
        *,
        type: MemoryType,
        title: str,
        content: str,
        project_cwd: str | None = None,
        source: str = "auto",
    ) -> MemoryRecord:
        if _looks_like_secret(title) or _looks_like_secret(content):
            raise SecretDetectedError(
                "Refusing to store: title/content looks like it contains a credential "
                "(API key, token, or private key). Remove the secret and try again."
            )
        if type is MemoryType.SOUL:
            self._ensure_soul_type_supported()
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
        if _looks_like_secret(title) or _looks_like_secret(content):
            raise SecretDetectedError(
                "Refusing to store: title/content looks like it contains a credential "
                "(API key, token, or private key). Remove the secret and try again."
            )
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

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by id. Returns False if it didn't exist. FTS
        cleanup happens via the memories_ad trigger; memories_vec (no such
        trigger) is cleaned up explicitly here."""
        if self._get(memory_id) is None:
            return False
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if self._vec_enabled:
            self._conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        self._conn.commit()
        return True

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

    def list_by_type(self, type: MemoryType, *, project_cwd: str | None = None) -> list[MemoryRecord]:
        """Return every memory of *type*, newest first — a direct read, not
        a similarity search. Used for Soul, which must be loaded with a
        guarantee, not conditioned on matching the current turn's text."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC, id DESC",
            (type.value,),
        ).fetchall()
        records = [_row_to_record(row) for row in rows]
        if project_cwd is not None:
            records = [r for r in records if r.project_cwd in (None, project_cwd)]
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
