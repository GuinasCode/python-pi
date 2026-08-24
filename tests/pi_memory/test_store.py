"""Tests for pi_memory.store.

Uses a fake EmbeddingManager (deterministic, no model download) so these
tests never touch the network or the real ONNX model.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from pi_memory.embeddings import EMBEDDING_DIM, EmbeddingManager
from pi_memory.store import SecretDetectedError, MemoryStore, MemoryType, _LEGACY_TYPES, _memories_table_ddl


class FakeEmbeddingManager(EmbeddingManager):
    """Deterministic bag-of-words embedding, good enough to test ranking."""

    def is_available(self) -> bool:
        return True

    def embed(self, text: str, *, task: str = "query") -> list[float]:  # type: ignore[override]
        vector = [0.0] * EMBEDDING_DIM
        for word in text.lower().split():
            idx = hash(word) % EMBEDDING_DIM
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class UnavailableEmbeddingManager(EmbeddingManager):
    def is_available(self) -> bool:
        return False


class TestMemoryStoreFtsOnly:
    """Without embeddings available, search falls back to FTS5 keyword matching."""

    def test_write_and_search(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.STYLE, title="Terse replies", content="User prefers terse responses.")
        store.write(type=MemoryType.DECISION, title="Use SQLite", content="Chose SQLite over Postgres for pi.")

        results = store.search("terse", top_k=3)
        assert len(results) == 1
        assert results[0].title == "Terse replies"
        store.close()

    def test_search_empty_query_returns_nothing(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.USER, title="x", content="y")
        assert store.search("   ") == []
        store.close()

    def test_search_no_matches(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.USER, title="x", content="y")
        assert store.search("zzz_no_such_term") == []
        store.close()


class TestMemoryStoreHybridSearch:
    """With a fake embedding backend available, hybrid FTS+vector search runs."""

    def test_semantic_match_without_exact_keyword(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        store.write(
            type=MemoryType.STYLE,
            title="Concise answers",
            content="The user wants short concise terse replies always.",
        )
        store.write(type=MemoryType.PROJECT, title="CI setup", content="Runs pytest and mypy on every push.")

        # Query overlaps in vocabulary with the first memory only.
        results = store.search("terse concise replies", top_k=3)
        assert len(results) >= 1
        assert results[0].title == "Concise answers"
        store.close()

    def test_top_k_limits_results(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        for i in range(5):
            store.write(type=MemoryType.REFERENCE, title=f"doc {i}", content=f"reference document number {i}")
        results = store.search("reference document", top_k=2)
        assert len(results) <= 2
        store.close()


class TestMemoryStoreScoping:
    def test_project_cwd_filters_results(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.PROJECT, title="a", content="alpha project fact", project_cwd="/proj/a")
        store.write(type=MemoryType.PROJECT, title="b", content="alpha project fact", project_cwd="/proj/b")

        results = store.search("alpha", top_k=5, project_cwd="/proj/a")
        titles = {r.title for r in results}
        assert titles == {"a"}
        store.close()

    def test_global_project_cwd_none_always_included(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.USER, title="global fact", content="alpha global preference")
        results = store.search("alpha", top_k=5, project_cwd="/proj/a")
        assert any(r.title == "global fact" for r in results)
        store.close()


class TestMemoryStoreWrite:
    def test_write_returns_populated_record(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.DECISION, title="t", content="c")
        assert record.id > 0
        assert record.type == MemoryType.DECISION
        assert record.source == "auto"
        store.close()

    def test_context_manager_closes(self) -> None:
        with MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager()) as store:
            store.write(type=MemoryType.USER, title="t", content="c")


class TestMemoryStoreFindSimilar:
    def test_finds_close_semantic_match(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        existing = store.write(
            type=MemoryType.STYLE,
            title="Terse replies",
            content="User prefers terse concise replies always.",
        )

        result = store.find_similar("Terse replies", "User prefers terse concise replies always.")
        assert result is not None
        record, distance = result
        assert record.id == existing.id
        assert distance <= 0.5
        store.close()

    def test_no_match_when_unrelated(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        store.write(type=MemoryType.STYLE, title="Terse replies", content="User prefers terse responses.")

        result = store.find_similar("CI setup", "Runs pytest and mypy on every push to main.")
        assert result is None
        store.close()

    def test_returns_none_without_embeddings(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.STYLE, title="Terse replies", content="User prefers terse responses.")

        assert store.find_similar("Terse replies", "User prefers terse responses.") is None
        store.close()


class TestMemoryStoreUpdate:
    def test_update_overwrites_title_and_content(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.STYLE, title="Terse", content="Prefers terse replies.")

        updated = store.update(record.id, title="Terse & direct", content="Prefers terse, direct replies.")
        assert updated is not None
        assert updated.id == record.id
        assert updated.title == "Terse & direct"
        assert updated.content == "Prefers terse, direct replies."
        store.close()

    def test_update_reembeds_for_vector_search(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        record = store.write(type=MemoryType.STYLE, title="Old title", content="Old unrelated content.")

        store.update(record.id, title="CI setup", content="Runs pytest and mypy on every push to main.")
        results = store.search("pytest mypy push", top_k=3)
        assert any(r.id == record.id for r in results)
        store.close()

    def test_update_missing_id_returns_none(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        assert store.update(999, title="x", content="y") is None
        store.close()


class TestMemoryStoreDelete:
    def test_delete_removes_record_and_from_search(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.SOUL, title="Removable", content="delete me later")
        assert store.delete(record.id) is True
        assert store.search("Removable", top_k=5) == []
        assert store.list_by_type(MemoryType.SOUL) == []
        store.close()

    def test_delete_missing_id_returns_false(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        assert store.delete(999) is False
        store.close()


class TestMemoryStoreSoul:
    def test_list_by_type_empty(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        assert store.list_by_type(MemoryType.SOUL) == []
        store.close()

    def test_list_by_type_one_entry(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="Be concise", content="Prefer short answers.")
        results = store.list_by_type(MemoryType.SOUL)
        assert len(results) == 1
        assert results[0].title == "Be concise"
        store.close()

    def test_list_by_type_multiple_entries_newest_first(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="First", content="a")
        store.write(type=MemoryType.SOUL, title="Second", content="b")
        results = store.list_by_type(MemoryType.SOUL)
        assert [r.title for r in results] == ["Second", "First"]
        store.close()

    def test_list_by_type_does_not_leak_other_types(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="Soul entry", content="a")
        store.write(type=MemoryType.USER, title="User entry", content="b")
        results = store.list_by_type(MemoryType.SOUL)
        assert [r.title for r in results] == ["Soul entry"]
        store.close()

    def test_list_by_type_no_embeddings_required(self) -> None:
        """Soul loading must not depend on sqlite-vec/embeddings being available."""
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="x", content="y")
        assert len(store.list_by_type(MemoryType.SOUL)) == 1
        store.close()

    def test_soul_persists_across_reopening_the_same_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        store = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="Persistent principle", content="Always verify before deleting.")
        store.close()

        reopened = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        results = reopened.list_by_type(MemoryType.SOUL)
        assert len(results) == 1
        assert results[0].title == "Persistent principle"
        reopened.close()

    def test_project_cwd_scoping_applies_to_soul(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="global", content="a")
        store.write(type=MemoryType.SOUL, title="scoped", content="b", project_cwd="/proj/a")
        results = store.list_by_type(MemoryType.SOUL, project_cwd="/proj/b")
        assert [r.title for r in results] == ["global"]
        store.close()

    def test_common_memory_write_and_search_unaffected_by_soul_support(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.DECISION, title="Use SQLite", content="Chosen over Postgres.")
        store.write(type=MemoryType.SOUL, title="Be terse", content="Prefer short answers.")
        results = store.search("SQLite", top_k=3)
        assert len(results) == 1
        assert results[0].type == MemoryType.DECISION
        store.close()


class TestSoulOverlapDetection:
    def test_no_matches_when_soul_is_empty(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        assert store.find_overlapping_soul("Prefer concise answers.") == []
        store.close()

    def test_finds_near_duplicate_text(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        original = store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")
        matches = store.find_overlapping_soul("I prefer concise answers please.")
        assert len(matches) == 1
        assert matches[0][0].id == original.id
        assert matches[0][1] > 0.6
        store.close()

    def test_no_match_for_unrelated_text(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")
        assert store.find_overlapping_soul("Always separate requirements from assumptions.") == []
        store.close()

    def test_excludes_given_id(self) -> None:
        """Editing an entry in place must not flag it as overlapping itself."""
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")
        matches = store.find_overlapping_soul("I prefer concise answers.", exclude_id=record.id)
        assert matches == []
        store.close()

    def test_only_compares_against_soul_type(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.DECISION, title="Be concise", content="I prefer concise answers.")
        assert store.find_overlapping_soul("I prefer concise answers.") == []
        store.close()

    def test_never_depends_on_embeddings(self) -> None:
        """Deterministic text overlap must work with no vector search at all."""
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        assert store._vec_enabled is False or not store._embeddings.is_available()
        store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")
        matches = store.find_overlapping_soul("I prefer concise answers now.")
        assert len(matches) == 1
        store.close()

    def test_multiple_matches_sorted_by_ratio_descending(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="A", content="I prefer concise answers.")
        store.write(type=MemoryType.SOUL, title="B", content="I prefer concise answers always.")
        matches = store.find_overlapping_soul("I prefer concise answers.")
        assert len(matches) == 2
        assert matches[0][1] >= matches[1][1]
        store.close()


class TestSecretGuard:
    def test_refuses_openai_style_key_in_content(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        try:
            store.write(
                type=MemoryType.USER,
                title="API key",
                content="key is sk-abcdefghijklmnopqrstuvwx1234567890",
            )
            raised = False
        except SecretDetectedError:
            raised = True
        assert raised
        assert store.search("API key", top_k=5) == []
        store.close()

    def test_refuses_secret_regardless_of_type(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        try:
            store.write(type=MemoryType.SOUL, title="token", content="ghp_" + "a" * 36)
            raised = False
        except SecretDetectedError:
            raised = True
        assert raised
        store.close()

    def test_allows_ordinary_content(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.USER, title="x", content="The user prefers dark mode.")
        assert record.id > 0
        store.close()


def _make_legacy_db(db_path: Path) -> None:
    """Build a memories table using the pre-Soul CHECK constraint, to
    simulate a real on-disk database created before this migration existed."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_memories_table_ddl("memories", _LEGACY_TYPES))
    conn.executescript(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
        "title, content, content='memories', content_rowid='id');"
        "CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN "
        "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END;"
        "CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN "
        "INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content); END;"
        "CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN "
        "INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES ('delete', old.id, old.title, old.content); "
        "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END;"
    )
    conn.execute(
        "INSERT INTO memories (type, title, content, project_cwd, created_at, updated_at, source) "
        "VALUES ('decision', 'Legacy decision', 'Chosen before Soul existed.', NULL, 1000, 1000, 'auto')"
    )
    conn.commit()
    conn.close()


class TestSoulSchemaMigration:
    def test_write_soul_migrates_legacy_db_and_preserves_existing_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        _make_legacy_db(db_path)

        store = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        legacy_before = store.search("Legacy decision", top_k=5)
        assert len(legacy_before) == 1
        legacy_id = legacy_before[0].id

        record = store.write(type=MemoryType.SOUL, title="New principle", content="Always ask before deleting.")
        assert record.id > 0

        legacy_after = store.search("Legacy decision", top_k=5)
        assert len(legacy_after) == 1
        assert legacy_after[0].id == legacy_id
        assert legacy_after[0].title == "Legacy decision"
        store.close()

    def test_fts_keeps_working_for_writes_made_after_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        _make_legacy_db(db_path)

        store = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="trigger migration", content="first soul entry")
        store.write(type=MemoryType.DECISION, title="Post-migration decision", content="Written after the rebuild.")

        results = store.search("Post-migration", top_k=5)
        assert any(r.title == "Post-migration decision" for r in results)
        store.close()

    def test_migration_creates_backup_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        _make_legacy_db(db_path)

        store = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="x", content="y")
        store.close()

        assert Path(f"{db_path}.bak-pre-soul").exists()

    def test_soul_write_on_fresh_db_needs_no_migration(self) -> None:
        """A brand-new database already has 'soul' in its CHECK constraint
        (created via the current _SCHEMA), so writing type=soul must not
        attempt any rebuild."""
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        record = store.write(type=MemoryType.SOUL, title="x", content="y")
        assert record.id > 0
        store.close()

    def test_migration_is_idempotent_across_multiple_soul_writes(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        _make_legacy_db(db_path)

        store = MemoryStore(db_path, embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.SOUL, title="first", content="a")
        store.write(type=MemoryType.SOUL, title="second", content="b")
        results = store.list_by_type(MemoryType.SOUL)
        assert len(results) == 2
        store.close()
