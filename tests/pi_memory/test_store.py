"""Tests for pi_memory.store.

Uses a fake EmbeddingManager (deterministic, no model download) so these
tests never touch the network or the real ONNX model.
"""

from __future__ import annotations

import math

from pi_memory.embeddings import EMBEDDING_DIM, EmbeddingManager
from pi_memory.store import MemoryStore, MemoryType


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
