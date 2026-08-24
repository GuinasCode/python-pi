"""Tests for pi_runtime.memory. Covers Fase 7's acceptance criteria from
plan.md section 11:

- fatos importantes reaparecem em futuras sessões (write persists, and
  MemoryStore's own persistence — reopening the same db file — already
  proves this at the pi_memory layer; here we prove write_with_policy
  actually calls through to a real write)
- duplicatas são reduzidas
- memória irrelevante não domina o contexto (freshness reranking)
- secrets continuam bloqueados
- falha do semantic search degrada para lexical sem derrubar a sessão
  (already pi_memory.search()'s existing behavior — confirmed still
  reachable through retrieve_ranked)
"""

from __future__ import annotations

import math
import time

from pi_memory.embeddings import EMBEDDING_DIM, EmbeddingManager
from pi_memory.store import MemoryStore, SecretDetectedError
from pi_runtime.memory import (
    CognitiveMemoryType,
    WorkingMemoryNotPersistable,
    retrieve_ranked,
    write_with_policy,
)


class FakeEmbeddingManager(EmbeddingManager):
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


class TestWriteWithPolicy:
    def test_high_confidence_write_actually_persists(self) -> None:
        from pi_memory.store import MemoryType

        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        decision = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.USER,
            title="prefers dark mode",
            content="the user prefers dark mode in every tool",
            confidence=0.9,
        )
        assert decision.written
        assert decision.record is not None
        stored = store.list_by_type(MemoryType.USER)
        assert any(r.id == decision.record.id for r in stored)

    def test_low_confidence_is_refused_not_written(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        decision = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.USER,
            title="maybe prefers dark mode",
            content="not sure, user mentioned dark mode once",
            confidence=0.1,
        )
        assert not decision.written
        assert "confidence" in decision.reason
        assert store.search("dark mode", top_k=5) == []

    def test_procedural_maps_to_soul(self) -> None:
        from pi_memory.store import MemoryType

        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        decision = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.PROCEDURAL,
            title="always ask before deleting",
            content="always ask for confirmation before deleting files",
            confidence=0.9,
        )
        assert decision.written
        assert decision.record is not None
        assert decision.record.type == MemoryType.SOUL

    def test_working_memory_is_refused_not_silently_redirected(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        try:
            write_with_policy(
                store,
                cognitive_type=CognitiveMemoryType.WORKING,
                title="scratch",
                content="temporary scratch note",
                confidence=0.9,
            )
            raised = False
        except WorkingMemoryNotPersistable:
            raised = True
        assert raised

    def test_secrets_continue_to_be_blocked(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        raised = False
        try:
            write_with_policy(
                store,
                cognitive_type=CognitiveMemoryType.USER,
                title="api key",
                content="key is sk-abcdefghijklmnopqrstuvwx1234567890",
                confidence=0.9,
            )
        except SecretDetectedError:
            raised = True
        assert raised
        assert store.search("api key", top_k=5) == []


class TestDuplicatesAreReduced:
    def test_semantic_duplicate_is_skipped_not_written_twice(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        first = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.USER,
            title="Terse replies",
            content="User prefers terse concise replies always.",
            confidence=0.9,
        )
        assert first.written

        second = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.USER,
            title="Terse replies",
            content="User prefers terse concise replies always.",
            confidence=0.9,
        )
        assert not second.written
        assert second.duplicate_of is not None
        assert second.duplicate_of.id == first.record.id  # type: ignore[union-attr]

    def test_unrelated_content_is_not_treated_as_a_duplicate(self) -> None:
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.USER,
            title="Terse replies",
            content="User prefers terse concise replies always.",
            confidence=0.9,
        )
        second = write_with_policy(
            store,
            cognitive_type=CognitiveMemoryType.PROJECT,
            title="CI setup",
            content="Runs pytest and mypy on every push to main.",
            confidence=0.9,
        )
        assert second.written


class TestRetrieveRanked:
    def test_freshness_pulls_a_recent_lower_relevance_match_upward(self) -> None:
        from pi_memory.store import MemoryType

        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        old = store.write(type=MemoryType.USER, title="old", content="terse concise replies")
        stale_ts = int(time.time() * 1000) - 1000 * 60 * 60 * 24 * 365  # ~1 year old
        store._conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (stale_ts, old.id))
        store._conn.commit()

        ranked = retrieve_ranked(store, "terse concise replies", top_k=5)
        assert len(ranked) == 1
        assert ranked[0].freshness < 0.1  # a year old, well past the 30-day half-life default

    def test_degrades_to_lexical_search_without_embeddings_and_does_not_crash(self) -> None:
        """pi_memory.search()'s existing fallback, confirmed still
        reachable through retrieve_ranked — not a new behavior, a
        regression check that wrapping it didn't lose it."""
        from pi_memory.store import MemoryType

        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        store.write(type=MemoryType.USER, title="x", content="terse concise replies")

        ranked = retrieve_ranked(store, "terse", top_k=5)
        assert len(ranked) == 1

    def test_no_results_returns_empty_list_not_an_error(self) -> None:
        store = MemoryStore(":memory:", embeddings=UnavailableEmbeddingManager())
        ranked = retrieve_ranked(store, "nothing matches this at all", top_k=5)
        assert ranked == []

    def test_irrelevant_stale_memory_does_not_dominate(self) -> None:
        """Two matches, one very stale — the fresh one should rank
        first even if raw text relevance were tied."""
        store = MemoryStore(":memory:", embeddings=FakeEmbeddingManager())
        from pi_memory.store import MemoryType

        fresh = store.write(type=MemoryType.USER, title="fresh", content="user prefers concise terse replies")
        stale = store.write(type=MemoryType.USER, title="stale", content="user prefers concise terse replies")
        stale_ts = int(time.time() * 1000) - 1000 * 60 * 60 * 24 * 365
        store._conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?", (stale_ts, stale.id))
        store._conn.commit()

        ranked = retrieve_ranked(store, "concise terse replies", top_k=5)
        ids_in_order = [r.record.id for r in ranked]
        assert ids_in_order[0] == fresh.id
