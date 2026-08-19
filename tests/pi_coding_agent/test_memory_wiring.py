"""Tests for the memory wiring in AgentSession / system_prompt / memory tools."""

from __future__ import annotations

import asyncio
import math

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_memory.embeddings import EMBEDDING_DIM, EmbeddingManager
from pi_memory.store import MemoryStore, MemoryType
from pi_memory.tools import create_memory_tools


class _UnavailableEmbeddingManager(EmbeddingManager):
    def is_available(self) -> bool:
        return False


class _FakeEmbeddingManager(EmbeddingManager):
    """Deterministic bag-of-words embedding, mirrors pi_memory/test_store.py's fake."""

    def is_available(self) -> bool:
        return True

    def embed(self, text: str, *, task: str = "query") -> list[float]:  # type: ignore[override]
        vector = [0.0] * EMBEDDING_DIM
        for word in text.lower().split():
            idx = hash(word) % EMBEDDING_DIM
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def _setup_session(memory_store: MemoryStore | None) -> AgentSession:
    handle = faux_provider()
    handle.set_responses([faux_assistant_message("ok")])
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    return AgentSession(
        AgentSessionOptions(
            models=models,
            model=model,
            cwd="/tmp",
            memory_store=memory_store,
        )
    )


class TestSystemPromptMemoryBlock:
    def test_memories_block_included_when_present(self) -> None:
        session = _setup_session(None)
        prompt = session._build_system_prompt(["[decision] Use SQLite: chosen over Postgres"])
        assert "<memories>" in prompt
        assert "Use SQLite" in prompt

    def test_memories_block_absent_when_none(self) -> None:
        session = _setup_session(None)
        prompt = session._build_system_prompt(None)
        assert "<memories>" not in prompt

    def test_memory_policy_present_only_when_store_configured(self) -> None:
        store = MemoryStore(":memory:", embeddings=_UnavailableEmbeddingManager())
        with_store = _setup_session(store)
        without_store = _setup_session(None)
        assert "<memory_policy>" in with_store._build_system_prompt(None)
        assert "<memory_policy>" not in without_store._build_system_prompt(None)
        store.close()


class TestAgentSessionMemoryTools:
    def test_remember_and_recall_tools_registered_when_store_set(self) -> None:
        store = MemoryStore(":memory:", embeddings=_UnavailableEmbeddingManager())
        session = _setup_session(store)
        tool_names = {t.name for t in session._tools}
        assert "remember" in tool_names
        assert "recall" in tool_names
        store.close()

    def test_memory_tools_absent_without_store(self) -> None:
        session = _setup_session(None)
        tool_names = {t.name for t in session._tools}
        assert "remember" not in tool_names
        assert "recall" not in tool_names


class TestRecallMemories:
    def test_recall_memories_formats_matches(self) -> None:
        store = MemoryStore(":memory:", embeddings=_UnavailableEmbeddingManager())
        store.write(type=MemoryType.STYLE, title="Terse replies", content="User prefers terse responses.")
        session = _setup_session(store)

        memories = asyncio.run(session._recall_memories("terse"))
        assert any("Terse replies" in m for m in memories)
        store.close()

    def test_recall_memories_empty_without_store(self) -> None:
        session = _setup_session(None)
        memories = asyncio.run(session._recall_memories("anything"))
        assert memories == []


class TestMemoryToolsExecute:
    def test_remember_tool_writes_and_recall_tool_finds_it(self) -> None:
        store = MemoryStore(":memory:", embeddings=_UnavailableEmbeddingManager())
        remember_tool, recall_tool = create_memory_tools(store)
        assert remember_tool.execute is not None
        assert recall_tool.execute is not None

        remember_result = asyncio.run(
            remember_tool.execute(
                "call-1",
                {"type": "decision", "title": "Use SQLite", "content": "Chosen over Postgres for pi."},
                None,
                None,
            )
        )
        assert not any("Invalid" in getattr(b, "text", "") for b in remember_result.content)

        recall_result = asyncio.run(recall_tool.execute("call-2", {"query": "SQLite"}, None, None))
        text = "".join(getattr(b, "text", "") for b in recall_result.content)
        assert "Use SQLite" in text
        store.close()

    def test_remember_tool_rejects_invalid_type(self) -> None:
        store = MemoryStore(":memory:", embeddings=_UnavailableEmbeddingManager())
        remember_tool, _recall_tool = create_memory_tools(store)
        assert remember_tool.execute is not None

        result = asyncio.run(
            remember_tool.execute("call-1", {"type": "bogus", "title": "t", "content": "c"}, None, None)
        )
        text = "".join(getattr(b, "text", "") for b in result.content)
        assert "Invalid type" in text
        store.close()


class TestRememberDuplicateFlow:
    def test_remember_flags_possible_duplicate_instead_of_writing(self) -> None:
        store = MemoryStore(":memory:", embeddings=_FakeEmbeddingManager())
        remember_tool, _recall_tool = create_memory_tools(store)
        assert remember_tool.execute is not None

        first = asyncio.run(
            remember_tool.execute(
                "call-1",
                {"type": "style", "title": "Terse replies", "content": "User prefers terse concise replies."},
                None,
                None,
            )
        )
        assert "Remembered" in "".join(getattr(b, "text", "") for b in first.content)

        second = asyncio.run(
            remember_tool.execute(
                "call-2",
                {"type": "style", "title": "Concise replies", "content": "User prefers terse concise replies."},
                None,
                None,
            )
        )
        text = "".join(getattr(b, "text", "") for b in second.content)
        assert "POSSIBLE_DUPLICATE" in text
        assert "Terse replies" in text
        # Nothing new was written — only the original entry exists.
        results = store.search("terse concise", top_k=5)
        assert len(results) == 1
        store.close()

    def test_remember_force_true_writes_a_separate_entry(self) -> None:
        store = MemoryStore(":memory:", embeddings=_FakeEmbeddingManager())
        remember_tool, _recall_tool = create_memory_tools(store)
        assert remember_tool.execute is not None

        asyncio.run(
            remember_tool.execute(
                "call-1",
                {"type": "style", "title": "Terse replies", "content": "User prefers terse concise replies."},
                None,
                None,
            )
        )
        second = asyncio.run(
            remember_tool.execute(
                "call-2",
                {
                    "type": "style",
                    "title": "Concise replies",
                    "content": "User prefers terse concise replies.",
                    "force": True,
                },
                None,
                None,
            )
        )
        text = "".join(getattr(b, "text", "") for b in second.content)
        assert "Remembered" in text
        results = store.search("terse concise", top_k=5)
        assert len(results) == 2
        store.close()

    def test_remember_merge_id_overwrites_existing_entry(self) -> None:
        store = MemoryStore(":memory:", embeddings=_FakeEmbeddingManager())
        remember_tool, _recall_tool = create_memory_tools(store)
        assert remember_tool.execute is not None

        first_result = asyncio.run(
            remember_tool.execute(
                "call-1",
                {"type": "style", "title": "Terse replies", "content": "User prefers terse concise replies."},
                None,
                None,
            )
        )
        assert "Remembered" in "".join(getattr(b, "text", "") for b in first_result.content)
        existing = store.search("terse concise", top_k=1)[0]

        merged = asyncio.run(
            remember_tool.execute(
                "call-2",
                {
                    "type": "style",
                    "title": "Terse & direct replies",
                    "content": "User prefers terse, direct replies without preamble.",
                    "merge_id": existing.id,
                },
                None,
                None,
            )
        )
        text = "".join(getattr(b, "text", "") for b in merged.content)
        assert "Merged" in text

        results = store.search("preamble direct", top_k=5)
        assert len(results) == 1
        assert results[0].id == existing.id
        assert results[0].title == "Terse & direct replies"
        store.close()
