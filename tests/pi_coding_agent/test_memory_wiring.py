"""Tests for the memory wiring in AgentSession / system_prompt / memory tools."""

from __future__ import annotations

import asyncio

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_memory.embeddings import EmbeddingManager
from pi_memory.store import MemoryStore, MemoryType
from pi_memory.tools import create_memory_tools


class _UnavailableEmbeddingManager(EmbeddingManager):
    def is_available(self) -> bool:
        return False


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
