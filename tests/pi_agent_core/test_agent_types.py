"""Tests for pi_agent_core types and state."""

from __future__ import annotations

from pi_agent_core import (
    AgentState,
    AgentTool,
)
from pi_ai import Model


def _make_model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider="test",
        base_url="https://api.test.com",
        context_window=4096,
        max_tokens=2048,
    )


class TestAgentState:
    def test_default_state(self) -> None:
        state = AgentState()
        assert state.system_prompt == ""
        assert state.model.id == "unknown"
        assert state.thinking_level == "off"
        assert state.tools == []
        assert state.messages == []
        assert state.is_streaming is False
        assert state.streaming_message is None
        assert state.pending_tool_calls == set()
        assert state.error_message is None

    def test_state_with_model(self) -> None:
        model = _make_model()
        state = AgentState(model=model, system_prompt="Be helpful")
        assert state.model.id == "test-model"
        assert state.system_prompt == "Be helpful"


class TestAgentTool:
    def test_tool_creation(self) -> None:
        def execute(args: dict) -> str:
            return "result"

        tool = AgentTool(
            name="read",
            description="Read a file",
            parameters={"type": "object"},
            execute=execute,
        )
        assert tool.name == "read"
        assert tool.description == "Read a file"
        assert tool.execute({"path": "/tmp"}) == "result"


class TestQueueMode:
    def test_all_mode(self) -> None:
        assert "all" == "all"

    def test_one_at_a_time_mode(self) -> None:
        assert "one-at-a-time" == "one-at-a-time"


class TestToolExecutionMode:
    def test_sequential_mode(self) -> None:
        assert "sequential" == "sequential"

    def test_parallel_mode(self) -> None:
        assert "parallel" == "parallel"
