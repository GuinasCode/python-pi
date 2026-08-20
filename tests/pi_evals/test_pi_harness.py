"""Tests for pi_evals.pi_harness — the AgentSession adapter for pytest-evals.

Uses the faux provider throughout, so these never touch a network or a real
LLM. _resolve_models (which auto-discovers real providers from env vars) is
monkeypatched out in the integration tests below so they stay hermetic
regardless of what API keys happen to be set in the environment.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pi_ai import AssistantMessage, Message, TextContent, ToolCall, ToolResultMessage, UserMessage
from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.agent_session import AgentSession
from pi_evals import pi_harness
from pi_evals.pi_harness import (
    PiCodingAgentHarness,
    PiCodingAgentHarnessOptions,
    PromptStep,
    ReloadStep,
    _serialize_session_snapshot,
    _to_transcript_events,
    create_pi_coding_agent_harness,
    resolve_model_selection,
)


class TestResolveModelSelection:
    def test_explicit_override_wins(self) -> None:
        provider, model_id = resolve_model_selection(("openai", "gpt-5"), {"PI_PROVIDER": "x", "PI_MODEL": "y"})
        assert (provider, model_id) == ("openai", "gpt-5")

    def test_falls_back_to_env_vars(self) -> None:
        provider, model_id = resolve_model_selection(None, {"PI_PROVIDER": "anthropic", "PI_MODEL": "claude"})
        assert (provider, model_id) == ("anthropic", "claude")

    def test_raises_when_neither_is_set(self) -> None:
        with pytest.raises(ValueError, match="Select a harness model"):
            resolve_model_selection(None, {})

    def test_raises_when_only_provider_is_set(self) -> None:
        with pytest.raises(ValueError):
            resolve_model_selection(None, {"PI_PROVIDER": "openai"})


class TestToTranscriptEvents:
    def test_normalizes_user_assistant_tool_messages(self) -> None:
        messages: list[Message] = [
            UserMessage(content="hi"),
            AssistantMessage(content=[TextContent(text="ok"), ToolCall(id="t1", name="read", arguments={"path": "x"})]),
            ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="file contents")]),
        ]
        events = _to_transcript_events(messages)
        assert events == [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "ok"},
            {"type": "tool_call", "id": "t1", "name": "read", "arguments": {"path": "x"}},
            {"type": "tool_result", "tool_call_id": "t1", "name": "read", "content": "file contents"},
        ]

    def test_tool_error_includes_error_field(self) -> None:
        messages: list[Message] = [
            ToolResultMessage(tool_call_id="t1", tool_name="bash", content=[TextContent(text="boom")], is_error=True)
        ]
        events = _to_transcript_events(messages)
        assert events[0]["error"] == {"message": "boom"}

    def test_assistant_message_with_no_text_produces_no_message_event(self) -> None:
        messages: list[Message] = [AssistantMessage(content=[ToolCall(id="t1", name="read", arguments={})])]
        events = _to_transcript_events(messages)
        assert all(e["type"] != "message" for e in events)


class TestSerializeSessionSnapshot:
    def test_round_trips_as_jsonl(self) -> None:
        messages: list[Message] = [
            UserMessage(content="hi", timestamp=1),
            AssistantMessage(content=[TextContent(text="ok")], model="faux-1", timestamp=2),
        ]
        snapshot = _serialize_session_snapshot(messages)
        lines = [json.loads(line) for line in snapshot.splitlines()]
        assert lines[0] == {"role": "user", "content": "hi", "timestamp": 1}
        assert lines[1]["role"] == "assistant"
        assert lines[1]["content"] == [{"type": "text", "text": "ok"}]


def _faux_models_and_model() -> tuple[MutableModels, object]:
    handle = faux_provider()
    handle.set_responses([faux_assistant_message(f"echo: {i}") for i in range(10)])
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    return models, model


class TestPiCodingAgentHarnessRun:
    def test_run_returns_output_events_and_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = create_pi_coding_agent_harness(model=("faux", "faux-1"))
        result = asyncio.run(harness.run("hello"))

        assert "echo: 0" in result.output
        assert any(e["type"] == "message" and e["role"] == "user" for e in result.events)
        assert result.usage.provider == "faux"
        assert result.usage.model == "faux-1"
        assert result.total_ms >= 0

    def test_output_transform_receives_response_and_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        def _output(response: str, session: AgentSession) -> dict[str, object]:
            return {"response": response, "active_tools": session.get_active_tool_names()}

        harness = PiCodingAgentHarness(PiCodingAgentHarnessOptions(model=("faux", "faux-1"), output=_output))
        result = asyncio.run(harness.run("hello"))

        assert result.output["response"] == "echo: 0"
        assert "read" in result.output["active_tools"]

    def test_no_tools_true_yields_empty_active_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = PiCodingAgentHarness(
            PiCodingAgentHarnessOptions(
                model=("faux", "faux-1"),
                no_tools=True,
                output=lambda _response, session: session.get_active_tool_names(),
            )
        )
        result = asyncio.run(harness.run("hello"))
        assert result.output == []

    def test_transform_system_prompt_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = PiCodingAgentHarness(
            PiCodingAgentHarnessOptions(
                model=("faux", "faux-1"),
                transform_system_prompt=lambda _default: "TRANSFORMED_MARKER",
                output=lambda _response, session: session.get_system_prompt(),
            )
        )
        result = asyncio.run(harness.run("hello"))
        assert result.output.startswith("TRANSFORMED_MARKER")

    def test_transform_system_prompt_rejects_empty_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = PiCodingAgentHarness(
            PiCodingAgentHarnessOptions(model=("faux", "faux-1"), transform_system_prompt=lambda _default: "   ")
        )
        with pytest.raises(ValueError, match="must not be empty"):
            asyncio.run(harness.run("hello"))

    def test_prompt_then_reload_sequence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = create_pi_coding_agent_harness(model=("faux", "faux-1"))
        steps: list[PromptStep | ReloadStep] = [
            {"type": "prompt", "content": "first"},
            {"type": "reload"},
            {"type": "prompt", "content": "second"},
        ]
        result = asyncio.run(harness.run(steps))
        assert result.output == "echo: 1"

    def test_empty_step_list_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = create_pi_coding_agent_harness(model=("faux", "faux-1"))
        reload_only: list[PromptStep | ReloadStep] = [{"type": "reload"}]
        with pytest.raises(ValueError, match="at least one prompt step"):
            asyncio.run(harness.run(reload_only))

    def test_isolated_run_does_not_leak_temp_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import tempfile
        from pathlib import Path

        before = {p.name for p in Path(tempfile.gettempdir()).glob("pi-eval-*")}
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = create_pi_coding_agent_harness(model=("faux", "faux-1"))
        asyncio.run(harness.run("hello"))

        after = {p.name for p in Path(tempfile.gettempdir()).glob("pi-eval-*")}
        assert after == before

    def test_enable_extensions_false_means_no_extension_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = PiCodingAgentHarness(
            PiCodingAgentHarnessOptions(
                model=("faux", "faux-1"),
                output=lambda _response, session: session.get_extension_paths(),
            )
        )
        result = asyncio.run(harness.run("hello"))
        assert result.output == []

    def test_enable_extensions_true_wires_a_runner_into_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        models, model = _faux_models_and_model()
        monkeypatch.setattr(pi_harness, "_resolve_models", lambda provider, model_id: (models, model))

        harness = PiCodingAgentHarness(
            PiCodingAgentHarnessOptions(
                model=("faux", "faux-1"),
                enable_extensions=True,
                output=lambda _response, session: session.get_extensions(),
            )
        )
        result = asyncio.run(harness.run("hello"))
        # No extension files exist in the throwaway cwd, so an empty (not
        # None/error-raising) result confirms the runner is really wired in.
        assert result.output.extensions == []
        assert result.output.errors == []
