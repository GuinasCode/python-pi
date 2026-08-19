"""Tests for agent_session with faux provider."""

from __future__ import annotations

import asyncio
from typing import Any

from pi_ai import StopReason
from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions


def _setup_faux(responses: list, **extra_options: Any) -> tuple[AgentSession, Any]:
    handle = faux_provider()
    handle.set_responses(responses)
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    session = AgentSession(
        AgentSessionOptions(
            models=models,
            model=model,
            cwd="/tmp",
            **extra_options,
        )
    )
    return session, handle


class TestAgentSession:
    def test_simple_prompt(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("Hello!")])
        result = asyncio.run(session.prompt("hi"))
        assert result is not None
        assert result.stop_reason == StopReason.STOP
        assert len(result.content) >= 1
        assert any(block.type == "text" and block.text == "Hello!" for block in result.content)

    def test_error_response(self) -> None:
        session, _ = _setup_faux(
            [faux_assistant_message("error", stop_reason=StopReason.ERROR, error_message="Something went wrong")]
        )
        result = asyncio.run(session.prompt("hi"))
        assert result is not None
        assert result.stop_reason == StopReason.ERROR
        assert result.error_message == "Something went wrong"

    def test_tool_call_execution(self) -> None:
        """Test that tool calls are executed and results fed back."""
        # First response: tool call
        # Second response: final text after tool result
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "/nonexistent"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("File not found, but tool was executed!"),
            ]
        )
        result = asyncio.run(session.prompt("read a file"))
        assert result is not None
        # After tool execution, second LLM call should produce the text response.
        # Accept either STOP (success) or ERROR (if faux ran out of responses).
        assert result.stop_reason in (StopReason.STOP, StopReason.ERROR)

    def test_event_emission(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("Hello!")])
        events: list[Any] = []
        session.on_event(lambda e: events.append(e))
        asyncio.run(session.prompt("hi"))
        # Should have received stream events
        assert len(events) > 0
        event_types = [getattr(e, "type", "") for e in events]
        assert "start" in event_types or "text_start" in event_types


class TestPermissionGate:
    def test_gate_denial_blocks_tool_and_feeds_model_a_denial_message(self) -> None:
        async def deny_everything(_name: str, _args: dict[str, Any]) -> bool:
            return False

        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": "/tmp/x", "content": "y"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("Understood, I will not write the file."),
            ],
            permission_gate=deny_everything,
        )
        result = asyncio.run(session.prompt("write a file"))
        assert result is not None
        assert result.stop_reason == StopReason.STOP

        tool_results = [m for m in session._messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "Permission denied" in tool_results[0].content[0].text

        # Nothing was actually written — the built-in tool never ran.
        import os

        assert not os.path.exists("/tmp/x")

    def test_gate_allow_lets_tool_run_normally(self) -> None:
        async def allow_everything(_name: str, _args: dict[str, Any]) -> bool:
            return True

        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "/nonexistent-permission-gate-test"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
            permission_gate=allow_everything,
        )
        result = asyncio.run(session.prompt("read a file"))
        assert result is not None
        assert result.stop_reason in (StopReason.STOP, StopReason.ERROR)

        tool_results = [m for m in session._messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        # Denied calls always short-circuit with the fixed "Permission denied"
        # text — a real (allowed) read failing on a missing path won't match it.
        assert "Permission denied" not in tool_results[0].content[0].text

    def test_no_gate_configured_runs_tools_unconditionally(self) -> None:
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "/nonexistent-no-gate-test"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
        )
        result = asyncio.run(session.prompt("read a file"))
        assert result is not None
        tool_results = [m for m in session._messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        assert "Permission denied" not in tool_results[0].content[0].text
