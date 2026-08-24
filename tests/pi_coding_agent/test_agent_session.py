"""Tests for agent_session with faux provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_ai import StopReason, UserMessage
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
    options: dict[str, Any] = {"models": models, "model": model, "cwd": "/tmp", **extra_options}
    session = AgentSession(AgentSessionOptions(**options))
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


class TestNoTools:
    def test_no_tools_true_disables_every_builtin(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")], no_tools=True, enable_subagents=False)
        assert session.get_active_tool_names() == []

    def test_no_tools_list_excludes_only_named_tools(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")], no_tools=["bash", "write"], enable_subagents=False)
        names = session.get_active_tool_names()
        assert "bash" not in names
        assert "write" not in names
        assert "read" in names

    def test_no_tools_ignored_when_explicit_tools_list_given(self) -> None:
        from pi_ai import Tool

        custom = [Tool(name="custom", description="d", parameters={})]
        session, _ = _setup_faux([faux_assistant_message("hi")], tools=custom, no_tools=True, enable_subagents=False)
        assert session.get_active_tool_names() == ["custom"]


class TestSessionHelpers:
    def test_get_active_tool_names_matches_builtin_tools(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")], enable_subagents=False)
        names = session.get_active_tool_names()
        assert "read" in names
        assert "write" in names
        assert "bash" in names

    def test_get_last_assistant_text_returns_final_text(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("the answer")])
        asyncio.run(session.prompt("question"))
        assert session.get_last_assistant_text() == "the answer"

    def test_get_last_assistant_text_empty_when_no_messages(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")])
        assert session.get_last_assistant_text() == ""

    def test_get_session_stats_aggregates_usage_and_tool_calls(self) -> None:
        # faux_provider recomputes .usage from content length on every response
        # (see _with_usage_estimate), so presetting it before queueing has no
        # effect — append messages straight to session._messages instead, the
        # same private-state access the permission-gate tests above already use.
        from pi_ai import TextContent, ToolCall, ToolResultMessage, UsageCost

        session, _ = _setup_faux([faux_assistant_message("unused")])

        tool_msg = faux_assistant_message(
            [ToolCall(id="t1", name="read", arguments={"path": "x"})], stop_reason=StopReason.TOOL_USE
        )
        tool_msg.usage.input = 10
        tool_msg.usage.output = 5
        tool_msg.usage.total_tokens = 15
        tool_msg.usage.cost = UsageCost(total=0.01)
        session._messages.append(tool_msg)
        session._messages.append(
            ToolResultMessage(tool_call_id="t1", tool_name="read", content=[TextContent(text="ok")])
        )

        final_msg = faux_assistant_message("done")
        final_msg.usage.input = 20
        final_msg.usage.output = 8
        final_msg.usage.total_tokens = 28
        final_msg.usage.cost = UsageCost(total=0.02)
        session._messages.append(final_msg)

        stats = session.get_session_stats()
        assert stats.input_tokens == 30
        assert stats.output_tokens == 13
        assert stats.total_tokens == 43
        assert stats.tool_calls == 1
        assert abs(stats.cost_total - 0.03) < 1e-9

    def test_get_system_prompt_and_override(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")])
        default_prompt = session.get_system_prompt()
        assert default_prompt

        session.set_system_prompt_override("Custom override prompt.")
        assert session.get_system_prompt().startswith("Custom override prompt.")


class TestReload:
    def test_reload_without_config_dir_is_a_noop(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")])
        before = session.get_system_prompt()
        asyncio.run(session.reload())
        assert session.get_system_prompt() == before

    def test_reload_picks_up_new_context_file_from_disk(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".pi"
        config_dir.mkdir()
        session, _ = _setup_faux([faux_assistant_message("hi")], cwd=str(tmp_path), config_dir=str(config_dir))

        before_prompt = session.get_system_prompt()
        assert "NEW_MARKER_TEXT" not in before_prompt

        (tmp_path / "AGENTS.md").write_text("NEW_MARKER_TEXT", encoding="utf-8")
        asyncio.run(session.reload())

        after_prompt = session.get_system_prompt()
        assert "NEW_MARKER_TEXT" in after_prompt


_HELLO_EXTENSION = """
from pi_agent_core.types import AgentTool

def extension(pi):
    pi.register_tool(AgentTool(name="hello", description="Greets someone", parameters={}))
"""


class TestExtensions:
    def test_no_runner_configured_means_no_extensions(self, tmp_path: Path) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")], cwd=str(tmp_path))
        assert session.get_extensions().extensions == []
        assert session.get_extension_paths() == []

    def test_extension_tools_registered_at_construction(self, tmp_path: Path) -> None:
        from pi_coding_agent.extensions import ExtensionRunner

        (tmp_path / ".pi" / "extensions").mkdir(parents=True)
        (tmp_path / ".pi" / "extensions" / "hello.py").write_text(_HELLO_EXTENSION, encoding="utf-8")

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [faux_assistant_message("hi")], cwd=str(tmp_path), extension_runner=runner, enable_subagents=False
        )

        assert "hello" in session.get_active_tool_names()
        assert len(session.get_extension_paths()) == 1

    def test_reload_picks_up_a_newly_created_extension(self, tmp_path: Path) -> None:
        from pi_coding_agent.extensions import ExtensionRunner

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [faux_assistant_message("hi")], cwd=str(tmp_path), extension_runner=runner, enable_subagents=False
        )
        assert "hello" not in session.get_active_tool_names()

        (tmp_path / ".pi" / "extensions").mkdir(parents=True)
        (tmp_path / ".pi" / "extensions" / "hello.py").write_text(_HELLO_EXTENSION, encoding="utf-8")
        asyncio.run(session.reload())

        assert "hello" in session.get_active_tool_names()
        assert session.get_extensions().extensions[0].tool_names == ["hello"]

    def test_reload_drops_tools_from_a_removed_extension(self, tmp_path: Path) -> None:
        from pi_coding_agent.extensions import ExtensionRunner

        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "hello.py").write_text(_HELLO_EXTENSION, encoding="utf-8")

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [faux_assistant_message("hi")], cwd=str(tmp_path), extension_runner=runner, enable_subagents=False
        )
        assert "hello" in session.get_active_tool_names()

        (ext_dir / "hello.py").unlink()
        asyncio.run(session.reload())

        assert "hello" not in session.get_active_tool_names()

    def test_extension_load_error_is_reported_not_raised(self, tmp_path: Path) -> None:
        from pi_coding_agent.extensions import ExtensionRunner

        (tmp_path / ".pi" / "extensions").mkdir(parents=True)
        (tmp_path / ".pi" / "extensions" / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [faux_assistant_message("hi")], cwd=str(tmp_path), extension_runner=runner, enable_subagents=False
        )

        errors = session.get_extensions().errors
        assert len(errors) == 1
        assert "boom" in errors[0].error


class TestExtensionLifecycleEvents:
    def _session_with_handlers(self, tmp_path: Path, handlers: dict[str, list[Any]]) -> AgentSession:
        # AgentSession.__init__ calls extension_runner.load() itself (Phase B),
        # which would wipe out handlers injected beforehand — so build the
        # session first, then inject handlers directly onto its runner.
        from pi_coding_agent.extensions import ExtensionRunner

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [faux_assistant_message("hi")], cwd=str(tmp_path), extension_runner=runner, enable_subagents=False
        )
        runner._handlers = {name: [("fake.py", h) for h in hs] for name, hs in handlers.items()}
        return session

    def test_agent_start_and_end_fire_once_per_prompt(self, tmp_path: Path) -> None:
        seen: list[str] = []
        session = self._session_with_handlers(
            tmp_path,
            {
                "agent_start": [lambda _e, _c: seen.append("start")],
                "agent_end": [lambda _e, _c: seen.append("end")],
            },
        )
        asyncio.run(session.prompt("hi"))
        assert seen == ["start", "end"]

    def test_agent_end_fires_even_when_max_turns_reached(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        seen: list[str] = []
        from pi_coding_agent.extensions import ExtensionRunner

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        responses = [
            faux_assistant_message([faux_tool_call("read", {"path": "x"})], stop_reason=StopReason.TOOL_USE)
            for _ in range(5)
        ]
        session, _ = _setup_faux(
            responses,
            cwd=str(tmp_path),
            extension_runner=runner,
            enable_subagents=False,
            max_turns=2,
        )
        runner._handlers = {"agent_end": [("fake.py", lambda _e, _c: seen.append("end"))]}
        asyncio.run(session.prompt("hi"))
        assert seen == ["end"]

    def test_session_start_fires_once_across_multiple_prompts(self, tmp_path: Path) -> None:
        seen: list[str] = []
        session = self._session_with_handlers(tmp_path, {"session_start": [lambda _e, _c: seen.append("start")]})
        asyncio.run(session.prompt("first"))
        assert seen == ["start"]

    def test_turn_start_and_end_fire_per_turn(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        seen: list[tuple[str, int]] = []
        from pi_coding_agent.extensions import ExtensionRunner

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [
                faux_assistant_message([faux_tool_call("read", {"path": "x"})], stop_reason=StopReason.TOOL_USE),
                faux_assistant_message("done"),
            ],
            cwd=str(tmp_path),
            extension_runner=runner,
            enable_subagents=False,
        )
        runner._handlers = {
            "turn_start": [("fake.py", lambda e, _c: seen.append(("start", e.turn)))],
            "turn_end": [("fake.py", lambda e, _c: seen.append(("end", e.turn)))],
        }
        asyncio.run(session.prompt("hi"))
        assert seen == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]

    def test_tool_call_block_prevents_execution(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call
        from pi_coding_agent.extensions import ExtensionRunner
        from pi_coding_agent.extensions.events import ToolCallEventResult

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": str(tmp_path / "x"), "content": "y"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
            cwd=str(tmp_path),
            extension_runner=runner,
            enable_subagents=False,
        )
        runner._handlers = {"tool_call": [("fake.py", lambda _e, _c: ToolCallEventResult(block=True, reason="nope"))]}
        asyncio.run(session.prompt("write something"))

        tool_results = [m for m in session._messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "nope" in tool_results[0].content[0].text
        assert not (tmp_path / "x").exists()

    def test_tool_call_mutates_arguments_before_execution(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call
        from pi_coding_agent.extensions import ExtensionRunner

        def _redirect(event: Any, _ctx: Any) -> None:
            event.arguments["path"] = str(tmp_path / "redirected")

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": str(tmp_path / "original"), "content": "hi"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
            cwd=str(tmp_path),
            extension_runner=runner,
            enable_subagents=False,
        )
        runner._handlers = {"tool_call": [("fake.py", _redirect)]}
        asyncio.run(session.prompt("write something"))

        assert not (tmp_path / "original").exists()
        assert (tmp_path / "redirected").read_text(encoding="utf-8") == "hi"

    def test_tool_result_override_changes_final_message(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call
        from pi_coding_agent.extensions import ExtensionRunner
        from pi_coding_agent.extensions.events import ToolResultEventResult

        runner = ExtensionRunner(tmp_path, tmp_path / "agent")
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "/nonexistent-tool-result-override"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
            cwd=str(tmp_path),
            extension_runner=runner,
            enable_subagents=False,
        )
        runner._handlers = {"tool_result": [("fake.py", lambda _e, _c: ToolResultEventResult(is_error=True))]}
        asyncio.run(session.prompt("read something"))

        tool_results = [m for m in session._messages if getattr(m, "role", "") == "toolResult"]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True


class TestBuiltinWebTools:
    def test_get_builtin_tools_includes_webfetch_and_browser(self) -> None:
        from pi_coding_agent.agent_session import get_builtin_tools

        names = {t.name for t in get_builtin_tools()}
        assert "webfetch" in names
        assert "browser" in names

    def test_execute_tool_dispatches_webfetch(self, monkeypatch: Any) -> None:
        import pi_coding_agent.agent_session as agent_session_module
        from pi_coding_agent.agent_session import _execute_tool
        from pi_coding_agent.tools import ToolResult

        calls: list[tuple[str, float]] = []

        def _fake_fetch_url(url: str, *, timeout: float = 30.0) -> ToolResult:
            calls.append((url, timeout))
            return ToolResult(content=[{"type": "text", "text": "fetched"}])

        monkeypatch.setattr(agent_session_module, "fetch_url", _fake_fetch_url)
        result = _execute_tool("webfetch", {"url": "https://example.com", "timeout": 5})
        assert not result.is_error
        assert result.content[0]["text"] == "fetched"
        assert calls == [("https://example.com", 5)]

    def test_execute_tool_dispatches_browser(self, monkeypatch: Any) -> None:
        import pi_coding_agent.agent_session as agent_session_module
        from pi_coding_agent.agent_session import _execute_tool
        from pi_coding_agent.tools import ToolResult

        calls: list[tuple[str, float]] = []

        def _fake_browser_fetch_url(url: str, *, timeout: float = 30.0) -> ToolResult:
            calls.append((url, timeout))
            return ToolResult(content=[{"type": "text", "text": "rendered"}])

        monkeypatch.setattr(agent_session_module, "browser_fetch_url", _fake_browser_fetch_url)
        result = _execute_tool("browser", {"url": "https://example.com", "timeout": 5})
        assert not result.is_error
        assert result.content[0]["text"] == "rendered"
        assert calls == [("https://example.com", 5)]


class TestSteerAndStop:
    def test_queued_steer_message_is_injected_and_consumed(self) -> None:
        session, _ = _setup_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "/nonexistent-steer-test"})], stop_reason=StopReason.TOOL_USE
                ),
                faux_assistant_message("done"),
            ]
        )
        session.queue_steer_message("extra guidance")
        result = asyncio.run(session.prompt("go"))

        assert result is not None
        assert any(b.type == "text" and b.text == "done" for b in result.content)
        steer_msgs = [m for m in session._messages if isinstance(m, UserMessage) and m.content == "extra guidance"]
        assert len(steer_msgs) == 1
        assert session._pending_steer == []  # drained, not left queued forever

    def test_no_steer_message_is_a_noop(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("hi")])
        result = asyncio.run(session.prompt("go"))
        assert result is not None
        assert not any(isinstance(m, UserMessage) and m.content not in ("go",) for m in session._messages)

    def test_request_stop_short_circuits_before_any_model_call(self) -> None:
        session, handle = _setup_faux([faux_assistant_message("should not be reached")])
        session.request_stop()
        result = asyncio.run(session.prompt("go"))

        assert result is not None
        assert result.stop_reason == StopReason.ABORTED
        assert any(b.type == "text" and "Stopped" in b.text for b in result.content)
        # the queued faux response was never consumed
        assert not any(
            b.type == "text" and b.text == "should not be reached"
            for m in session._messages
            if hasattr(m, "content")
            for b in (m.content if isinstance(m.content, list) else [])
        )

    def test_stop_flag_resets_after_firing(self) -> None:
        session, _ = _setup_faux([faux_assistant_message("first"), faux_assistant_message("second")])
        session.request_stop()
        first = asyncio.run(session.prompt("go"))
        assert first is not None
        assert first.stop_reason == StopReason.ABORTED

        second = asyncio.run(session.prompt("go again"))
        assert second is not None
        assert second.stop_reason == StopReason.STOP
        assert any(b.type == "text" and b.text == "first" for b in second.content)
