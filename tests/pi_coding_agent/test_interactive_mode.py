"""Tests for interactive mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.interactive_mode import InteractiveSession
from pi_coding_agent.session_manager import SessionManager


def _make_session(tmp_path: Path) -> InteractiveSession:
    handle = faux_provider()
    handle.set_responses([faux_assistant_message("hello from interactive")])
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    session_mgr = SessionManager(tmp_path / "sessions")
    info = session_mgr.create_session(cwd=str(tmp_path), name="test")
    return InteractiveSession(
        models=models,
        model=model,
        cwd=str(tmp_path),
        config_dir=tmp_path / ".pi",
        session_manager=session_mgr,
        session_id=info.id,
    )


def test_interactive_run_turn(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    result = asyncio.run(session.run_turn("hello"))
    assert result is not None
    assert any(block.type == "text" and "interactive" in block.text for block in result.content)


def test_on_status_change_fires_during_a_turn(tmp_path: Path) -> None:
    """Phase T6: PiApp refreshes its footer live by hooking this callback
    — it must fire at least once per turn (on "done") so the footer isn't
    stuck showing whatever status was true before the turn started."""
    session = _make_session(tmp_path)
    calls = 0

    def _on_change() -> None:
        nonlocal calls
        calls += 1

    session._on_status_change = _on_change
    asyncio.run(session.run_turn("hello"))
    assert calls > 0


def test_interactive_slash_help(tmp_path: Path, capsys: object) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/help")) is True


def test_interactive_slash_exit(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/exit")) is False


def test_interactive_slash_model(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/model")) is True


def test_interactive_slash_clear(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/clear")) is True


def test_interactive_slash_tools(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/tools")) is True


def test_interactive_slash_session(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/session")) is True


def test_interactive_unknown_command(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert asyncio.run(session._handle_command("/unknown")) is True


class TestPermissionModeFooter:
    def test_starts_in_default_mode(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert session._permission_mode.value == "default"
        assert "default mode" in session._mode_line()

    def test_cycle_advances_through_modes_and_wraps(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        session._cycle_permission_mode()
        assert "accept edits on" in session._mode_line()
        session._cycle_permission_mode()
        assert "plan mode on" in session._mode_line()
        session._cycle_permission_mode()
        assert "default mode" in session._mode_line()

    def test_mode_line_always_mentions_shift_tab_hint(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert "shift+tab to cycle" in session._mode_line()


class TestPermissionGate:
    def test_default_mode_asks_and_honors_yes(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", return_value="y"):
            allowed = asyncio.run(session._permission_gate("write", {"path": "x", "content": "y"}))
        assert allowed is True

    def test_default_mode_asks_and_honors_no(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", return_value="n"):
            allowed = asyncio.run(session._permission_gate("write", {"path": "x", "content": "y"}))
        assert allowed is False

    def test_default_mode_treats_empty_answer_as_no(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", return_value=""):
            allowed = asyncio.run(session._permission_gate("bash", {"command": "ls"}))
        assert allowed is False

    def test_plan_mode_denies_without_prompting(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        session._permission_mode = session._permission_mode.PLAN
        with patch("builtins.input", side_effect=AssertionError("should not prompt in plan mode")):
            allowed = asyncio.run(session._permission_gate("write", {"path": "x", "content": "y"}))
        assert allowed is False

    def test_accept_edits_allows_write_without_prompting(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        session._permission_mode = session._permission_mode.ACCEPT_EDITS
        with patch("builtins.input", side_effect=AssertionError("should not prompt for accepted edits")):
            allowed = asyncio.run(session._permission_gate("edit", {"path": "x"}))
        assert allowed is True

    def test_read_only_tools_never_prompt(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", side_effect=AssertionError("should not prompt for read")):
            allowed = asyncio.run(session._permission_gate("read", {"path": "x"}))
        assert allowed is True


class TestRepoLine:
    def test_repo_line_none_outside_git_repo(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("pi_coding_agent.interactive_mode.get_git_repo_line", return_value=None):
            assert session._repo_line() is None

    def test_repo_line_renders_repo_and_branch(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("pi_coding_agent.interactive_mode.get_git_repo_line", return_value="(python-pi:main)"):
            line = session._repo_line()
        assert line is not None
        assert "(python-pi:main)" in line


_GREET_EXTENSION = """
def _greet(args_text, ctx):
    return f"hello {args_text}"

def extension(pi):
    pi.register_command("greet", _greet, description="Greets someone")
"""


class TestExtensionsIntegration:
    def test_extensions_command_reports_no_extensions(self, tmp_path: Path, capsys: object) -> None:
        session = _make_session(tmp_path)
        assert asyncio.run(session._handle_command("/extensions")) is True

    def test_loaded_extension_is_reported_by_slash_extensions(self, tmp_path: Path, capsys: object) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "greet.py").write_text(_GREET_EXTENSION, encoding="utf-8")

        session = _make_session(tmp_path)
        assert asyncio.run(session._handle_command("/extensions")) is True

    def test_extension_registered_command_is_dispatched(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "greet.py").write_text(_GREET_EXTENSION, encoding="utf-8")

        session = _make_session(tmp_path)
        assert asyncio.run(session._handle_command("/greet Bob")) is True

    def test_unregistered_command_still_falls_through_to_unknown(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert asyncio.run(session._handle_command("/not_a_real_command")) is True
