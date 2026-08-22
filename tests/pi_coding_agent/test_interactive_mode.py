"""Tests for interactive mode."""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from rich.console import Console

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


def test_interactive_slash_model_lists_every_provider_and_its_models(tmp_path: Path) -> None:
    """/model used to only print the currently active provider/model — it
    should also list every other provider (and each of its models) that
    MutableModels knows about, e.g. ones an extension registered via
    pi.register_provider alongside the one actually in use."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    extra_provider: Provider[str] = Provider(
        id="acme",
        name="Acme AI",
        models=[Model(id="acme-large", provider="acme"), Model(id="acme-small", provider="acme")],
    )
    session._models.set_provider(extra_provider)

    printed: list[str] = []

    def _record_print(markup: str = "", *, end: str = "\n") -> None:
        printed.append(markup)

    session._output.print = _record_print  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model")) is True
    output = "\n".join(printed)
    assert "Acme AI" in output
    assert "acme-large" in output
    assert "acme-small" in output


def test_interactive_slash_model_with_id_switches_the_active_model(tmp_path: Path) -> None:
    """/model used to be read-only — /model <id> now actually switches,
    both the REPL's own self._model and the underlying AgentSession's
    (which is what actually gets streamed against)."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    extra_provider: Provider[str] = Provider(
        id="acme", name="Acme AI", models=[Model(id="acme-large", provider="acme")]
    )
    session._models.set_provider(extra_provider)

    assert asyncio.run(session._handle_command("/model acme-large")) is True
    assert session._model.id == "acme-large"
    assert session._agent_session.get_model().id == "acme-large"


def test_interactive_slash_model_with_provider_slash_id_switches(tmp_path: Path) -> None:
    """The provider/id form disambiguates two providers that happen to
    share a model id."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    session._models.set_provider(Provider(id="acme", name="Acme AI", models=[Model(id="shared-id", provider="acme")]))
    session._models.set_provider(
        Provider(id="other", name="Other AI", models=[Model(id="shared-id", provider="other")])
    )

    assert asyncio.run(session._handle_command("/model other/shared-id")) is True
    assert session._model.provider == "other"


def test_interactive_slash_model_with_unknown_id_reports_error_and_does_not_switch(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    original = session._model
    printed: list[str] = []

    def _record_print(markup: str = "", *, end: str = "\n") -> None:
        printed.append(markup)

    session._output.print = _record_print  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model does-not-exist")) is True
    assert session._model is original
    assert "no such model" in "\n".join(printed)


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


_RENDERING_EXTENSION = """
def _shout(text, ctx):
    return text.upper()

def extension(pi):
    pi.register_markdown_transformer(_shout)
"""

_ENTRY_RENDERING_EXTENSION = """
def _render_write(phase, event, ctx):
    if phase == "start":
        return "[custom] starting write"
    return "[custom] write done"

def extension(pi):
    pi.register_entry_renderer("write", _render_write)
"""


class TestRenderingHooksIntegration:
    def test_markdown_transformer_runs_before_rendering(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "rendering.py").write_text(_RENDERING_EXTENSION, encoding="utf-8")

        session = _make_session(tmp_path)
        printed: list[object] = []

        def _record_renderable(renderable: object) -> None:
            printed.append(renderable)

        session._output.print_renderable = _record_renderable  # type: ignore[method-assign]
        session._text_block_buf = "hello world"
        session._flush_text_block()
        # Markdown's __str__ is its repr, not its source — check .markup
        # (the actual text passed to Markdown()) instead, matching how
        # test_tui_app.py's _transcript_text() helper avoids the same trap.
        assert any(getattr(r, "markup", None) == "HELLO WORLD" for r in printed)

    def test_entry_renderer_replaces_default_tool_call_rendering(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "rendering.py").write_text(_ENTRY_RENDERING_EXTENSION, encoding="utf-8")

        session = _make_session(tmp_path)
        printed: list[str] = []

        def _record_print(markup: str = "", *, end: str = "\n") -> None:
            printed.append(markup)

        session._output.print = _record_print  # type: ignore[method-assign]

        class _Event:
            type = "tool_call_start"
            name = "write"
            args: ClassVar[dict[str, object]] = {}

        session._handle_event(_Event())
        assert printed == ["[custom] starting write"]


_Render = tuple[str, int, tuple[int, int] | None]
_RenderInput = str | tuple[str, int] | _Render


class TestPromptInputFullRedraw:
    """_prompt_input's live footer used to corrupt the screen once typed
    input wrapped past one terminal row (fixed row-offset assumptions
    baked in when the footer was first drawn stopped matching reality as
    soon as the input grew past that). It now fully repaints, anchored at
    a saved cursor position, on every keystroke instead — these drive that
    through a fake read_line_with_cycle and check the escape sequences it
    writes are well-formed (in particular, every absolute-column request
    stays within [1, terminal width], the bug's telltale symptom when it
    regresses)."""

    @staticmethod
    def _run_with_fake_input(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        width: int,
        renders: list[_RenderInput],
    ) -> tuple[str, str]:
        # A bare str means "cursor at the end, no selection"; a 2-tuple
        # adds an explicit cursor; a 3-tuple adds a selection too — most
        # tests don't care about mid-buffer cursor/selection, only
        # test_cursor_* and test_selection_* below do.
        def _normalize(r: _RenderInput) -> _Render:
            if isinstance(r, str):
                return r, len(r), None
            if len(r) == 2:
                return r[0], r[1], None
            return r

        triples = [_normalize(r) for r in renders]
        final_text = triples[-1][0] if triples else ""

        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        monkeypatch.setattr(interactive_mode, "_console", Console(width=width, force_terminal=True))

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any) -> str:
            on_render("", 0, None)
            for text, cursor, selection in triples:
                on_render(text, cursor, selection)
            on_cycle()
            on_render(final_text, len(final_text), None)
            return final_text

        monkeypatch.setattr(interactive_mode, "read_line_with_cycle", _fake_read_line_with_cycle)

        fake_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdout", fake_stdout)

        result = asyncio.run(session._prompt_input())
        assert result is not None
        return result, fake_stdout.getvalue()

    def test_short_text_no_wrap(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        result, output = self._run_with_fake_input(monkeypatch, tmp_path, width=40, renders=["h", "hi"])
        assert result == "hi"
        self._assert_columns_in_range(output, width=40)

    def test_long_text_wraps_across_multiple_rows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        long_text = "x" * 90  # several times a 20-column width, forces wrapping
        result, output = self._run_with_fake_input(
            monkeypatch, tmp_path, width=20, renders=["x", "xx", long_text, long_text[:-1]]
        )
        assert result == long_text[:-1]
        self._assert_columns_in_range(output, width=20)

    def test_text_exactly_a_multiple_of_width(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        exact_text = "y" * 40  # (prompt "> " + 38 chars) lands short of a clean check either way
        result, output = self._run_with_fake_input(monkeypatch, tmp_path, width=20, renders=[exact_text])
        assert result == exact_text
        self._assert_columns_in_range(output, width=20)

    def test_cursor_in_the_middle_of_wrapped_text(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Navigating back into already-wrapped text (Left/Home) and
        editing there used to be impossible (cursor was always pinned to
        the end) — this exercises _render with the cursor away from the
        end of a multi-row buffer, the case that needs the general (row,
        col) positioning rather than "always the last row"."""
        long_text = "x" * 90
        result, output = self._run_with_fake_input(
            monkeypatch,
            tmp_path,
            width=20,
            renders=[
                (long_text, len(long_text)),
                (long_text, 5),
                (long_text[:5] + "Y" + long_text[5:], 6),
            ],
        )
        assert result == long_text[:5] + "Y" + long_text[5:]
        self._assert_columns_in_range(output, width=20)

    def test_cursor_moved_back_to_column_one_of_a_row(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Cursor position exactly at a wrap boundary (start of a wrapped
        row) is the same edge case that needed care for the end-of-text
        column math — check it doesn't request column 0 or a negative row
        offset when exercised for an arbitrary (non-end) cursor too."""
        text = "x" * 25  # width 20: wraps once, position 20 is column 1 of row 2
        result, output = self._run_with_fake_input(
            monkeypatch, tmp_path, width=20, renders=[(text, len(text)), (text, 20), (text, 25)]
        )
        assert result == text
        self._assert_columns_in_range(output, width=20)

    def test_selection_is_wrapped_in_reverse_video(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        result, output = self._run_with_fake_input(
            monkeypatch, tmp_path, width=40, renders=[("hello", 5, None), ("hello", 2, (2, 5))]
        )
        assert result == "hello"
        assert "\x1b[7mllo\x1b[27m" in output
        self._assert_columns_in_range(output, width=40)

    def test_no_selection_writes_no_reverse_video_codes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _, output = self._run_with_fake_input(monkeypatch, tmp_path, width=40, renders=["hi"])
        assert "\x1b[7m" not in output
        assert "\x1b[27m" not in output

    @staticmethod
    def _assert_columns_in_range(output: str, *, width: int) -> None:
        # No negative-parameter escape sequences at all (e.g. "\x1b[-1B")
        # — \d+ below wouldn't match those, silently hiding a row/column
        # miscalculation instead of catching it.
        assert "\x1b[-" not in output, "negative row/column offset written"
        columns = [int(n) for n in re.findall(r"\x1b\[(\d+)G", output)]
        assert columns, "expected at least one absolute-column cursor move"
        assert all(1 <= c <= width for c in columns), columns
