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
from pi_coding_agent.session_manager import SessionEntry, SessionManager
from pi_coding_agent.subagent.registry import SubagentResult
from pi_memory.store import MemoryType


class _FakeSubagentStdin:
    def __init__(self) -> None:
        self.written: list[bytes] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            raise BrokenPipeError()
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self._closing


class _FakeSubagentProc:
    """Bare-minimum stand-in for asyncio.subprocess.Process — enough for
    SubagentHandle.steer()/stop() to work, without actually driving a
    process (no _drive task involved in these REPL-command tests)."""

    def __init__(self) -> None:
        self.stdin = _FakeSubagentStdin()

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return 0


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


def test_interactive_slash_model_bare_with_a_single_model_reports_it_without_a_picker(tmp_path: Path) -> None:
    """Nothing to navigate between with only one model configured (the
    default faux-provider setup here) — /model should just say so, not
    launch a picker (interactive or dialog-based) with a single option."""
    session = _make_session(tmp_path)
    printed: list[str] = []
    session._output.print = printed.append  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model")) is True
    output = "\n".join(printed)
    assert "only model available" in output
    assert session._model.id == "faux-1"


def test_interactive_slash_model_bare_launches_the_repl_picker_when_stdin_is_a_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With multiple models configured and a real (or real-looking) TTY,
    bare /model should drive the arrow-key picker (pi_tui.raw_input.
    select_from_list) rather than printing the old plain listing — and
    switch to whatever index it reports back."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    session._models.set_provider(Provider(id="acme", name="Acme AI", models=[Model(id="acme-large", provider="acme")]))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _fake_select_from_list(count: int, *, on_render: Any, initial: int) -> int:
        on_render(initial)
        return count - 1  # the last entry — acme/acme-large

    monkeypatch.setattr("pi_tui.raw_input.select_from_list", _fake_select_from_list)

    printed: list[str] = []
    session._output.print = printed.append  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model")) is True
    assert session._model.provider == "acme"
    assert session._model.id == "acme-large"
    assert session._agent_session.get_model().id == "acme-large"
    assert "switched to" in "\n".join(printed)


def test_interactive_slash_model_bare_reports_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Escape/Ctrl+C from the REPL picker (select_from_list returning
    None) must leave the active model untouched and say so, not silently
    switch to the first entry."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    original = session._model
    session._models.set_provider(Provider(id="acme", name="Acme AI", models=[Model(id="acme-large", provider="acme")]))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("pi_tui.raw_input.select_from_list", lambda count, *, on_render, initial: None)

    printed: list[str] = []
    session._output.print = printed.append  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model")) is True
    assert session._model is original
    assert "cancelled" in "\n".join(printed)


def test_interactive_slash_model_bare_uses_the_ui_context_when_not_the_noop_one(tmp_path: Path) -> None:
    """Any front-end with a real ExtensionUIContext (the Textual app) gets
    its own select() call instead of the REPL's raw-terminal picker —
    e.g. Textual's SelectDialog, which already handles arrow-key
    navigation and Enter/Escape natively."""
    from pi_ai import Model
    from pi_ai.models import Provider

    session = _make_session(tmp_path)
    session._models.set_provider(Provider(id="acme", name="Acme AI", models=[Model(id="acme-large", provider="acme")]))

    class _FakeUiContext:
        def __init__(self) -> None:
            self.asked: tuple[str, list[str]] | None = None

        async def select(self, message: str, choices: list[str]) -> str | None:
            self.asked = (message, choices)
            return choices[-1]

    fake_ui = _FakeUiContext()
    session._ui_context = fake_ui  # type: ignore[assignment]

    printed: list[str] = []
    session._output.print = printed.append  # type: ignore[method-assign]

    assert asyncio.run(session._handle_command("/model")) is True
    assert fake_ui.asked is not None
    assert "acme/acme-large" in fake_ui.asked[1]
    assert session._model.id == "acme-large"


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

    def test_remember_soul_always_asks_even_in_accept_edits_mode(self, tmp_path: Path) -> None:
        """A confirm_soul-style second self-issued tool call is not real user
        confirmation — remember(type=soul) must route to a real y/N prompt,
        unconditionally, not be silently ALLOWed just because the user
        happens to be in acceptEdits mode (which is only about file edits)."""
        session = _make_session(tmp_path)
        session._permission_mode = session._permission_mode.ACCEPT_EDITS
        with patch("builtins.input", return_value="y") as mock_input:
            allowed = asyncio.run(
                session._permission_gate("remember", {"type": "soul", "title": "x", "content": "y"})
            )
        assert allowed is True
        mock_input.assert_called_once()

    def test_remember_soul_honors_rejection(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", return_value="n"):
            allowed = asyncio.run(
                session._permission_gate("remember", {"type": "soul", "title": "x", "content": "y"})
            )
        assert allowed is False

    def test_remember_non_soul_type_unaffected(self, tmp_path: Path) -> None:
        """remember isn't in MUTATING_TOOL_NAMES, so a non-soul type keeps
        the pre-existing behavior: no gate, no prompt at all."""
        session = _make_session(tmp_path)
        with patch("builtins.input", side_effect=AssertionError("should not prompt for non-soul remember")):
            allowed = asyncio.run(
                session._permission_gate("remember", {"type": "decision", "title": "x", "content": "y"})
            )
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
    soon as the input grew past that). It now fully repaints on every
    keystroke instead, using only relative cursor moves (never DECSC/
    DECRC save/restore, which doesn't reliably survive a scroll event
    happening between the save and a later restore — that desync is what
    caused a second, different bug: the footer visibly duplicating itself
    down the screen after a turn, once a long assistant reply had scrolled
    the screen before the next prompt cycle's DECSC save even happened).

    These drive _render through a fake read_line_with_cycle and check the
    escape sequences it writes are well-formed (every absolute-column
    request stays within [1, terminal width], no negative offsets — the
    first bug's telltale symptom), and (TestFooterDoesNotDuplicate below)
    interpret those sequences against a minimal virtual terminal to check
    the actual on-screen *result* — a plain string capture can't tell a
    correctly-clearing redraw apart from stale content silently piling up
    beneath it, which is exactly what the duplication bug looked like."""

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

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any, history: Any = None) -> str:
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


class _VirtualTerminal:
    """Interprets exactly the escape sequences _prompt_input's _render/
    land_below_footer emit (\\r, \\n, CSI A/B/G/J, SGR is ignored — it
    doesn't move the cursor) against a plain list-of-lines screen buffer,
    tracking cursor row/col — enough to check the actual *rendered*
    result of a sequence of writes, not just the raw bytes. A StringIO
    capture can't distinguish "redrew correctly in place" from "cleared
    the wrong region and left stale content sitting there", which is
    exactly what the footer-duplication bug looked like; this can.
    """

    def __init__(self) -> None:
        self.lines: list[str] = [""]
        self.row = 0
        self.col = 0

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b" and data[i + 1 : i + 2] == "[":
                j = i + 2
                while j < len(data) and not data[j].isalpha() and data[j] != "~":
                    j += 1
                params, final = data[i + 2 : j], data[j] if j < len(data) else ""
                self._apply_csi(params, final)
                i = j + 1
                continue
            if ch == "\r":
                self.col = 0
                i += 1
                continue
            if ch == "\n":
                self.row += 1
                self._ensure_row(self.row)
                i += 1
                continue
            self._write_char(ch)
            i += 1

    def _ensure_row(self, row: int) -> None:
        while row >= len(self.lines):
            self.lines.append("")

    def _write_char(self, ch: str) -> None:
        self._ensure_row(self.row)
        line = self.lines[self.row]
        if self.col >= len(line):
            line = line + (" " * (self.col - len(line))) + ch
        else:
            line = line[: self.col] + ch + line[self.col + 1 :]
        self.lines[self.row] = line
        self.col += 1

    def _apply_csi(self, params: str, final: str) -> None:
        n = int(params) if params.isdigit() else 1
        if final == "A":
            self.row = max(0, self.row - n)
        elif final == "B":
            self.row += n
            self._ensure_row(self.row)
        elif final == "G":
            self.col = n - 1
        elif final == "J" and params in ("0", ""):
            self._ensure_row(self.row)
            self.lines[self.row] = self.lines[self.row][: self.col]
            self.lines = self.lines[: self.row + 1]
        # SGR ("m") and anything else: no cursor-position effect, ignore.

    def rendered_text(self) -> str:
        return "\n".join(self.lines)


class TestFooterDoesNotDuplicate:
    """Regression test for the DECSC/DECRC desync bug: after a long
    assistant reply scrolls the screen, the *next* input cycle's footer
    used to visibly duplicate itself down the screen — each keystroke's
    save-then-restore landing at a different, drifting offset from where
    the input row actually was. Runs _prompt_input's actual output
    through _VirtualTerminal and checks the state/mode line text appears
    exactly once on the resulting screen, across a sequence of renders
    interleaved with unrelated output (standing in for the scrolling a
    real assistant reply would cause)."""

    def test_repeated_renders_leave_exactly_one_footer(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        monkeypatch.setattr(interactive_mode, "_console", Console(width=40, force_terminal=True))

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any, history: Any = None) -> str:
            on_render("", 0, None)
            for text in ["h", "he", "hel", "hell", "hello", "hello world, this wraps a bit"]:
                on_render(text, len(text), None)
            return "hello world, this wraps a bit"

        monkeypatch.setattr(interactive_mode, "read_line_with_cycle", _fake_read_line_with_cycle)

        fake_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdout", fake_stdout)

        asyncio.run(session._prompt_input())

        term = _VirtualTerminal()
        term.feed(fake_stdout.getvalue())
        rendered = term.rendered_text()
        assert rendered.count("shift+tab to cycle") == 1, rendered

    def test_footer_does_not_duplicate_when_the_state_line_itself_wraps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """footer_rows used to be a fixed "rule + state + mode [+ repo]"
        count (one row per footer *line*), which silently assumed every
        one of those lines was short enough to fit within the terminal
        width. state_line (status + provider/model + session id, all on
        one line) is long enough to wrap on its own once a longer model
        id is active on a narrower terminal — undercounting footer_rows
        by however many extra rows that wrap added made every subsequent
        render's move-up land short of the input's actual first row,
        clearing only part of the stale previous render and leaving the
        rest sitting there — visibly duplicated input text, one stale
        copy per keystroke in the worst case."""
        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        session._model.provider = "nvidia"
        session._model.id = "minimaxai/minimax-m3"
        session._session_id = "9f9977bb-aaaa-bbbb-cccc-dddddddddddd"
        # Narrow enough that "* ready * nvidia/minimaxai/minimax-m3 *
        # session:9f9977bb" (state_line) wraps to 2 rows on its own.
        monkeypatch.setattr(interactive_mode, "_console", Console(width=50, force_terminal=True))

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any, history: Any = None) -> str:
            on_render("", 0, None)
            on_cycle()  # default -> accept edits
            on_render("", 0, None)
            on_cycle()  # accept edits -> plan mode (longer label too)
            on_render("", 0, None)
            text = ""
            for ch in "Execute este plano:":
                text += ch
                on_render(text, len(text), None)
            return text

        monkeypatch.setattr(interactive_mode, "read_line_with_cycle", _fake_read_line_with_cycle)

        fake_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdout", fake_stdout)

        asyncio.run(session._prompt_input())

        term = _VirtualTerminal()
        term.feed(fake_stdout.getvalue())
        rendered = term.rendered_text()
        assert rendered.count("Execute este plano") == 1, rendered

    def test_footer_does_not_duplicate_across_multiple_turns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Three simulated REPL turns, each with a chunk of "assistant
        reply" text printed directly to stdout between _prompt_input
        calls (standing in for what repl_loop actually does) — the exact
        shape that triggered the duplication bug in practice."""
        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        monkeypatch.setattr(interactive_mode, "_console", Console(width=40, force_terminal=True))

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any, history: Any = None) -> str:
            on_render("", 0, None)
            on_render("hi", 2, None)
            return "hi"

        monkeypatch.setattr(interactive_mode, "read_line_with_cycle", _fake_read_line_with_cycle)

        fake_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdout", fake_stdout)

        for _ in range(3):
            asyncio.run(session._prompt_input())
            # A long-ish "assistant reply" — several wrapped lines, like
            # what actually preceded the reported duplication.
            fake_stdout.write(("assistant reply line. " * 10 + "\r\n") * 4)

        term = _VirtualTerminal()
        term.feed(fake_stdout.getvalue())
        rendered = term.rendered_text()
        assert rendered.count("shift+tab to cycle") == 3, rendered


class TestPromptHistory:
    """Up/Down recalling prior submitted lines (pi_tui.raw_input's own
    tests cover the actual navigation logic) — here just the REPL-side
    wiring: _prompt_input passes the running session's history through,
    and repl_loop actually grows it as lines get submitted."""

    def test_prompt_input_passes_session_history_to_read_line_with_cycle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        session._prompt_history.extend(["earlier", "later"])
        monkeypatch.setattr(interactive_mode, "_console", Console(width=40, force_terminal=True))

        seen_history: list[str] | None = None

        def _fake_read_line_with_cycle(_prompt: str, *, on_render: Any, on_cycle: Any, history: Any = None) -> str:
            nonlocal seen_history
            seen_history = history
            on_render("", 0, None)
            return ""

        monkeypatch.setattr(interactive_mode, "read_line_with_cycle", _fake_read_line_with_cycle)
        monkeypatch.setattr("sys.stdout", io.StringIO())

        asyncio.run(session._prompt_input())
        assert seen_history == ["earlier", "later"]
        assert seen_history is session._prompt_history  # the live list, not a copy

    def test_repl_loop_appends_submitted_lines_to_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import pi_coding_agent.interactive_mode as interactive_mode

        session = _make_session(tmp_path)
        monkeypatch.setattr(interactive_mode, "_console", Console(width=40, force_terminal=True))

        inputs = iter(["/clear", "hello there", None])

        async def _fake_prompt_input() -> str | None:
            return next(inputs)

        monkeypatch.setattr(session, "_prompt_input", _fake_prompt_input)

        asyncio.run(session.repl_loop())
        assert session._prompt_history == ["/clear", "hello there"]


class TestSessionCommand:
    def test_show_default_reports_current_session(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/session"))
        assert any(session._session_id in p for p in printed if session._session_id)

    def test_list_with_no_query_lists_sessions(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        mgr = session._session_manager
        assert mgr is not None
        other = mgr.create_session(cwd=str(tmp_path), name="other-session")

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/session list"))
        assert any(other.id in p for p in printed)
        assert any(session._session_id in p for p in printed if session._session_id)

    def test_list_with_query_searches(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        mgr = session._session_manager
        assert mgr is not None
        target = mgr.create_session(cwd=str(tmp_path), name="refactor-auth")
        mgr.append_entry(
            target.id,
            SessionEntry(seq=0, parent_seq=None, kind="message", data={"role": "user", "content": "fix the login bug"}),
        )
        mgr.create_session(cwd=str(tmp_path), name="unrelated-topic")

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/session list refactor"))
        assert any(target.id in p for p in printed)
        assert not any("unrelated-topic" in p for p in printed)

    def test_list_query_with_no_matches_reports_nothing_found(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/session list nonexistent-topic-xyz"))
        assert any("no sessions match" in p for p in printed)

    def test_resume_switches_active_session_and_restores_transcript(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        mgr = session._session_manager
        assert mgr is not None
        original_id = session._session_id
        target = mgr.create_session(cwd=str(tmp_path), name="past-chat")
        mgr.append_entry(
            target.id,
            SessionEntry(seq=0, parent_seq=None, kind="message", data={"role": "user", "content": "hello from the past"}),
        )

        asyncio.run(session._handle_command(f"/session resume {target.id}"))
        assert session._session_id == target.id
        assert session._session_id != original_id
        assert session._message_count == 1
        assert len(session._agent_session._messages) == 1

    def test_resume_by_unambiguous_prefix(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        mgr = session._session_manager
        assert mgr is not None
        target = mgr.create_session(cwd=str(tmp_path), name="past-chat")

        asyncio.run(session._handle_command(f"/session resume {target.id[:6]}"))
        assert session._session_id == target.id

    def test_resume_unknown_id_does_not_switch(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        original_id = session._session_id
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/session resume doesnotexist"))
        assert session._session_id == original_id
        assert any("no such session" in p for p in printed)

    def test_resume_without_id_shows_usage(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        original_id = session._session_id
        asyncio.run(session._handle_command("/session resume"))
        assert session._session_id == original_id

    def test_resume_same_session_is_a_noop(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        original_id = session._session_id
        assert original_id is not None
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command(f"/session resume {original_id}"))
        assert any("already the active session" in p for p in printed)

    def test_resume_leaves_original_session_file_untouched(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        mgr = session._session_manager
        assert mgr is not None
        original_id = session._session_id
        assert original_id is not None
        original_entries_before = mgr.get_entries(original_id)

        target = mgr.create_session(cwd=str(tmp_path), name="past-chat")
        asyncio.run(session._handle_command(f"/session resume {target.id}"))

        assert mgr.get_entries(original_id) == original_entries_before

    def test_unknown_subcommand_does_not_raise(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        asyncio.run(session._handle_command("/session bogus"))


class TestSoulCommand:
    def test_show_empty_offers_onboarding_and_declining_is_a_noop(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        with patch("builtins.input", return_value="n"):
            asyncio.run(session._handle_command("/soul"))
        assert store.list_by_type(MemoryType.SOUL) == []

    def test_add_confirmed_saves_entry(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        with patch("builtins.input", return_value="y"):
            asyncio.run(session._handle_command("/soul add always ask before deleting files"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert "always ask before deleting files" in records[0].content

    def test_add_declined_saves_nothing(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        with patch("builtins.input", return_value="n"):
            asyncio.run(session._handle_command("/soul add always ask before deleting files"))
        assert store.list_by_type(MemoryType.SOUL) == []

    def test_add_without_text_shows_usage_and_does_not_prompt(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        with patch("builtins.input", side_effect=AssertionError("should not prompt without text")):
            asyncio.run(session._handle_command("/soul add"))

    def test_show_lists_existing_entries(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="Be careful", content="Always confirm destructive actions.")

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/soul show"))
        assert any("Be careful" in line for line in printed)

    def test_edit_updates_entry(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        record = store.write(type=MemoryType.SOUL, title="Old", content="Old content")

        asyncio.run(session._handle_command(f"/soul edit {record.id} New content here"))
        updated = store.list_by_type(MemoryType.SOUL)[0]
        assert updated.id == record.id
        assert "New content here" in updated.content

    def test_edit_missing_id_reports_error(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        asyncio.run(session._handle_command("/soul edit 999 some text"))  # should not raise

    def test_remove_deletes_entry(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        record = store.write(type=MemoryType.SOUL, title="Removable", content="delete me")

        asyncio.run(session._handle_command(f"/soul remove {record.id}"))
        assert store.list_by_type(MemoryType.SOUL) == []

    def test_clear_confirmed_removes_all(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="1")
        store.write(type=MemoryType.SOUL, title="b", content="2")

        with patch("builtins.input", return_value="y"):
            asyncio.run(session._handle_command("/soul clear"))
        assert store.list_by_type(MemoryType.SOUL) == []

    def test_clear_declined_keeps_entries(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="1")

        with patch("builtins.input", return_value="n"):
            asyncio.run(session._handle_command("/soul clear"))
        assert len(store.list_by_type(MemoryType.SOUL)) == 1

    def test_unknown_subcommand_does_not_raise(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        asyncio.run(session._handle_command("/soul bogus"))


class TestSoulOverlapOnAdd:
    def test_no_overlap_saves_normally_with_single_confirm(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        with patch("builtins.input", return_value="y"):
            asyncio.run(session._handle_command("/soul add always separate requirements from assumptions"))
        assert len(store.list_by_type(MemoryType.SOUL)) == 1

    def test_overlap_keep_both_creates_second_record(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")

        with patch("builtins.input", side_effect=["1", "y"]):
            asyncio.run(session._handle_command("/soul add I prefer concise answers please"))
        assert len(store.list_by_type(MemoryType.SOUL)) == 2

    def test_overlap_replace_updates_existing_without_new_record(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        original = store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")

        with patch("builtins.input", side_effect=["2"]):
            asyncio.run(session._handle_command("/soul add I prefer very concise answers"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == original.id
        assert records[0].content == "I prefer very concise answers"

    def test_overlap_merge_uses_accepted_suggestion(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        original = store.write(type=MemoryType.SOUL, title="Be concise", content="Answer concisely.")

        with patch("builtins.input", side_effect=["3", "y"]):
            asyncio.run(session._handle_command("/soul add Answer very concisely please"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == original.id
        assert "Answer concisely." in records[0].content
        assert "Answer very concisely please" in records[0].content

    def test_overlap_merge_declined_suggestion_prompts_for_typed_text(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        original = store.write(type=MemoryType.SOUL, title="Be concise", content="Answer concisely.")

        with patch("builtins.input", side_effect=["3", "n", "Custom merged principle text"]):
            asyncio.run(session._handle_command("/soul add Answer very concisely please"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == original.id
        assert records[0].content == "Custom merged principle text"

    def test_overlap_cancel_saves_nothing(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")

        with patch("builtins.input", side_effect=["4"]):
            asyncio.run(session._handle_command("/soul add I prefer concise answers please"))
        assert len(store.list_by_type(MemoryType.SOUL)) == 1

    def test_overlap_check_applies_to_onboarding_too(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        existing = store.write(type=MemoryType.SOUL, title="Be concise", content="I prefer concise answers.")

        # onboarding: 1st question answered with overlapping text -> "salvar?"
        # confirmed -> overlap prompt appears -> choice "4" cancels that one
        # question -> the remaining 3 onboarding questions are skipped (blank).
        with patch(
            "builtins.input",
            side_effect=["I prefer concise answers please", "y", "4", "", "", ""],
        ):
            asyncio.run(session._soul_onboarding(store))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == existing.id


class TestSoulAudit:
    def test_fewer_than_two_principles_reports_nothing_to_audit(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="Answer concisely.")

        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            asyncio.run(session._handle_command("/soul audit"))

    def test_no_overlap_among_existing_principles_reports_none_found(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="Answer concisely.")
        store.write(type=MemoryType.SOUL, title="b", content="Always separate requirements from assumptions.")

        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            asyncio.run(session._handle_command("/soul audit"))

    def test_overlap_found_keep_only_one_deletes_the_other(self, tmp_path: Path) -> None:
        """list_by_type (and therefore _soul_audit's outer loop) iterates
        newest-first, so the "record" being audited each time is the most
        recently written of an overlapping pair — "keep only #record" keeps
        that newer one."""
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="I prefer concise answers.")
        newer = store.write(type=MemoryType.SOUL, title="b", content="I prefer concise answers always.")

        with patch("builtins.input", side_effect=["2"]):
            asyncio.run(session._handle_command("/soul audit"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == newer.id

    def test_overlap_found_merge_consolidates_into_one_record(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="Answer concisely.")
        newer = store.write(type=MemoryType.SOUL, title="b", content="Answer very concisely please.")

        with patch("builtins.input", side_effect=["3", "y"]):
            asyncio.run(session._handle_command("/soul audit"))
        records = store.list_by_type(MemoryType.SOUL)
        assert len(records) == 1
        assert records[0].id == newer.id
        assert "Answer concisely." in records[0].content
        assert "Answer very concisely please." in records[0].content

    def test_overlap_found_default_choice_keeps_both(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        store = session._agent_session._memory_store
        assert store is not None
        store.write(type=MemoryType.SOUL, title="a", content="I prefer concise answers.")
        store.write(type=MemoryType.SOUL, title="b", content="I prefer concise answers always.")

        with patch("builtins.input", side_effect=[""]):
            asyncio.run(session._handle_command("/soul audit"))
        assert len(store.list_by_type(MemoryType.SOUL)) == 2


class TestAgentsCommand:
    def test_list_with_no_subagents_reports_none_ran(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/agents"))
        assert any("no subagents have run yet" in p for p in printed)

    def test_list_shows_running_and_finished(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        running = registry.register("scout", "explore the repo", _FakeSubagentProc())
        finished = registry.register("writer", "write docs", _FakeSubagentProc())
        finished.mark_done(SubagentResult(output="done!", exit_code=0, agent_name="writer", status="done"))

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/agents list"))
        assert any(running.id in p for p in printed)
        assert any(finished.id in p for p in printed)
        assert any("running" in p for p in printed)

    def test_stop_unknown_id_reports_error(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/agents stop doesnotexist"))
        assert any("no such subagent" in p for p in printed)

    def test_stop_without_id_shows_usage(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/agents stop"))
        assert any("usage" in p for p in printed)

    def test_stop_already_finished_agent_is_a_noop(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        handle = registry.register("scout", "task", _FakeSubagentProc())
        handle.mark_done(SubagentResult(output="x", exit_code=0, agent_name="scout", status="done"))

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command(f"/agents stop {handle.id}"))
        assert any("already finished" in p for p in printed)

    def test_stop_running_agent_that_acknowledges_quickly(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        handle = registry.register("scout", "task", _FakeSubagentProc())

        async def _ack_soon() -> None:
            await asyncio.sleep(0.01)
            handle.mark_done(SubagentResult(output="stopped early", exit_code=0, agent_name="scout", status="killed"))

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]

        async def _run() -> None:
            await asyncio.gather(session._handle_command(f"/agents stop {handle.id}"), _ack_soon())

        asyncio.run(_run())
        assert handle.status == "killed"
        assert any("stopped" in p for p in printed)
        # the stop command itself (not just the ack) was actually sent
        assert handle.proc.stdin.written  # type: ignore[attr-defined]

    def test_steer_unknown_id_reports_error(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command("/agents steer doesnotexist do the thing"))
        assert any("no such subagent" in p for p in printed)

    def test_steer_without_text_shows_usage(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        handle = registry.register("scout", "task", _FakeSubagentProc())

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command(f"/agents steer {handle.id}"))
        assert any("usage" in p for p in printed)

    def test_steer_running_agent_writes_command(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        handle = registry.register("scout", "task", _FakeSubagentProc())

        asyncio.run(session._handle_command(f"/agents steer {handle.id} also check the tests"))
        written = handle.proc.stdin.written  # type: ignore[attr-defined]
        assert written
        assert b"also check the tests" in written[-1]

    def test_steer_finished_agent_is_a_noop(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        registry = session._agent_session.get_subagent_registry()
        assert registry is not None
        handle = registry.register("scout", "task", _FakeSubagentProc())
        handle.mark_done(SubagentResult(output="x", exit_code=0, agent_name="scout", status="done"))

        printed: list[str] = []
        session._output.print = lambda markup="", **_: printed.append(markup)  # type: ignore[method-assign]
        asyncio.run(session._handle_command(f"/agents steer {handle.id} too late"))
        assert any("no longer running" in p for p in printed)

    def test_unknown_subcommand_does_not_raise(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        asyncio.run(session._handle_command("/agents bogus"))
