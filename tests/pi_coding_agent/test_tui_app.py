"""Tests for pi_coding_agent.tui_app.PiApp — the Phase T0 Textual front-end.

Uses Textual's own App.run_test() harness (a real, driven app instance,
not a mock) against InteractiveSession wired to the faux provider, so
these exercise the actual widget tree/event flow without a network call.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import OptionList, Static

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.dialogs import ConfirmDialog
from pi_coding_agent.interactive_mode import InteractiveSession
from pi_coding_agent.output_sink import ConsoleOutputSink
from pi_coding_agent.permission_mode import PermissionMode
from pi_coding_agent.prompt_editor import PromptTextArea
from pi_coding_agent.session_manager import SessionManager
from pi_coding_agent.tui_app import PiApp, _TranscriptSink


def _make_session(tmp_path: Path, responses: list[Any] | None = None) -> InteractiveSession:
    handle = faux_provider()
    handle.set_responses(responses or [faux_assistant_message("hello from tui")])
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


def _transcript_text(app: PiApp) -> str:
    """Plain text of everything mounted in the transcript so far.

    child.content can be a plain string or a Rich renderable (e.g. the
    Markdown blocks flushed text streaming produces) — render everything
    through a recording Console instead of str()-ing it, which would just
    show a renderable's repr rather than what it actually displays.
    """
    transcript = app.query_one("#transcript", VerticalScroll)
    console = Console(record=True, width=100, file=io.StringIO())
    for child in transcript.children:
        if isinstance(child, Static):
            console.print(child.content)
    return console.export_text()


async def _settle(app: PiApp, pilot: Pilot[None]) -> None:
    """Wait for a submission to fully process.

    _handle_submission runs as a @work(exclusive=True) worker (needed so
    it can push_screen_wait a modal) — a plain pilot.pause() only pumps
    the message queue once and isn't guaranteed to wait long enough for
    that background task to actually finish, which showed up as an
    intermittent race when this suite ran alongside other tests. Waiting
    on the app's WorkerManager directly is the robust way to know the
    submission has actually completed.
    """
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def _fake_scroll() -> tuple[VerticalScroll, list[Static]]:
    """A VerticalScroll with mount()/scroll_end() faked out to just record
    calls — _TranscriptSink's accumulation logic doesn't need a running
    app, only something that looks like these two methods."""
    scroll = VerticalScroll()
    mounted: list[Static] = []

    def _fake_mount(*widgets: object, **_k: object) -> None:
        mounted.extend(w for w in widgets if isinstance(w, Static))

    def _fake_scroll_end(*_a: object, **_k: object) -> None:
        pass

    scroll.mount = _fake_mount  # type: ignore[assignment]
    scroll.scroll_end = _fake_scroll_end  # type: ignore[method-assign]
    return scroll, mounted


class TestTranscriptSink:
    def test_streaming_end_empty_accumulates_into_one_widget(self) -> None:
        scroll, mounted = _fake_scroll()
        sink = _TranscriptSink(scroll)

        sink.print("a", end="")
        sink.print("b", end="")
        sink.print("c", end="")
        assert len(mounted) == 1
        assert mounted[0].content == "abc"

    def test_end_newline_finalizes_and_starts_a_new_widget(self) -> None:
        scroll, mounted = _fake_scroll()
        sink = _TranscriptSink(scroll)

        sink.print("first line")
        sink.print("second line")
        assert len(mounted) == 2

    def test_print_renderable_finalizes_current_line_too(self) -> None:
        from rich.text import Text

        scroll, mounted = _fake_scroll()
        sink = _TranscriptSink(scroll)

        sink.print("streaming", end="")
        sink.print_renderable(Text("a diff or markdown block"))
        sink.print("new streaming", end="")
        assert len(mounted) == 3
        assert sink._current_text == "new streaming"


class TestPiApp:
    @pytest.mark.asyncio
    async def test_on_mount_replaces_output_with_transcript_sink(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test():
            assert isinstance(session._output, _TranscriptSink)

    @pytest.mark.asyncio
    async def test_submitting_a_message_runs_a_turn_and_renders_the_reply(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, [faux_assistant_message("the answer is 42")])
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "what is the answer?"
            await pilot.press("enter")
            await _settle(app, pilot)

            assert input_widget.text == ""
            text = _transcript_text(app)
            assert "what is the answer?" in text
            assert "42" in text

    @pytest.mark.asyncio
    async def test_empty_submit_does_nothing(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = ""
            await pilot.press("enter")
            await _settle(app, pilot)
            transcript = app.query_one("#transcript", VerticalScroll)
            assert len(transcript.children) == 0

    @pytest.mark.asyncio
    async def test_slash_command_is_dispatched_and_does_not_run_a_turn(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/model"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert session._agent_session._messages == []

    @pytest.mark.asyncio
    async def test_slash_exit_closes_the_app(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/exit"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert app._exit is True

    @pytest.mark.asyncio
    async def test_shift_tab_cycles_permission_mode_and_updates_footer(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            assert session._permission_mode == PermissionMode.DEFAULT
            await pilot.press("shift+tab")
            await pilot.pause()
            # mypy narrows _permission_mode to Literal[DEFAULT] from the assert
            # above and doesn't invalidate it across the awaits in between
            # (it has no reason to suspect pilot.press mutates `session`) —
            # a real state change at runtime, just a static-analysis gap.
            assert session._permission_mode == PermissionMode.ACCEPT_EDITS  # type: ignore[comparison-overlap]

            footer = app.query_one("#status-footer", Static)
            assert "accept edits on" in str(footer.content)

    @pytest.mark.asyncio
    async def test_extension_command_output_appears_in_transcript(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "greet.py").write_text(
            'def _greet(args_text, ctx):\n    return f"hello {args_text}"\n\n'
            'def extension(pi):\n    pi.register_command("greet", _greet)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/greet Bob"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert "hello Bob" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_extension_shortcut_is_dispatched_and_does_not_leak_into_the_editor(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "hotkey.py").write_text(
            'def _on_hotkey(ctx):\n    return "hotkey fired"\n\n'
            'def extension(pi):\n    pi.register_shortcut("ctrl+g", _on_hotkey)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.focus()
            await pilot.press("ctrl+g")
            await _settle(app, pilot)
            assert "hotkey fired" in _transcript_text(app)
            # the shortcut key must not also land in the focused editor
            assert input_widget.text == ""

    @pytest.mark.asyncio
    async def test_unregistered_key_is_not_dispatched_as_a_shortcut(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.focus()
            await pilot.press("g")
            await _settle(app, pilot)
            # with no extension registered, a plain key just types normally
            assert input_widget.text == "g"

    @pytest.mark.asyncio
    async def test_typing_a_slash_command_prefix_shows_matching_suggestions(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/mo"
            await pilot.pause()
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is True
            assert str(suggestions.get_option_at_index(0).prompt) == "/model"

    @pytest.mark.asyncio
    async def test_suggestions_hide_once_the_text_no_longer_matches(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/mo"
            await pilot.pause()
            input_widget.text = "hello"
            await pilot.pause()
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is False

    @pytest.mark.asyncio
    async def test_tab_accepts_the_highlighted_suggestion(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.focus()
            input_widget.text = "/mo"
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert input_widget.text == "/model "
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is False

    @pytest.mark.asyncio
    async def test_escape_dismisses_the_suggestion_popup_without_changing_text(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.focus()
            input_widget.text = "/mo"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert input_widget.text == "/mo"
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is False

    @pytest.mark.asyncio
    async def test_extension_commands_are_included_in_suggestions(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "greet.py").write_text(
            'def _greet(args_text, ctx):\n    return "hi"\n\n'
            'def extension(pi):\n    pi.register_command("greet", _greet)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/gr"
            await pilot.pause()
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is True
            assert str(suggestions.get_option_at_index(0).prompt) == "/greet"

    @pytest.mark.asyncio
    async def test_extension_theme_is_registered_and_selectable(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "theme.py").write_text(
            'def extension(pi):\n    pi.register_theme("midnight", primary="#1e1e2e", dark=True)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test():
            assert "midnight" in app.available_themes
            app.theme = "midnight"
            assert app.theme == "midnight"

    @pytest.mark.asyncio
    async def test_default_mode_tool_call_shows_confirm_dialog_and_allows_on_yes(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        target = tmp_path / "written.txt"
        session = _make_session(
            tmp_path,
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": str(target), "content": "hi"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
        )
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "write the file"
            await pilot.press("enter")
            await pilot.pause()

            # The dialog should be up now, blocking the tool call.
            assert isinstance(app.screen, ConfirmDialog)
            assert not target.exists()

            await pilot.press("y")
            await _settle(app, pilot)

            assert target.read_text(encoding="utf-8") == "hi"

    @pytest.mark.asyncio
    async def test_default_mode_tool_call_blocked_on_no(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        target = tmp_path / "written.txt"
        session = _make_session(
            tmp_path,
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": str(target), "content": "hi"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
        )
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "write the file"
            await pilot.press("enter")
            await pilot.pause()

            await pilot.press("n")
            await _settle(app, pilot)

            assert not target.exists()
            tool_results = [m for m in session._agent_session._messages if getattr(m, "role", "") == "toolResult"]
            assert len(tool_results) == 1
            assert tool_results[0].is_error is True


def test_console_output_sink_is_the_default_for_a_plain_session(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert isinstance(session._output, ConsoleOutputSink)
