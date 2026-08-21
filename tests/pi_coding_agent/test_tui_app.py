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
from textual.widgets import Input, Static

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.interactive_mode import InteractiveSession
from pi_coding_agent.output_sink import ConsoleOutputSink
from pi_coding_agent.permission_mode import PermissionMode
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
            input_widget = app.query_one("#prompt-input", Input)
            input_widget.value = "what is the answer?"
            await pilot.press("enter")
            await pilot.pause()

            assert input_widget.value == ""
            text = _transcript_text(app)
            assert "what is the answer?" in text
            assert "42" in text

    @pytest.mark.asyncio
    async def test_empty_submit_does_nothing(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", Input)
            input_widget.value = ""
            await pilot.press("enter")
            await pilot.pause()
            transcript = app.query_one("#transcript", VerticalScroll)
            assert len(transcript.children) == 0

    @pytest.mark.asyncio
    async def test_slash_command_is_dispatched_and_does_not_run_a_turn(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", Input)
            input_widget.value = "/model"
            await pilot.press("enter")
            await pilot.pause()
            assert session._agent_session._messages == []

    @pytest.mark.asyncio
    async def test_slash_exit_closes_the_app(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", Input)
            input_widget.value = "/exit"
            await pilot.press("enter")
            await pilot.pause()
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
            input_widget = app.query_one("#prompt-input", Input)
            input_widget.value = "/greet Bob"
            await pilot.press("enter")
            await pilot.pause()
            assert "hello Bob" in _transcript_text(app)


def test_console_output_sink_is_the_default_for_a_plain_session(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    assert isinstance(session._output, ConsoleOutputSink)
