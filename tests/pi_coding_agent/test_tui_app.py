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
from pi_coding_agent.output_sink import ConsoleOutputSink, FooterAwareOutputSink
from pi_coding_agent.permission_mode import PermissionMode
from pi_coding_agent.prompt_editor import PromptTextArea
from pi_coding_agent.session_manager import SessionManager
from pi_coding_agent.tui_app import AppFooter, PiApp, TransientFooterContent, _TranscriptSink


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


def _make_multi_model_session(tmp_path: Path, model_ids: list[str]) -> InteractiveSession:
    """Like _make_session, but with more than one model registered on the
    same provider — the bare `/model` picker only shows its list at all
    once there's an actual choice to make (a single-model provider gets
    the short "only model available" note instead — see
    PiApp._select_model_transient)."""
    handle = faux_provider(models=[{"id": model_id, "name": model_id} for model_id in model_ids])
    handle.set_responses([faux_assistant_message("ok") for _ in range(10)])
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


def _footer_text(app: PiApp) -> str:
    """Plain text of the unified footer's *stored* content — it's not
    always a bare string (AppFooter.set_extension wraps it in a
    rich.console.Group once extension content is present), so render it
    through a recording Console the same way _transcript_text does.

    This checks what AppFooter was TOLD to show, not what actually ends
    up painted on screen — see _footer_screen_text, which checks the
    latter and is the one that would have caught the real regression
    this class guards against (below)."""
    footer = app.query_one("#app-footer", AppFooter)
    console = Console(record=True, width=100, file=io.StringIO())
    console.print(footer.content)
    return console.export_text()


def _footer_screen_text(app: PiApp) -> str:
    """Plain text of what the footer widget actually paints, row by row,
    via the same Widget.render_line Textual's own compositor calls to
    put pixels on screen — as opposed to _footer_text, which only checks
    the renderable AppFooter was *given* (``.content``).

    That distinction mattered for real: an earlier version of AppFooter
    routed set_status/set_extension through one shared internal "compute
    the combined renderable, then self.update(...)" helper method.
    ``.content``/``.visual`` still reported the right text with that
    version — but ``render_line`` came back blank on every row, a real
    on-screen regression _footer_text alone could not see. Asserting
    against this function anywhere multiline/extension footer content is
    checked is what actually guards against that regressing again."""
    footer = app.query_one("#app-footer", AppFooter)
    lines = [
        "".join(segment.text for segment in footer.render_line(y)) for y in range(footer.outer_size.height)
    ]
    return "\n".join(lines)


def _footer_compositor_lines(app: PiApp) -> list[str]:
    """The footer's own on-screen rows exactly as the app's compositor
    paints them — border included. ``Widget.render_line`` (used by
    _footer_screen_text above) only ever returns the widget's *content*
    area; border decoration is layered on afterwards by the compositor
    (StylesCache.render_widget), so render_line can never see it — this
    reads the same full-screen strips the real terminal frame is built
    from (app.screen._compositor.render_strips()) and slices out just
    the rows the footer's own region covers."""
    footer = app.query_one("#app-footer", AppFooter)
    region = footer.region
    strips = app.screen._compositor.render_strips()
    lines: list[str] = []
    for y in range(region.y, region.y + region.height):
        if 0 <= y < len(strips):
            lines.append("".join(segment.text for segment in strips[y]))
        else:
            lines.append("")
    return lines


def _footer_has_both_borders(app: PiApp) -> bool:
    """True if the footer's first and last on-screen rows (border
    included — see _footer_compositor_lines) both contain a real
    box-drawing border glyph."""
    lines = _footer_compositor_lines(app)
    if len(lines) < 2:
        return False
    border_chars = set("─│┌┐└┘")
    top, bottom = lines[0], lines[-1]
    return any(c in border_chars for c in top) and any(c in border_chars for c in bottom)


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
    async def test_prompt_input_grows_with_multiline_content(self, tmp_path: Path) -> None:
        """The input box used to have a fixed height of 3 rows with a
        border eating 2 of them, leaving exactly 1 visible content row —
        typing (or wrapping) past that pushed the cursor below the visible
        box entirely instead of the box growing to show it."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            empty_height = input_widget.content_size.height
            assert empty_height == 1

            input_widget.text = "line one\nline two\nline three"
            await pilot.pause()
            assert input_widget.content_size.height >= 3

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
    async def test_mid_turn_plain_text_is_queued_as_steer_not_a_second_turn(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            app._turn_in_progress = True
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "keep this in mind"
            await pilot.press("enter")
            await pilot.pause()

            assert session._agent_session._pending_steer == ["keep this in mind"]
            assert session._agent_session._messages == []  # never started a second, concurrent turn
            assert "queued for the next turn boundary" in _transcript_text(app)
            assert "keep this in mind" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_mid_turn_explicit_steer_prefix_is_stripped(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            app._turn_in_progress = True
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/steer focus on the tests"
            await pilot.press("enter")
            await pilot.pause()

            assert session._agent_session._pending_steer == ["focus on the tests"]

    @pytest.mark.asyncio
    async def test_mid_turn_bare_steer_with_no_text_is_a_usage_error(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            app._turn_in_progress = True
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/steer"
            await pilot.press("enter")
            await pilot.pause()

            assert session._agent_session._pending_steer == []
            assert "usage:" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_mid_turn_stop_requests_abort_at_the_next_boundary(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            app._turn_in_progress = True
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/stop"
            await pilot.press("enter")
            await pilot.pause()

            assert session._agent_session._stop_requested is True
            assert "stop requested" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_submission_after_a_turn_finishes_is_a_normal_new_turn(self, tmp_path: Path) -> None:
        """_turn_in_progress must reset to False once run_turn() actually
        returns — otherwise every later submission would incorrectly be
        routed as a steer message forever."""
        session = _make_session(tmp_path, [faux_assistant_message("first reply")])
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "first message"
            await pilot.press("enter")
            await _settle(app, pilot)

            assert app._turn_in_progress is False
            assert len(session._agent_session._messages) == 2  # user + assistant

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

            footer = app.query_one("#app-footer", AppFooter)
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
    async def test_on_mount_wires_live_footer_updates(self, tmp_path: Path) -> None:
        """Phase T6: the footer must refresh live as a turn progresses, not
        only right before/after submission — on_mount wires
        InteractiveSession's status-change hook straight to _update_footer."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test():
            assert session._on_status_change == app._update_footer

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
    async def test_extension_command_can_confirm_via_ctx_ui(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "confirmcmd.py").write_text(
            "async def _handler(args_text, ctx):\n"
            '    ok = await ctx.ui.confirm("Really?")\n'
            '    return "confirmed" if ok else "declined"\n\n'
            'def extension(pi):\n    pi.register_command("doit", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/doit"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await _settle(app, pilot)
            assert "confirmed" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_extension_command_notify_reaches_the_app(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "notifycmd.py").write_text(
            'def _handler(args_text, ctx):\n    ctx.ui.notify("hi there")\n\n'
            'def extension(pi):\n    pi.register_command("ping", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        notified: list[str] = []
        app.notify = lambda message, **_k: notified.append(message)  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/ping"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert notified == ["hi there"]

    @pytest.mark.asyncio
    async def test_extension_widget_slots_show_and_hide(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "chrome.py").write_text(
            "def _handler(args_text, ctx):\n"
            '    ctx.ui.set_header("custom header")\n'
            '    ctx.ui.set_footer("custom footer")\n'
            '    ctx.ui.set_title("custom title")\n'
            '    ctx.ui.set_widget("custom widget")\n\n'
            'def extension(pi):\n    pi.register_command("chrome", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            header = app.query_one("#ext-header", Static)
            footer = app.query_one("#app-footer", AppFooter)
            widget = app.query_one("#ext-widget", Static)
            assert header.display is False
            # Unlike header/widget, the footer is never hidden — it's the
            # one always-visible structural region (see AppFooter's
            # docstring) — so before the extension runs it's already
            # displayed, just showing session status with no extension
            # content in it yet.
            assert footer.display is True
            assert "custom footer" not in _footer_text(app)
            assert widget.display is False

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/chrome"
            await pilot.press("enter")
            await _settle(app, pilot)

            assert header.display is True
            assert str(header.content) == "custom header"
            # set_footer() must land its content *inside* the same
            # structural footer widget, alongside the live status line —
            # never a second, independent footer box.
            assert footer.display is True
            assert "custom footer" in _footer_text(app)
            assert widget.display is True
            assert str(widget.content) == "custom widget"
            assert app.title == "custom title"

    @pytest.mark.asyncio
    async def test_set_tools_expanded_false_hides_the_result_preview(self, tmp_path: Path) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        target = tmp_path / "written.txt"
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "collapse.py").write_text(
            "def _handler(args_text, ctx):\n    ctx.ui.set_tools_expanded(False)\n\n"
            'def extension(pi):\n    pi.register_command("collapse", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(
            tmp_path,
            [
                faux_assistant_message(
                    [faux_tool_call("write", {"path": str(target), "content": "some content here"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("done"),
            ],
        )
        session._permission_mode = PermissionMode.ACCEPT_EDITS
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/collapse"
            await pilot.press("enter")
            await _settle(app, pilot)

            input_widget.text = "write the file"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert "some content here" not in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_extension_theme_get_set_via_ctx_ui(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "themecmd.py").write_text(
            "def _handler(args_text, ctx):\n"
            '    ctx.ui.set_theme("midnight")\n'
            '    return f"theme is now {ctx.ui.get_theme()}"\n\n'
            "def extension(pi):\n"
            '    pi.register_theme("midnight", primary="#1e1e2e")\n'
            '    pi.register_command("usetheme", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/usetheme"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert app.theme == "midnight"
            assert "theme is now midnight" in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_add_autocomplete_provider_merges_into_the_popup(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "mentions.py").write_text(
            'def _provider(text):\n    return ["@bob", "@alice"] if text.startswith("@") else []\n\n'
            "def _setup(args_text, ctx):\n    ctx.ui.add_autocomplete_provider(_provider)\n\n"
            'def extension(pi):\n    pi.register_command("mentions", _setup)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/mentions"
            await pilot.press("enter")
            await _settle(app, pilot)

            input_widget.text = "@"
            await pilot.pause()
            suggestions = app.query_one("#suggestions", OptionList)
            assert suggestions.display is True
            options = [str(suggestions.get_option_at_index(i).prompt) for i in range(suggestions.option_count)]
            assert options == ["@bob", "@alice"]

    @pytest.mark.asyncio
    async def test_editor_text_get_set_paste_via_ctx_ui(self, tmp_path: Path) -> None:
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "editorcmd.py").write_text(
            "def _set(args_text, ctx):\n    ctx.ui.set_editor_text('hello')\n\n"
            "def _paste(args_text, ctx):\n    ctx.ui.paste_to_editor(' world')\n\n"
            "def _report(args_text, ctx):\n    return f'editor says: {ctx.ui.get_editor_text()}'\n\n"
            "def extension(pi):\n"
            '    pi.register_command("set", _set)\n'
            '    pi.register_command("paste", _paste)\n'
            '    pi.register_command("report", _report)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)

            input_widget.text = "/set"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert input_widget.text == "hello"

            input_widget.text = "/paste"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert input_widget.text == " world"

            input_widget.text = "/report"
            await pilot.press("enter")
            await _settle(app, pilot)
            assert "editor says: " in _transcript_text(app)

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


class TestAppFooter:
    """Structural-footer invariants — see AppFooter's docstring
    (tui_app.py) for the "why": one always-visible region, intrinsic
    (content + border) height, no fixed `height: N`, no on_mount-cached
    sizing, and extension content merged into the same widget rather
    than a second competing footer."""

    @pytest.mark.asyncio
    async def test_exactly_one_footer_is_visible_on_mount(self, tmp_path: Path) -> None:
        """Teste 1 — sempre presente."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test():
            footers = app.query(AppFooter)
            assert len(footers) == 1
            assert footers.first().display is True
            # The old two-widget split is gone outright, not just hidden.
            assert len(app.query("#status-footer")) == 0
            assert len(app.query("#ext-footer")) == 0

    @pytest.mark.asyncio
    async def test_footer_height_grows_and_shrinks_with_content_lines(self, tmp_path: Path) -> None:
        """Teste 2 — altura dinâmica: N content lines -> N+2 rows (top +
        bottom border), for 1, 2, and 4 lines — never a fixed height."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer = app.query_one(AppFooter)

            footer.set_status("one line")
            await pilot.pause()
            assert footer.outer_size.height == 3

            footer.set_status("line one\nline two")
            await pilot.pause()
            assert footer.outer_size.height == 4

            footer.set_status("line one\nline two\nline three\nline four")
            await pilot.pause()
            assert footer.outer_size.height == 6

            # And back down again — it's not a high-water mark.
            footer.set_status("one line")
            await pilot.pause()
            assert footer.outer_size.height == 3

    @pytest.mark.asyncio
    async def test_multiline_status_is_rendered_in_full_without_truncation(self, tmp_path: Path) -> None:
        """Teste 3 — conteúdo multiline. Checked against what's actually
        painted (_footer_screen_text), not just what AppFooter was given
        (_footer_text) — see that helper's docstring for why the
        distinction matters."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer = app.query_one(AppFooter)
            lines = [f"status line {i}" for i in range(5)]
            footer.set_status("\n".join(lines))
            await pilot.pause()

            stored = _footer_text(app)
            painted = _footer_screen_text(app)
            for line in lines:
                assert line in stored
                assert line in painted
            assert footer.outer_size.height == len(lines) + 2

    @pytest.mark.asyncio
    async def test_footer_content_is_actually_painted_not_just_stored(self, tmp_path: Path) -> None:
        """Regression test: AppFooter.content (and .visual) reporting the
        right renderable is not proof it's on screen — see
        _footer_screen_text's docstring for the real bug this would have
        caught (set_status/set_extension routed through a shared
        "compute renderable, then self.update()" helper one call away
        from the setter; .content stayed correct but render_line came
        back blank on every row). Assert directly against what
        Widget.render_line actually paints, for both status-only and
        status+extension content."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer = app.query_one(AppFooter)

            footer.set_status("hello world")
            await pilot.pause()
            assert "hello world" in _footer_screen_text(app)

            footer.set_extension("extension line")
            await pilot.pause()
            painted = _footer_screen_text(app)
            assert "hello world" in painted
            assert "extension line" in painted

    @pytest.mark.asyncio
    async def test_status_transitions_update_the_same_footer_instance(self, tmp_path: Path) -> None:
        """Teste 4 — atualização: ready -> thinking... -> running: bash ->
        ready all land in the one footer widget, which is never rebuilt."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer = app.query_one(AppFooter)
            for status in ("ready", "thinking...", "running: bash", "ready"):
                session._status = status
                app._update_footer()
                await pilot.pause()
                assert status in _footer_text(app)
                assert app.query_one(AppFooter) is footer
            assert len(app.query(AppFooter)) == 1

    @pytest.mark.asyncio
    async def test_extension_set_footer_lands_inside_the_structural_footer(self, tmp_path: Path) -> None:
        """Teste 5 — extensão: ctx.ui.set_footer() content appears inside
        the existing structural footer, alongside live status, and no
        second footer widget is ever created."""
        ext_dir = tmp_path / ".pi" / "extensions"
        ext_dir.mkdir(parents=True)
        (ext_dir / "chrome.py").write_text(
            'def _handler(args_text, ctx):\n    ctx.ui.set_footer("custom footer")\n\n'
            'def extension(pi):\n    pi.register_command("chrome", _handler)\n',
            encoding="utf-8",
        )
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/chrome"
            await pilot.press("enter")
            await _settle(app, pilot)

            assert len(app.query(AppFooter)) == 1
            assert len(app.query("#ext-footer")) == 0
            text = _footer_text(app)
            assert "custom footer" in text
            # Live status is still there too — merged, not replaced.
            assert "ready" in text
            # And it's actually painted, not just stored — see
            # _footer_screen_text's docstring.
            painted = _footer_screen_text(app)
            assert "custom footer" in painted
            assert "ready" in painted

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40), (160, 50)])
    async def test_footer_and_layout_stay_correct_at_various_terminal_sizes(
        self, tmp_path: Path, size: tuple[int, int]
    ) -> None:
        """Teste 9 (tamanhos obrigatórios) — footer visible, full viewport
        width, and every region (transcript/footer/prompt) fits inside the
        viewport with no overlap or overflow."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            width, height = size
            footer = app.query_one(AppFooter)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt = app.query_one("#prompt-input", PromptTextArea)

            assert footer.display is True
            assert footer.outer_size.width == width
            assert transcript.region.bottom <= height
            assert footer.region.bottom <= height
            assert prompt.region.bottom <= height
            # Non-overlapping, top-to-bottom stacking order.
            assert transcript.region.bottom <= footer.region.y
            assert footer.region.bottom <= prompt.region.y

    @pytest.mark.asyncio
    async def test_footer_width_and_layout_recompute_on_terminal_resize(self, tmp_path: Path) -> None:
        """Teste 10 — redimensionamento: a live resize (not just a
        different size chosen once at app start) must reflow the footer's
        width and the whole layout — nothing here may depend on a value
        cached once in on_mount."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test(size=(80, 24)) as pilot:
            footer = app.query_one(AppFooter)
            await pilot.pause()
            assert footer.outer_size.width == 80

            await pilot.resize_terminal(120, 40)
            await pilot.pause()
            assert footer.outer_size.width == 120

            await pilot.resize_terminal(60, 20)
            await pilot.pause()
            assert footer.outer_size.width == 60

    @pytest.mark.asyncio
    async def test_large_footer_shrinks_transcript_without_overflow_and_prompt_stays_usable(
        self, tmp_path: Path
    ) -> None:
        """Teste 7 — footer grande: footer grows, transcript shrinks to
        make room (never overflows the viewport), and the prompt is still
        present, correctly positioned, and still accepts input."""
        session = _make_session(tmp_path)
        app = PiApp(session)
        async with app.run_test(size=(80, 24)) as pilot:
            footer = app.query_one(AppFooter)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt = app.query_one("#prompt-input", PromptTextArea)

            footer.set_status("one line")
            await pilot.pause()
            small_transcript_height = transcript.size.height

            footer.set_status("\n".join(f"status line {i}" for i in range(6)))
            await pilot.pause()

            # 6 content lines vs. 1 before: +5 lines, same top/bottom
            # border either way, so the footer grows by exactly 5 rows —
            # and the transcript (the only 1fr region) must give up
            # exactly that much space, not more, not less.
            assert footer.outer_size.height == 6 + 2
            assert transcript.size.height == small_transcript_height - 5
            assert transcript.size.height > 0
            assert footer.region.bottom <= 24
            assert prompt.region.bottom <= 24
            assert transcript.region.bottom <= footer.region.y

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "still usable"
            await pilot.pause()
            assert input_widget.text == "still usable"


class TestModelSelectorTransient:
    """Second-iteration bugs: the bare `/model` picker's list is
    transient interaction state, never persistent footer state — see
    TransientFooterContent and PiApp._select_model_transient. Covers the
    exact required sequence: MODEL_SELECTOR_OPEN -> show models ->
    MODEL_SELECTED -> hide models -> switch model -> append "switched
    to <model>" to transcript -> persistent footer remains visible."""

    @pytest.mark.asyncio
    async def test_model_list_is_hidden_before_and_after_selection(self, tmp_path: Path) -> None:
        session = _make_multi_model_session(tmp_path, ["model-a", "model-b", "model-c"])
        app = PiApp(session)
        async with app.run_test() as pilot:
            transient = app.query_one("#transient-footer", TransientFooterContent)
            assert transient.display is False

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/model"
            await pilot.press("enter")
            await pilot.pause()
            assert transient.display is True
            assert transient.option_count == 3

            await pilot.press("enter")  # pick whatever's highlighted (model-a)
            await _settle(app, pilot)

            # Gone immediately, not left behind — this is the exact bug
            # report: "lista de modelos continua visível" after picking.
            assert transient.display is False
            assert transient.option_count == 0

    @pytest.mark.asyncio
    async def test_selecting_a_model_switches_it_and_logs_to_transcript_once(self, tmp_path: Path) -> None:
        session = _make_multi_model_session(tmp_path, ["model-a", "model-b"])
        app = PiApp(session)
        async with app.run_test() as pilot:
            assert session._model.id == "model-a"

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/model"
            await pilot.press("enter")
            await pilot.pause()

            transient = app.query_one("#transient-footer", TransientFooterContent)
            transient.highlighted = 1  # model-b
            await pilot.press("enter")
            await _settle(app, pilot)

            assert session._model.id == "model-b"
            text = _transcript_text(app)
            assert "switched to" in text
            assert "model-b" in text
            # Not duplicated as "models... models... switched to...".
            assert text.count("switched to") == 1

    @pytest.mark.asyncio
    async def test_escape_cancels_without_switching_or_leaving_the_list_visible(self, tmp_path: Path) -> None:
        session = _make_multi_model_session(tmp_path, ["model-a", "model-b"])
        app = PiApp(session)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/model"
            await pilot.press("enter")
            await pilot.pause()

            transient = app.query_one("#transient-footer", TransientFooterContent)
            assert transient.display is True

            await pilot.press("escape")
            await _settle(app, pilot)

            assert transient.display is False
            assert session._model.id == "model-a"  # unchanged
            assert "switched to" not in _transcript_text(app)

    @pytest.mark.asyncio
    async def test_persistent_footer_stays_visible_with_both_borders_throughout(self, tmp_path: Path) -> None:
        """Teste de persistência: opening/closing the picker never touches
        AppFooter's own visibility or borders — it's a separate widget
        entirely (TransientFooterContent), not footer content."""
        session = _make_multi_model_session(tmp_path, ["model-a", "model-b"])
        app = PiApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            footer = app.query_one("#app-footer", AppFooter)
            footer_identity = id(footer)
            assert footer.display is True
            assert _footer_has_both_borders(app)

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "/model"
            await pilot.press("enter")
            await pilot.pause()

            # Model list open: footer is completely unaffected.
            assert app.query_one("#app-footer", AppFooter) is footer
            assert footer.display is True
            assert _footer_has_both_borders(app)

            await pilot.press("enter")
            await _settle(app, pilot)

            assert id(app.query_one("#app-footer", AppFooter)) == footer_identity
            assert footer.display is True
            assert _footer_has_both_borders(app)
            # And the status line now reflects the switched-to model.
            assert "model-a" in _footer_screen_text(app)


class TestFooterSurvivesStreaming:
    """Second-iteration BUG 3: the footer container/visibility/borders
    must never change during READY -> THINKING -> STREAMING -> TOOL ->
    STREAMING -> READY. Sampled by wrapping PiApp._update_footer (the
    exact hook InteractiveSession fires after every event during a turn
    — see interactive_mode._handle_event's on_status_change call) so
    every real status transition is observed deterministically, with no
    sleeps/timers/polling."""

    @pytest.mark.asyncio
    async def test_footer_widget_identity_is_stable_across_a_full_turn(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path, [faux_assistant_message("hello from tui")])
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer_id = id(app.query_one("#app-footer", AppFooter))

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "hi"
            await pilot.press("enter")
            await _settle(app, pilot)

            assert id(app.query_one("#app-footer", AppFooter)) == footer_id

    @pytest.mark.asyncio
    async def test_footer_never_hides_or_loses_borders_across_ready_thinking_tool_streaming_ready(
        self, tmp_path: Path
    ) -> None:
        from pi_ai import StopReason
        from pi_ai.providers.faux import faux_tool_call

        session = _make_session(
            tmp_path,
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "x.txt"})],
                    stop_reason=StopReason.TOOL_USE,
                ),
                faux_assistant_message("final streamed response"),
            ],
        )
        app = PiApp(session)
        async with app.run_test() as pilot:
            footer = app.query_one("#app-footer", AppFooter)
            footer_identity = id(footer)
            transcript = app.query_one("#transcript", VerticalScroll)
            prompt = app.query_one("#prompt-input", PromptTextArea)

            samples: list[tuple[bool, bool, bool]] = []
            # Patched onto InteractiveSession, not PiApp: on_mount binds
            # session._on_status_change = self._update_footer *once*,
            # capturing that bound-method object — _handle_event later
            # calls that already-captured reference directly on every
            # event, so patching app._update_footer as a fresh instance
            # attribute after mount would never actually intercept those
            # calls (only code that looks up self._update_footer fresh
            # each time, e.g. _handle_submission's own end-of-turn call,
            # would see it). The session's hook is the one InteractiveSession
            # genuinely fires after every event — see _handle_event's
            # on_status_change call in interactive_mode.py.
            original_on_status_change = session._on_status_change
            assert original_on_status_change is not None

            def _sampling_status_change() -> None:
                original_on_status_change()
                live_footer = app.query_one("#app-footer", AppFooter)
                # Structural border check (styles.border, what CSS
                # declared), not a pixel probe here: render_line reflects
                # whatever the compositor's *last completed* layout pass
                # painted, and this sampler runs synchronously mid-event
                # (right when a status change fires), potentially before
                # that pass has caught up with a just-changed content
                # height — a timing artifact of probing off-cycle, not a
                # real dropped border (Textual only ever paints a frame
                # after a message batch fully settles, never mid-callback).
                # The pixel-exact check (_footer_has_both_borders) is used
                # instead for the pre/post assertions below, where an
                # explicit pilot.pause()/_settle guarantees a stable,
                # fully-composited frame to inspect.
                samples.append(
                    (
                        live_footer.display,
                        id(live_footer) == footer_identity,
                        live_footer.styles.border_top[0] != "" and live_footer.styles.border_bottom[0] != "",
                    )
                )

            session._on_status_change = _sampling_status_change

            input_widget = app.query_one("#prompt-input", PromptTextArea)
            input_widget.text = "please read the file"
            await pilot.press("enter")
            await _settle(app, pilot)

            # #transcript / #app-footer / #prompt-input all still exist,
            # unchanged in kind and position.
            assert app.query_one("#transcript", VerticalScroll) is transcript
            assert app.query_one("#app-footer", AppFooter) is footer
            assert app.query_one("#prompt-input", PromptTextArea) is prompt

            # At least one sample per status change (tool start, tool
            # end, done) actually fired — not a vacuously-true empty list.
            assert len(samples) >= 2
            assert all(display for display, _, _ in samples), samples
            assert all(same_identity for _, same_identity, _ in samples), samples
            assert all(has_borders for _, _, has_borders in samples), samples

            assert footer.display is True
            assert _footer_has_both_borders(app)
            assert "final streamed response" in _transcript_text(app)


def test_console_output_sink_is_the_default_for_a_plain_session(tmp_path: Path) -> None:
    """A plain session (no explicit `output=`, what the classic REPL
    constructs) still ends up backed by a real ConsoleOutputSink — wrapped
    in FooterAwareOutputSink now, which keeps the mid-turn status footer
    visible (see TestMidTurnFooterVisibility in test_interactive_mode.py)
    but still forwards every print straight to the console underneath,
    same as before that wrapper existed."""
    session = _make_session(tmp_path)
    assert isinstance(session._output, FooterAwareOutputSink)
    assert isinstance(session._output._inner, ConsoleOutputSink)
