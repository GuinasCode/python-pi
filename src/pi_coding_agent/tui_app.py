"""Textual-based interactive UI (``--ui-mode fullscreen`` / ``--alt``).

Phase T0 of the extension-system UI foundation (see ARCHITECTURE.md's
extension-system status note for the full phase plan, T0-T6 then G/H).
A real alt-screen Textual app that reuses InteractiveSession's event and
slash-command handling — via the OutputSink it already renders through —
instead of duplicating that logic for a second front-end. The classic
REPL (``interactive_mode.repl_loop``) is untouched and stays the default;
this is opt-in via ``--ui-mode fullscreen``/``--alt`` while it matures.

Scope for T0/T1/T2/T3 so far: the app shell (transcript + input + footer),
streaming turn rendering, slash commands (including extension-registered
ones), a real multi-line prompt editor, permission-mode confirmation via a
modal dialog (Phase T2), and a keybinding dispatcher for
extension-registered shortcuts (Phase T3). Deliberately NOT wired yet:
rendering hooks (Phases T4-T6, G, H).
"""

from __future__ import annotations

from typing import Any, ClassVar

from rich.console import RenderableType
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.widgets import Static

from pi_ai import StopReason
from pi_coding_agent.dialogs import ConfirmDialog
from pi_coding_agent.interactive_mode import InteractiveSession, _fmt_args
from pi_coding_agent.permission_mode import permission_mode_label
from pi_coding_agent.prompt_editor import PromptTextArea

__all__ = ["PiApp"]


class _TranscriptSink:
    """OutputSink that appends into the transcript ScrollView instead of
    printing to stdout.

    InteractiveSession's event handler calls ``print(delta, end="")``
    repeatedly while a response streams in (one call per token) — mounting
    a fresh widget per call would be both wasteful and visually wrong (a
    wall of one-token widgets instead of a paragraph). Instead, calls with
    ``end=""`` accumulate into the *same* trailing Static via ``.update()``;
    a call with the default ``end="\\n"`` (or a full renderable, e.g. a
    Markdown block or diff) finalizes that line and starts a fresh one.
    """

    def __init__(self, transcript: VerticalScroll) -> None:
        self._transcript = transcript
        self._current: Static | None = None
        self._current_text = ""

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        if self._current is None:
            self._current = Static("", markup=True)
            self._transcript.mount(self._current)
        self._current_text += markup
        self._current.update(self._current_text)
        self._transcript.scroll_end(animate=False)
        if end == "\n":
            self._current = None
            self._current_text = ""

    def print_renderable(self, renderable: RenderableType) -> None:
        self._current = None
        self._current_text = ""
        self._transcript.mount(Static(renderable))
        self._transcript.scroll_end(animate=False)


class PiApp(App[None]):
    """Alt-screen Textual front-end for an InteractiveSession."""

    CSS = """
    #transcript {
        height: 1fr;
        padding: 0 1;
    }
    #status-footer {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: $panel;
    }
    #prompt-input {
        dock: bottom;
        height: 3;
        border: solid $accent;
    }
    """

    # priority=True: Textual's default Screen already binds shift+tab to
    # focus_previous, which — being a closer DOM ancestor of the focused
    # Input than the App — would otherwise win the normal walk-up-from-
    # focused-widget resolution and this binding would never fire. Priority
    # bindings are checked before that walk, so ours wins instead.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("shift+tab", "cycle_permission_mode", "Cycle permission mode", show=False, priority=True),
    ]

    def __init__(self, session: InteractiveSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="transcript")
        yield Static(id="status-footer")
        yield PromptTextArea(id="prompt-input")

    def on_mount(self) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        self._session._output = _TranscriptSink(transcript)
        self._session._confirm_tool_fn = self._confirm_tool_via_modal
        self._update_footer()
        self.query_one("#prompt-input", PromptTextArea).focus()

    async def _confirm_tool_via_modal(self, tool_name: str, args: dict[str, Any]) -> bool:
        """InteractiveSession._confirm_tool_fn: replaces the REPL's
        blocking y/N input() with an async modal dialog — the blocking
        version would freeze this app's event loop entirely."""
        args_str = _fmt_args(args)
        question = f"Allow [bold]{tool_name}[/bold]({args_str})?"
        result = await self.push_screen_wait(ConfirmDialog(question))
        return bool(result)

    def _update_footer(self) -> None:
        mode_label = permission_mode_label(self._session._permission_mode)
        footer = self.query_one("#status-footer", Static)
        footer.update(f"{self._session._state_line()}\n{mode_label} [dim](shift+tab to cycle)[/dim]")

    def action_cycle_permission_mode(self) -> None:
        self._session._cycle_permission_mode()
        self._update_footer()

    def on_key(self, event: events.Key) -> None:
        """Phase T3 keybinding dispatcher: fire a matching
        extension-registered shortcut (``pi.register_shortcut``) before the
        key reaches whatever widget is focused — e.g. so a shortcut key
        doesn't also get typed as a literal character into the prompt
        editor. A no-op check on every keystroke when no extension
        registered any shortcuts, which is the common case.
        """
        if not any(s.key == event.key for s in self._session._agent_session.get_extension_shortcuts()):
            return
        event.stop()
        event.prevent_default()
        self._dispatch_shortcut(event.key)

    @work(exclusive=True)
    async def _dispatch_shortcut(self, key: str) -> None:
        """Runs as a worker for the same reason _handle_submission does:
        the handler may end up awaiting push_screen_wait several calls
        down (e.g. a future confirm/select dialog), which only works from
        inside a worker context."""
        await self._session._handle_extension_shortcut(key)
        self._update_footer()

    def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        text = event.value.strip()
        event.text_area.text = ""
        if not text:
            return
        self._handle_submission(text)

    @work(exclusive=True)
    async def _handle_submission(self, text: str) -> None:
        """Runs as a Textual worker (not a plain message-handler coroutine)
        because it may end up calling push_screen_wait (via
        _confirm_tool_via_modal, several calls down through run_turn's tool
        loop) — push_screen_wait only works from inside a worker context.
        """
        if text.startswith("/"):
            should_continue = await self._session._handle_command(text)
            self._update_footer()
            if not should_continue:
                self.exit()
            return

        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static(f"[bold]> {text}[/bold]", markup=True))
        transcript.scroll_end(animate=False)

        result = await self._session.run_turn(text)
        self._update_footer()
        if result is not None and result.stop_reason == StopReason.ERROR:
            error_msg = result.error_message or "Unknown error"
            transcript.mount(Static(f"[red]error:[/red] {error_msg}", markup=True))
            transcript.scroll_end(animate=False)
