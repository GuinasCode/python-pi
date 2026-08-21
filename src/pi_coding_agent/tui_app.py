"""Textual-based interactive UI (``--ui-mode fullscreen`` / ``--alt``).

Phase T0 of the extension-system UI foundation (see ARCHITECTURE.md's
extension-system status note for the full phase plan, T0-T6 then G/H).
A real alt-screen Textual app that reuses InteractiveSession's event and
slash-command handling — via the OutputSink it already renders through —
instead of duplicating that logic for a second front-end. The classic
REPL (``interactive_mode.repl_loop``) is untouched and stays the default;
this is opt-in via ``--ui-mode fullscreen``/``--alt`` while it matures.

Scope for T0 specifically: the app shell (transcript + input + footer),
streaming turn rendering, and slash commands (including extension-
registered ones). Deliberately NOT wired yet:

- Permission-mode confirmation dialogs: needs Phase T2's modal/dialog
  system. Shift+Tab still cycles the mode and updates the footer for
  visual parity with the REPL, but ``permission_gate`` is left unset for
  this app for now, so tools always run without a confirmation prompt —
  same as running with no permission mode configured at all. This is a
  real, intentional scope cut for T0, not a silent regression: plan/ask
  mode semantics return once T2 lands a dialog InteractiveSession can
  await on without blocking Textual's event loop (unlike the REPL's
  blocking ``input()``, which cannot run here).
- register_shortcut / rendering hooks (Phases T3-T6, G, H).
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.widgets import Static

from pi_ai import StopReason
from pi_coding_agent.interactive_mode import InteractiveSession
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
        self._update_footer()
        self.query_one("#prompt-input", PromptTextArea).focus()

    def _update_footer(self) -> None:
        mode_label = permission_mode_label(self._session._permission_mode)
        footer = self.query_one("#status-footer", Static)
        footer.update(f"{self._session._state_line()}\n{mode_label} [dim](shift+tab to cycle)[/dim]")

    def action_cycle_permission_mode(self) -> None:
        self._session._cycle_permission_mode()
        self._update_footer()

    async def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        text = event.value.strip()
        event.text_area.text = ""
        if not text:
            return

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
