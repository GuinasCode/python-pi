"""Where interactive event/turn rendering actually goes.

InteractiveSession's turn/event handling (streaming text, tool call
previews, diffs, slash-command replies, ...) is real, shared logic that
both the classic REPL and the new Textual app (pi_coding_agent.tui_app,
launched via --ui-mode fullscreen) need identically. Rather than
duplicating that logic per front-end, InteractiveSession renders through
an injected OutputSink instead of a hardcoded rich Console — the REPL
gets ConsoleOutputSink (prints straight to the terminal, today's
behavior, unchanged), the Textual app gets its own sink that appends
into a transcript widget instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from rich.console import Console, RenderableType


class OutputSink(Protocol):
    """Sink for interactive-mode output. Two calls only, deliberately:
    everything InteractiveSession renders is either a line of rich markup
    text or a pre-built rich renderable (Markdown, a diff panel, ...)."""

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        """Print a line of rich markup text."""
        ...

    def print_renderable(self, renderable: RenderableType) -> None:
        """Print a pre-built rich renderable (e.g. Markdown(...), a diff panel)."""
        ...


class ConsoleOutputSink:
    """Prints straight to a rich Console — what the classic REPL always did."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        self._console.print(markup, end=end)

    def print_renderable(self, renderable: RenderableType) -> None:
        # soft_wrap=False overrides the console's own soft_wrap=True default
        # (set for single-line markup prints — see interactive_mode._console)
        # for exactly this call. Without it, Rich's render_options.no_wrap
        # propagates into every renderable's __rich_console__ — including
        # Markdown's paragraph/list-item Text elements and diff Table cells
        # — disabling Rich's own word-wrap. A line longer than the terminal
        # width then has no embedded newline at all; whether it visually
        # continues on the next row is left entirely to the terminal's own
        # auto-wrap, which some terminals don't honor the same way Rich's
        # crop/pad buffer flush does — the intermittent line truncation this
        # fixes.
        self._console.print(renderable, soft_wrap=False)


class FooterAwareOutputSink:
    """Wraps another OutputSink, calling `before` immediately before
    each print and `after` immediately after. The classic REPL uses
    this to keep its status footer (session/mode info, pinned to the
    bottom of the terminal) visible for the whole turn — thinking,
    streaming text, running tools — not just while the input box itself
    is on screen: `before` erases the footer so the new line lands where
    it would have anyway, `after` redraws it below whatever just
    printed, so the two never end up interleaved or duplicated on the
    screen. Both hooks are no-ops outside a turn (see
    InteractiveSession._footer_active), so slash-command output and
    other non-turn prints are completely unaffected."""

    def __init__(self, inner: OutputSink, *, before: Callable[[], None], after: Callable[[], None]) -> None:
        self._inner = inner
        self._before = before
        self._after = after

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        self._before()
        self._inner.print(markup, end=end)
        self._after()

    def print_renderable(self, renderable: RenderableType) -> None:
        self._before()
        self._inner.print_renderable(renderable)
        self._after()


class NullOutputSink:
    """Discards everything. Used by tests/harnesses that only care about
    AgentSession state, not what would have been printed."""

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        pass

    def print_renderable(self, renderable: Any) -> None:
        pass
