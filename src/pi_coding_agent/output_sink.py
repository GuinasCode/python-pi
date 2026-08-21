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
        self._console.print(renderable)


class NullOutputSink:
    """Discards everything. Used by tests/harnesses that only care about
    AgentSession state, not what would have been printed."""

    def print(self, markup: str = "", *, end: str = "\n") -> None:
        pass

    def print_renderable(self, renderable: Any) -> None:
        pass
