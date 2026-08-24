"""Tests for pi_coding_agent.output_sink."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from pi_coding_agent.output_sink import ConsoleOutputSink, NullOutputSink


class TestConsoleOutputSink:
    def test_print_writes_markup_to_the_console(self) -> None:
        console = Console(file=None, force_terminal=False, width=80, record=True)
        sink = ConsoleOutputSink(console)
        sink.print("[bold]hi[/bold]")
        assert "hi" in console.export_text()

    def test_print_respects_end(self) -> None:
        console = Console(file=None, force_terminal=False, width=80, record=True)
        sink = ConsoleOutputSink(console)
        sink.print("a", end="")
        sink.print("b", end="")
        text = console.export_text()
        assert "ab" in text.replace("\n", "")

    def test_print_renderable_renders_it(self) -> None:
        console = Console(file=None, force_terminal=False, width=80, record=True)
        sink = ConsoleOutputSink(console)
        sink.print_renderable(Text("a renderable"))
        assert "a renderable" in console.export_text()

    def test_print_renderable_wraps_long_lines_despite_console_soft_wrap(self) -> None:
        """A console-level soft_wrap=True (as interactive_mode's shared
        _console uses, for single-line markup prints) must not leak into
        print_renderable: without overriding it per-call, a long Text line
        gets no embedded newline at all and relies entirely on the
        terminal's own auto-wrap — which is what produced the intermittent
        cut-off-without-wrap bug reported against Markdown assistant
        replies. print_renderable must always word-wrap at the console
        width regardless of the console's soft_wrap default."""
        console = Console(file=None, force_terminal=False, width=20, soft_wrap=True, record=True)
        sink = ConsoleOutputSink(console)
        long_line = "one two three four five six seven eight"
        sink.print_renderable(Text(long_line))
        text = console.export_text()
        assert "one two three four" in text  # actually wrapped, not cropped
        assert all(len(line) <= 20 for line in text.splitlines())
        # every word survived somewhere in the (multi-line) output
        for word in long_line.split():
            assert word in text


class TestNullOutputSink:
    def test_print_and_print_renderable_are_noops(self) -> None:
        sink = NullOutputSink()
        sink.print("anything")  # does not raise
        sink.print_renderable(Text("anything"))  # does not raise
