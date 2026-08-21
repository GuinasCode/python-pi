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


class TestNullOutputSink:
    def test_print_and_print_renderable_are_noops(self) -> None:
        sink = NullOutputSink()
        sink.print("anything")  # does not raise
        sink.print_renderable(Text("anything"))  # does not raise
