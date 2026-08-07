"""Left-aligned, box-free Markdown rendering.

``rich.markdown.Markdown`` hardcodes headings to be centered, and wraps
level-1 headings in a bordered panel (see ``rich.markdown.Heading.
__rich_console__``). This overrides that so headings render like normal
left-aligned text, distinguished only by weight/color — no boxes, no
centering.
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, Markdown, MarkdownElement
from rich.text import Text

from pi_coding_agent.styles import TITLE_STYLE


class _LeftHeading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text
        text.justify = "left"
        text.stylize(TITLE_STYLE)
        yield text
        if self.tag == "h1":
            yield Text("")


class LeftMarkdown(Markdown):
    """Markdown renderer with left-aligned, unboxed headings."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _LeftHeading,
    }
