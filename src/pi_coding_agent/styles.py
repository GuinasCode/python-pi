"""Shared color palette for terminal and HTML session output.

Pastel blue/red instead of rich's default cyan/magenta; white for titles,
grey for body text.
"""

from __future__ import annotations

from rich.theme import Theme

PASTEL_BLUE = "#8ab4f8"
PASTEL_RED = "#f28b82"
PASTEL_GREEN = "#a6e3a1"
PASTEL_YELLOW = "#f9e2af"

TITLE_STYLE = "bold white"
TEXT_STYLE = "grey70"
DIM_STYLE = "grey50"

# rich.markdown ships its own default styles (inline/fenced code and
# blockquotes in cyan, blockquotes and lists in magenta/cyan, ...) that
# the "pastel blue/red instead of cyan/magenta" choice above never
# actually reached — those only apply to markup this project writes
# itself, not to a Console's built-in theme, which is what every
# ``rich.markdown.Markdown`` render (assistant replies, ``/export``'s
# HTML) falls back to unless a Console overrides it. Passed as
# ``Console(theme=PI_THEME, ...)`` at every Console this project
# constructs, so a code span or blockquote in a model's reply gets the
# same palette as everything else instead of rich's raw defaults.
PI_THEME = Theme(
    {
        "markdown.code": f"bold {PASTEL_BLUE}",
        "markdown.code_block": PASTEL_BLUE,
        "markdown.block_quote": PASTEL_RED,
        "markdown.list": PASTEL_BLUE,
        "markdown.link": PASTEL_BLUE,
        "markdown.link_url": f"underline {PASTEL_BLUE}",
    }
)
