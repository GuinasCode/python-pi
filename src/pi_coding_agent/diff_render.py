"""Render a unified diff as line-numbered, IDE-style colored output.

Shared by the interactive-mode terminal transcript and the HTML session
export — both print through a rich ``Console``, so a single renderer
produces consistent output in both places.
"""

from __future__ import annotations

import re

from rich.table import Table
from rich.text import Text

from pi_coding_agent.styles import DIM_STYLE, PASTEL_GREEN, PASTEL_RED, TEXT_STYLE

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# Pastel add/remove colors, dark-background friendly.
_ADD_STYLE = f"{PASTEL_GREEN} on #1e3a2a"
_DEL_STYLE = f"{PASTEL_RED} on #3a1e1e"
_CONTEXT_STYLE = TEXT_STYLE
_LINENO_STYLE = DIM_STYLE


def render_diff(diff_lines: list[str]) -> Table:
    """Render unified-diff lines (as from ``difflib.unified_diff``) as a table."""
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style=_LINENO_STYLE, no_wrap=True)
    table.add_column(justify="right", style=_LINENO_STYLE, no_wrap=True)
    table.add_column()

    old_no = new_no = 0
    for raw in diff_lines:
        line = raw.rstrip("\n")
        if line.startswith("---") or line.startswith("+++"):
            continue

        hunk = _HUNK_RE.match(line)
        if hunk:
            old_no, new_no = int(hunk.group(1)), int(hunk.group(2))
            table.add_row("", "", Text(line, style=_LINENO_STYLE))
            continue

        if line.startswith("+"):
            table.add_row("", str(new_no), Text(line, style=_ADD_STYLE))
            new_no += 1
        elif line.startswith("-"):
            table.add_row(str(old_no), "", Text(line, style=_DEL_STYLE))
            old_no += 1
        else:
            table.add_row(str(old_no), str(new_no), Text(line, style=_CONTEXT_STYLE))
            old_no += 1
            new_no += 1

    return table
