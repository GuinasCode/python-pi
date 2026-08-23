"""Export a session's message history to a self-contained HTML file.

Renders each message as Markdown (with syntax-highlighted code blocks) using
rich's HTML export, avoiding the vendored JS bundle the original TypeScript
implementation shipped (highlight.js/marked.js) in favor of Python's own
markdown/syntax rendering, which is already a project dependency via rich.

Output style: left-aligned throughout, no boxed titles, pastel accent
colors instead of rich's default cyan/magenta (see ``pi_coding_agent.
styles``), and unified diffs rendered as line-numbered +/- tables (see
``pi_coding_agent.diff_render``).
"""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console, Group
from rich.syntax import Syntax
from rich.text import Text

from pi_coding_agent.diff_render import render_diff
from pi_coding_agent.markdown_render import LeftMarkdown as Markdown
from pi_coding_agent.session_manager import SessionEntry
from pi_coding_agent.styles import DIM_STYLE, PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED, PI_THEME

_HTML_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _render_user_content(content: Any) -> Any:
    if isinstance(content, str):
        return Markdown(content)
    parts: list[Any] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(Markdown(block.get("text", "")))
        elif isinstance(block, dict) and block.get("type") == "image":
            parts.append(Text("[image attachment]", style=DIM_STYLE + " italic"))
    return Group(*parts) if parts else Text("")


def _render_assistant_content(content: list[dict[str, Any]]) -> Any:
    parts: list[Any] = []
    for block in content or []:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(Markdown(block.get("text", "")))
        elif block_type == "thinking":
            parts.append(Text("thinking", style=DIM_STYLE))
            parts.append(Text(block.get("thinking", ""), style=DIM_STYLE + " italic"))
        elif block_type == "toolCall":
            args_str = "\n".join(f"{k}: {v!r}" for k, v in (block.get("arguments") or {}).items())
            parts.append(Text(f"tool call: {block.get('name', '?')}", style=f"bold {PASTEL_BLUE}"))
            parts.append(Syntax(args_str or "(no arguments)", "yaml", word_wrap=True, background_color="default"))
    return Group(*parts) if parts else Text("")


def _render_tool_result_content(data: dict[str, Any]) -> Any:
    is_error = bool(data.get("is_error"))
    label_style = f"bold {PASTEL_RED if is_error else PASTEL_GREEN}"
    parts: list[Any] = [Text(f"tool result: {data.get('tool_name', '?')}", style=label_style)]

    diff = (data.get("details") or {}).get("diff") if isinstance(data.get("details"), dict) else None
    if diff:
        parts.append(render_diff(diff))
        return Group(*parts)

    text = "\n".join(block.get("text", "") for block in (data.get("content") or []) if block.get("type") == "text")
    parts.append(Text(text.strip() or "(no output)"))
    return Group(*parts)


def render_session_html(entries: list[SessionEntry], *, title: str = "Pi Session") -> str:
    """Render a session's message entries to a self-contained HTML document."""
    console = Console(record=True, file=io.StringIO(), width=100, theme=PI_THEME)

    for entry in entries:
        if entry.kind != "message":
            continue
        data = entry.data
        role = data.get("role", "user")

        if role == "user":
            console.print(Markdown("## User"))
            console.print(_render_user_content(data.get("content", "")))
        elif role == "assistant":
            console.print(Markdown(f"## Assistant ({data.get('model', '?')})"))
            console.print(_render_assistant_content(data.get("content", [])))
        elif role == "toolResult":
            console.print(Markdown("## Tool"))
            console.print(_render_tool_result_content(data))
        else:
            continue
        console.print()

    code_format = (
        "<pre style=\"font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;"
        'white-space:pre-wrap">{code}</pre>'
    )
    body = console.export_html(inline_styles=True, code_format=code_format)
    return _HTML_WRAPPER.format(title=title, body=body)
