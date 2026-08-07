"""Built-in tools for the Pi coding agent.

Mirrors packages/coding-agent/src/core/tools/.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_agent_core.shell import build_subprocess_args


@dataclass
class ToolResult:
    """Result from a tool execution."""

    content: list[dict[str, str]]
    details: Any = None
    is_error: bool = False
    usage: dict[str, int] | None = None


def _text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


# --- Bash tool ---


def execute_bash(
    command: str,
    *,
    cwd: str | Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> ToolResult:
    """Execute a bash command and return the output.

    Mirrors packages/coding-agent/src/core/tools/bash.ts.
    """
    run_args, use_shell = build_subprocess_args(command)
    try:
        result = subprocess.run(
            run_args,
            shell=use_shell,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr
        output = output.strip()
        if not output:
            output = "(no output)"
        is_error = result.returncode != 0
        if is_error and output:
            output = f"Command failed with exit code {result.returncode}\n{output}"
        return ToolResult(
            content=[_text_content(output)],
            details={"exitCode": result.returncode, "command": command},
            is_error=is_error,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            content=[_text_content(f"Command timed out after {timeout} seconds")],
            details={"command": command, "timeout": timeout},
            is_error=True,
        )
    except Exception as exc:
        return ToolResult(
            content=[_text_content(f"Error executing command: {exc}")],
            details={"command": command},
            is_error=True,
        )


# --- Read tool ---


def read_file(
    path: str | Path,
    *,
    offset: int = 0,
    limit: int = 2000,
) -> ToolResult:
    """Read a file and return its contents.

    Mirrors packages/coding-agent/src/core/tools/read.ts.
    """
    file_path = Path(path)
    if not file_path.exists():
        return ToolResult(
            content=[_text_content(f"File not found: {path}")],
            is_error=True,
        )
    if not file_path.is_file():
        return ToolResult(
            content=[_text_content(f"Not a file: {path}")],
            is_error=True,
        )
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return ToolResult(
            content=[_text_content(f"Error reading file: {exc}")],
            is_error=True,
        )

    lines = text.splitlines()
    total = len(lines)
    if offset > 0 and offset >= total:
        return ToolResult(
            content=[_text_content(f"Offset {offset} is beyond file length {total}")],
            is_error=True,
        )

    start = offset
    end = min(start + limit, total)
    selected = lines[start:end]

    # Format with line numbers
    formatted_lines = []
    for i, line in enumerate(selected):
        line_num = start + i + 1
        formatted_lines.append(f"{line_num:6d}|{line}")
    output = "\n".join(formatted_lines)

    return ToolResult(
        content=[_text_content(output)],
        details={"totalLines": total, "startLine": start + 1, "endLine": end},
    )


# --- Write tool ---


def write_file(path: str | Path, content: str) -> ToolResult:
    """Write content to a file, creating parent directories if needed.

    Mirrors packages/coding-agent/src/core/tools/write.ts.
    """
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return ToolResult(
            content=[_text_content(f"File written: {path}")],
            details={"path": str(file_path), "bytes": len(content)},
        )
    except Exception as exc:
        return ToolResult(
            content=[_text_content(f"Error writing file: {exc}")],
            is_error=True,
        )


# --- Edit tool ---


def edit_file(
    path: str | Path,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> ToolResult:
    """Edit a file by replacing old_string with new_string.

    Mirrors packages/coding-agent/src/core/tools/edit.ts.
    """
    file_path = Path(path)
    if not file_path.exists():
        return ToolResult(
            content=[_text_content(f"File not found: {path}")],
            is_error=True,
        )
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        return ToolResult(
            content=[_text_content(f"Error reading file: {exc}")],
            is_error=True,
        )

    count = content.count(old_string)
    if count == 0:
        return ToolResult(
            content=[_text_content(f"old_string not found in {path}")],
            is_error=True,
        )
    if count > 1 and not replace_all:
        return ToolResult(
            content=[_text_content(f"old_string found {count} times in {path}. Use replace_all=True.")],
            is_error=True,
        )

    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

    try:
        file_path.write_text(new_content, encoding="utf-8")
        return ToolResult(
            content=[_text_content(f"File edited: {path} ({count} replacement(s))")],
            details={"path": str(file_path), "replacements": count},
        )
    except Exception as exc:
        return ToolResult(
            content=[_text_content(f"Error writing file: {exc}")],
            is_error=True,
        )


# --- Grep tool ---


def grep_search(
    pattern: str,
    *,
    path: str | Path = ".",
    include: str = "*",
    ignore_case: bool = False,
    max_results: int = 50,
) -> ToolResult:
    """Search file contents with a regex pattern.

    Mirrors packages/coding-agent/src/core/tools/grep.ts.
    """
    search_path = Path(path)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return ToolResult(
            content=[_text_content(f"Invalid regex pattern: {exc}")],
            is_error=True,
        )

    results: list[str] = []
    if search_path.is_file():
        files = [search_path] if search_path.match(include) else []
    else:
        files = sorted(search_path.rglob(include))

    for file_path in files:
        if len(results) >= max_results:
            break
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                results.append(f"{file_path}:{i}:{line}")
                if len(results) >= max_results:
                    break

    output = "\n".join(results) if results else "No matches found"
    return ToolResult(
        content=[_text_content(output)],
        details={"pattern": pattern, "matchCount": len(results), "maxResults": max_results},
    )


# --- List files tool ---


def list_files(
    path: str | Path = ".",
    *,
    max_depth: int = 3,
) -> ToolResult:
    """List files in a directory tree.

    Mirrors packages/coding-agent/src/core/tools/ls.ts.
    """
    search_path = Path(path)
    if not search_path.exists():
        return ToolResult(
            content=[_text_content(f"Directory not found: {path}")],
            is_error=True,
        )

    results: list[str] = []
    for root, dirs, files in os.walk(search_path):
        rel_root = Path(root).relative_to(search_path)
        depth = len(rel_root.parts)
        if depth >= max_depth:
            dirs.clear()
            continue
        for item in sorted(dirs + files):
            display_path = str(rel_root / item) if str(rel_root) != "." else item
            is_dir = (Path(root) / item).is_dir()
            results.append(f"{'d' if is_dir else 'f'}  {display_path}")

    output = "\n".join(results) if results else "(empty)"
    return ToolResult(
        content=[_text_content(output)],
        details={"path": str(search_path), "itemCount": len(results)},
    )
