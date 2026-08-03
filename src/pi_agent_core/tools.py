"""Built-in tool executors for the agent harness.

Mirrors ``packages/agent/src/harness/tools/`` from the TypeScript Pi agent.

This module provides the ``AgentToolExecutor`` — a helper that creates
:class:`pi_agent_core.AgentTool` instances for the built-in bash, read, write,
and edit tools — plus shared path utilities and tool result creation helpers.

The tool executors use Python's standard library (``subprocess``, ``pathlib``,
``json``) rather than a pluggable ``ExecutionEnv`` abstraction. Tests can
substitute their own ``AgentTool.execute`` callbacks.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_ai import ImageContent, TextContent

from .types import AgentTool, AgentToolResult, AgentToolUpdateCallback

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_LINES",
    "MAX_TIMEOUT_SECONDS",
    "BashToolDetails",
    "EditToolDetails",
    "ReadToolDetails",
    "TruncationResult",
    "AgentToolExecutor",
    "create_bash_tool",
    "create_edit_tool",
    "create_read_tool",
    "create_write_tool",
    "detect_supported_image_mime_type",
    "format_size",
    "normalize_tool_path",
    "resolve_tool_path",
    "text_result",
    "truncate_head",
    "truncate_tail",
]

# --- Constants ---

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 256_000  # 256 KB
MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000

_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202F"


# --- Path utilities (path-utils.ts) ---


def normalize_tool_path(path: str) -> str:
    """Normalize a tool path: strip leading ``@`` and normalize unicode spaces."""
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


def resolve_tool_path(path: str, *, cwd: str | None = None) -> str:
    """Resolve a tool path to an absolute path."""
    normalized = normalize_tool_path(path)
    p = Path(normalized)
    if p.is_absolute():
        return str(p)
    base = Path(cwd or os.getcwd())
    return str((base / p).resolve())


def resolve_read_tool_path(path: str, *, cwd: str | None = None) -> str:
    """Resolve a read tool path, trying common path variants for case/space differences."""
    resolved = resolve_tool_path(path, cwd=cwd)
    variants = [
        resolved,
        re.sub(r" (AM|PM)\.", f"{_NARROW_NO_BREAK_SPACE}\\1.", resolved, flags=re.IGNORECASE),
    ]
    # NFD-normalized variant
    import unicodedata

    variants.append(unicodedata.normalize("NFD", resolved))
    # Right single quote variant
    variants.append(resolved.replace("'", "\u2019"))
    variants.append(unicodedata.normalize("NFD", resolved).replace("'", "\u2019"))

    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        if Path(variant).exists():
            return variant
    return resolved


# --- Truncation utilities ---


@dataclass
class TruncationResult:
    """Result of truncating output to a line/byte budget."""

    content: str = ""
    truncated: bool = False
    truncated_by: str = ""  # "lines" | "bytes" | ""
    output_lines: int = 0
    output_bytes: int = 0
    total_lines: int = 0
    total_bytes: int = 0
    last_line_partial: bool = False


def format_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable size string."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def _truncate_lines(content: str, max_lines: int) -> tuple[str, int, bool]:
    """Truncate to the last ``max_lines`` lines. Returns (content, line_count, truncated)."""
    lines = content.split("\n")
    if len(lines) <= max_lines:
        return content, len(lines), False
    kept = lines[-max_lines:]
    return "\n".join(kept), max_lines, True


def truncate_tail(content: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate output to the last ``max_lines`` or ``max_bytes``, whichever is hit first."""
    total_lines = content.count("\n") + 1 if content else 0
    total_bytes = len(content.encode("utf-8"))

    if total_bytes <= max_bytes and total_lines <= max_lines:
        return TruncationResult(
            content=content,
            truncated=False,
            output_lines=total_lines,
            output_bytes=total_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    # Check if first line exceeds byte limit
    first_line = content.split("\n", 1)[0] if content else ""
    if len(first_line.encode("utf-8")) > max_bytes:
        return TruncationResult(
            content=content,
            truncated=True,
            truncated_by="bytes",
            output_lines=0,
            output_bytes=0,
            total_lines=total_lines,
            total_bytes=total_bytes,
            last_line_partial=True,
        )

    # Truncate by lines first
    line_truncated, output_lines, lines_truncated = _truncate_lines(content, max_lines)
    output_bytes = len(line_truncated.encode("utf-8"))

    if output_bytes <= max_bytes:
        return TruncationResult(
            content=line_truncated,
            truncated=True,
            truncated_by="lines" if lines_truncated else "",
            output_lines=output_lines,
            output_bytes=output_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    # Further truncate by bytes
    encoded = line_truncated.encode("utf-8")
    truncated_bytes = encoded[-max_bytes:]
    try:
        truncated_text = truncated_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Find a valid UTF-8 boundary
        for i in range(1, 4):
            try:
                truncated_text = truncated_bytes[i:].decode("utf-8")
                break
            except UnicodeDecodeError:
                continue
        else:
            truncated_text = truncated_bytes.decode("utf-8", errors="replace")

    return TruncationResult(
        content=truncated_text,
        truncated=True,
        truncated_by="bytes",
        output_lines=output_lines,
        output_bytes=max_bytes,
        total_lines=total_lines,
        total_bytes=total_bytes,
    )


def truncate_head(content: str, *, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    """Truncate output to the first ``max_lines`` or ``max_bytes``, whichever is hit first."""
    total_lines = content.count("\n") + 1 if content else 0
    total_bytes = len(content.encode("utf-8"))

    if total_bytes <= max_bytes and total_lines <= max_lines:
        return TruncationResult(
            content=content,
            truncated=False,
            output_lines=total_lines,
            output_bytes=total_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    # Check if first line exceeds byte limit
    first_line = content.split("\n", 1)[0] if content else ""
    if len(first_line.encode("utf-8")) > max_bytes:
        return TruncationResult(
            content=content,
            truncated=True,
            truncated_by="bytes",
            output_lines=0,
            output_bytes=0,
            total_lines=total_lines,
            total_bytes=total_bytes,
            last_line_partial=True,
        )

    # Truncate by lines first (take first max_lines)
    lines = content.split("\n")
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        line_truncated = "\n".join(kept)
        truncated_by = "lines"
        output_lines = max_lines
    else:
        line_truncated = content
        truncated_by = ""
        output_lines = len(lines)

    output_bytes = len(line_truncated.encode("utf-8"))

    if output_bytes <= max_bytes:
        return TruncationResult(
            content=line_truncated,
            truncated=bool(truncated_by),
            truncated_by=truncated_by,
            output_lines=output_lines,
            output_bytes=output_bytes,
            total_lines=total_lines,
            total_bytes=total_bytes,
        )

    # Further truncate by bytes
    encoded = line_truncated.encode("utf-8")
    truncated_bytes = encoded[:max_bytes]
    try:
        truncated_text = truncated_bytes.decode("utf-8")
    except UnicodeDecodeError:
        truncated_text = truncated_bytes.decode("utf-8", errors="replace")

    return TruncationResult(
        content=truncated_text,
        truncated=True,
        truncated_by="bytes",
        output_lines=output_lines,
        output_bytes=max_bytes,
        total_lines=total_lines,
        total_bytes=total_bytes,
    )


# --- Tool result helpers ---


def text_result(text: str, *, details: Any = None, terminate: bool | None = None) -> AgentToolResult:
    """Create a simple text :class:`AgentToolResult`."""
    return AgentToolResult(content=[TextContent(text=text)], details=details, terminate=terminate)


def error_result(message: str) -> AgentToolResult:
    """Create an error :class:`AgentToolResult`."""
    return AgentToolResult(content=[TextContent(text=message)])


# --- Tool details types ---


@dataclass
class BashToolDetails:
    """Details for the bash tool."""

    truncation: TruncationResult | None = None
    full_output_path: str | None = None


@dataclass
class ReadToolDetails:
    """Details for the read tool."""

    truncation: TruncationResult | None = None


@dataclass
class Edit:
    """A single edit replacement."""

    old_text: str
    new_text: str


@dataclass
class EditToolDetails:
    """Details for the edit tool."""

    diff: str = ""
    patch: str = ""
    first_changed_line: int | None = None


# --- Image detection (image.ts) ---

_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def detect_supported_image_mime_type(data: bytes) -> str | None:
    """Detect supported image MIME type from file content bytes."""
    if len(data) >= 4 and data[:3] == bytes([0xFF, 0xD8, 0xFF]):
        if data[3] == 0xF7:
            return None
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == _PNG_SIGNATURE:
        return "image/png" if _is_png(data) and not _is_animated_png(data) else None
    if len(data) >= 3 and data[:3] == b"GIF":
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 2 and data[:2] == b"BM" and _is_bmp(data):
        return "image/bmp"
    return None


def _is_png(data: bytes) -> bool:
    return len(data) >= 16 and _read_uint32_be(data, len(_PNG_SIGNATURE)) == 13 and data[12:16] == b"IHDR"


def _is_animated_png(data: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        chunk_length = _read_uint32_be(data, offset)
        chunk_type_offset = offset + 4
        if data[chunk_type_offset : chunk_type_offset + 4] == b"acTL":
            return True
        if data[chunk_type_offset : chunk_type_offset + 4] == b"IDAT":
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(data):
            return False
        offset = next_offset
    return False


def _is_bmp(data: bytes) -> bool:
    if len(data) < 26:
        return False
    declared_file_size = _read_uint32_le(data, 2)
    pixel_data_offset = _read_uint32_le(data, 10)
    dib_header_size = _read_uint32_le(data, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(data, 22)
        bits_per_pixel = _read_uint16_le(data, 24)
    elif 40 <= dib_header_size <= 124:
        if len(data) < 30:
            return False
        color_planes = _read_uint16_le(data, 26)
        bits_per_pixel = _read_uint16_le(data, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _read_uint16_le(data: bytes, offset: int) -> int:
    return (data[offset] if offset < len(data) else 0) + (
        (data[offset + 1] if offset + 1 < len(data) else 0) << 8
    )


def _read_uint32_be(data: bytes, offset: int) -> int:
    return (
        (data[offset] if offset < len(data) else 0) * 0x1000000
        + ((data[offset + 1] if offset + 1 < len(data) else 0) << 16)
        + ((data[offset + 2] if offset + 2 < len(data) else 0) << 8)
        + (data[offset + 3] if offset + 3 < len(data) else 0)
    )


def _read_uint32_le(data: bytes, offset: int) -> int:
    return (
        (data[offset] if offset < len(data) else 0)
        + ((data[offset + 1] if offset + 1 < len(data) else 0) << 8)
        + ((data[offset + 2] if offset + 2 < len(data) else 0) << 16)
        + ((data[offset + 3] if offset + 3 < len(data) else 0) * 0x1000000)
    )


# --- Tool executors ---


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, int | float) or timeout <= 0 or timeout != timeout:  # NaN check
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- Bash tool ---


def create_bash_tool(
    *,
    command_prefix: str | None = None,
    cwd: str | None = None,
    timeout: int = 120,
) -> AgentTool:
    """Create the bash tool.

    Executes a bash command and returns stdout/stderr, truncated to
    ``DEFAULT_MAX_LINES`` / ``DEFAULT_MAX_BYTES``.
    """
    bash_description = (
        f"Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        f"(whichever is hit first). Optionally provide a timeout in seconds."
    )

    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        if not isinstance(args, dict):
            return error_result("bash tool requires arguments object")
        command = args.get("command", "")
        user_timeout = args.get("timeout")
        if user_timeout is not None:
            _validate_timeout(user_timeout)
            exec_timeout = int(user_timeout)
        else:
            exec_timeout = timeout

        full_command = f"{command_prefix}\n{command}" if command_prefix else command
        work_dir = cwd or os.getcwd()

        try:
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=exec_timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            msg = f"Command timed out after {exec_timeout} seconds"
            if output:
                msg = f"{output}\n\n{msg}"
            raise RuntimeError(msg) from exc
        except Exception as exc:
            raise RuntimeError(f"Error executing command: {exc}") from exc

        output_text = result.stdout
        if result.stderr:
            if output_text:
                output_text += f"\nSTDERR:\n{result.stderr}"
            else:
                output_text = result.stderr

        output_text = output_text.strip() if output_text else ""

        truncation = truncate_tail(output_text)
        details: BashToolDetails | None = None
        if truncation.truncated:
            details = BashToolDetails(truncation=truncation)
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                output_text = (
                    truncation.content
                    + f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output saved.]"
                )
            elif truncation.truncated_by == "lines":
                output_text = (
                    truncation.content
                    + f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}.]"
                )
            else:
                output_text = (
                    truncation.content
                    + f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit).]"
                )
        elif not output_text:
            output_text = "(no output)"

        if result.returncode != 0:
            raise RuntimeError(f"Command exited with code {result.returncode}\n{output_text}")

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentTool(
        name="bash",
        label="bash",
        description=bash_description,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
            },
            "required": ["command"],
        },
        execute=execute,
    )


# --- Read tool ---


def create_read_tool(*, cwd: str | None = None) -> AgentTool:
    """Create the read tool."""
    description = (
        f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
        f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB. "
        "Use offset/limit for large files."
    )

    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        if not isinstance(args, dict):
            return error_result("read tool requires arguments object")
        path = args.get("path", "")
        offset = args.get("offset")
        limit = args.get("limit")

        absolute_path = resolve_read_tool_path(path, cwd=cwd)
        file_path = Path(absolute_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Read as binary for image detection
        raw_bytes = file_path.read_bytes()
        mime_type = detect_supported_image_mime_type(raw_bytes)
        if mime_type is not None:
            import base64

            if mime_type == "image/bmp":
                return text_result(
                    f"Read image file [image/bmp]\n[Image omitted: BMP requires conversion.]"
                )
            encoded = base64.b64encode(raw_bytes).decode("ascii")
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{mime_type}]"),
                    ImageContent(data=encoded, mime_type=mime_type),
                ]
            )

        # Text file
        text_content = raw_bytes.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (offset - 1)) if offset else 0
        start_line_display = start_line + 1

        if start_line >= total_file_lines:
            raise ValueError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")

        if limit is not None:
            end_line = min(start_line + limit, total_file_lines)
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None

        if truncation.last_line_partial:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_BYTES)} limit.]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                    f"{total_file_lines}. Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                    f"{total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
            remaining = total_file_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output_text = truncation.content

        return AgentToolResult(content=[TextContent(text=output_text)], details=details)

    return AgentTool(
        name="read",
        label="read",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "number", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        execute=execute,
    )


# --- Write tool ---


def create_write_tool(*, cwd: str | None = None) -> AgentTool:
    """Create the write tool."""
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )

    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        if not isinstance(args, dict):
            return error_result("write tool requires arguments object")
        path = args.get("path", "")
        content = args.get("content", "")

        absolute_path = resolve_tool_path(path, cwd=cwd)
        file_path = Path(absolute_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        return text_result(f"Successfully wrote {len(content)} bytes to {path}")

    return AgentTool(
        name="write",
        label="write",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        execute=execute,
    )


# --- Edit tool ---


def _prepare_edit_arguments(input: Any) -> Any:
    """Normalize edit tool arguments, handling legacy oldText/newText format."""
    if not isinstance(input, dict):
        return input

    args = dict(input)
    # Handle edits passed as JSON string
    if isinstance(args.get("edits"), str):
        try:
            import json

            parsed = json.loads(args["edits"])
            if isinstance(parsed, list):
                args["edits"] = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # Legacy: oldText/newText at top level
    old_text = args.get("oldText")
    new_text = args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = list(args.get("edits", []))
        edits.append({"oldText": old_text, "newText": new_text})
        args.pop("oldText", None)
        args.pop("newText", None)
        args["edits"] = edits

    return args


def _validate_edit_input(input: Any) -> tuple[str, list[Edit]]:
    if not isinstance(input, dict):
        raise ValueError("Edit tool input must be an object")
    edits_raw = input.get("edits")
    if not isinstance(edits_raw, list) or len(edits_raw) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    edits: list[Edit] = []
    for e in edits_raw:
        if not isinstance(e, dict):
            raise ValueError("Each edit must be an object with oldText and newText")
        edits.append(Edit(old_text=e["oldText"], new_text=e["newText"]))
    return input["path"], edits


def _apply_edits(content: str, edits: list[Edit], path: str) -> tuple[str, str]:
    """Apply edits to content. Returns (base_content, new_content)."""
    base = content
    for edit in edits:
        count = base.count(edit.old_text)
        if count == 0:
            raise ValueError(f"oldText not found in {path}")
        if count > 1:
            raise ValueError(f"oldText found {count} times in {path}. Must be unique.")
        base = base.replace(edit.old_text, edit.new_text, 1)
    return content, base


def _generate_diff(base: str, new: str) -> tuple[str, int | None]:
    """Generate a simple diff string and first changed line number."""
    import difflib

    diff_lines = list(difflib.unified_diff(base.splitlines(keepends=True), new.splitlines(keepends=True), n=3))
    diff_text = "".join(diff_lines)

    first_changed: int | None = None
    base_lines = base.splitlines()
    new_lines = new.splitlines()
    for i, (b, n) in enumerate(zip(base_lines, new_lines, strict=False)):
        if b != n:
            first_changed = i + 1
            break
    if first_changed is None and len(base_lines) != len(new_lines):
        first_changed = min(len(base_lines), len(new_lines)) + 1

    return diff_text, first_changed


def _generate_patch(path: str, base: str, new: str) -> str:
    """Generate a unified diff patch string."""
    import difflib

    diff = difflib.unified_diff(
        base.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


def create_edit_tool(*, cwd: str | None = None) -> AgentTool:
    """Create the edit tool."""
    description = (
        "Edit a single file using exact text replacement. Every edits[].oldText must match a "
        "unique, non-overlapping region of the original file."
    )

    async def execute(
        tool_call_id: str,
        args: Any,
        signal: Any | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        prepared = _prepare_edit_arguments(args)
        path, edits = _validate_edit_input(prepared)

        absolute_path = resolve_tool_path(path, cwd=cwd)
        file_path = Path(absolute_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        content = file_path.read_text(encoding="utf-8")
        base_content, new_content = _apply_edits(content, edits, path)

        file_path.write_text(new_content, encoding="utf-8")

        diff_text, first_changed = _generate_diff(base_content, new_content)
        patch = _generate_patch(path, base_content, new_content)

        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
            details=EditToolDetails(diff=diff_text, patch=patch, first_changed_line=first_changed),
        )

    return AgentTool(
        name="edit",
        label="edit",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string", "description": "Exact text for one targeted replacement."},
                            "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                        },
                        "required": ["oldText", "newText"],
                    },
                    "description": "One or more targeted replacements.",
                },
            },
            "required": ["path", "edits"],
        },
        prepare_arguments=_prepare_edit_arguments,
        execute=execute,
    )


# --- AgentToolExecutor ---


@dataclass
class AgentToolExecutor:
    """Creates and manages the set of built-in agent tools.

    Mirrors the tool factory pattern from ``tools/index.ts``. Hosts can use this
    to create a default set of tools or cherry-pick individual factories.
    """

    cwd: str = field(default_factory=os.getcwd)
    bash_timeout: int = 120
    command_prefix: str | None = None

    def create_bash_tool(self) -> AgentTool:
        return create_bash_tool(
            command_prefix=self.command_prefix,
            cwd=self.cwd,
            timeout=self.bash_timeout,
        )

    def create_read_tool(self) -> AgentTool:
        return create_read_tool(cwd=self.cwd)

    def create_write_tool(self) -> AgentTool:
        return create_write_tool(cwd=self.cwd)

    def create_edit_tool(self) -> AgentTool:
        return create_edit_tool(cwd=self.cwd)

    def create_all_tools(self) -> list[AgentTool]:
        """Create all built-in tools: bash, read, write, edit."""
        return [
            self.create_bash_tool(),
            self.create_read_tool(),
            self.create_write_tool(),
            self.create_edit_tool(),
        ]
