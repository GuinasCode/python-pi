"""Tests for pi_agent_core.tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pi_agent_core.shell import resolve_posix_shell
from pi_agent_core.tools import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    AgentToolExecutor,
    detect_supported_image_mime_type,
    format_size,
    normalize_tool_path,
    resolve_tool_path,
    truncate_head,
    truncate_tail,
)

requires_posix_shell = pytest.mark.skipif(
    os.name == "nt" and resolve_posix_shell() is None,
    reason="no POSIX-compatible shell (bash/sh) found on PATH",
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestPathUtils:
    def test_normalize_strips_at_prefix(self) -> None:
        assert normalize_tool_path("@/foo/bar") == "/foo/bar"

    def test_normalize_keeps_normal_path(self) -> None:
        assert normalize_tool_path("foo/bar") == "foo/bar"

    def test_normalize_unicode_spaces(self) -> None:
        assert normalize_tool_path("foo\u00a0bar") == "foo bar"

    def test_resolve_relative(self, temp_dir: Path) -> None:
        result = resolve_tool_path("test.txt", cwd=str(temp_dir))
        assert "test.txt" in result

    def test_resolve_absolute(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "test.txt")
        result = resolve_tool_path(abs_path)
        assert result == abs_path


class TestTruncateTail:
    def test_no_truncation_small_content(self) -> None:
        result = truncate_tail("hello\nworld")
        assert not result.truncated
        assert result.content == "hello\nworld"
        assert result.output_lines == 2

    def test_truncate_by_lines(self) -> None:
        content = "\n".join(f"line {i}" for i in range(DEFAULT_MAX_LINES + 10))
        result = truncate_tail(content)
        assert result.truncated
        assert result.output_lines <= DEFAULT_MAX_LINES

    def test_truncate_by_bytes(self) -> None:
        # Create a few very long lines that exceed byte limit
        content = "x" * (DEFAULT_MAX_BYTES + 100)
        result = truncate_tail(content)
        assert result.truncated


class TestTruncateHead:
    def test_no_truncation(self) -> None:
        result = truncate_head("hello")
        assert not result.truncated
        assert result.content == "hello"

    def test_truncate_by_lines(self) -> None:
        content = "\n".join(f"line {i}" for i in range(DEFAULT_MAX_LINES + 10))
        result = truncate_head(content)
        assert result.truncated
        assert result.output_lines <= DEFAULT_MAX_LINES


class TestFormatSize:
    def test_bytes(self) -> None:
        assert format_size(100) == "100B"

    def test_kb(self) -> None:
        assert "KB" in format_size(2048)

    def test_mb(self) -> None:
        assert "MB" in format_size(1024 * 1024 * 2)


class TestImageDetection:
    def test_jpeg(self) -> None:
        data = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"rest"
        assert detect_supported_image_mime_type(data) == "image/jpeg"

    def test_png(self) -> None:
        # Minimal PNG header
        data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        data += b"\x00\x00\x00\x0d" + b"IHDR" + b"\x00" * 20
        result = detect_supported_image_mime_type(data)
        assert result == "image/png"

    def test_gif(self) -> None:
        data = b"GIF89a" + b"rest"
        assert detect_supported_image_mime_type(data) == "image/gif"

    def test_webp(self) -> None:
        data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"rest"
        assert detect_supported_image_mime_type(data) == "image/webp"

    def test_not_image(self) -> None:
        assert detect_supported_image_mime_type(b"plain text") is None


class TestBashTool:
    @pytest.mark.asyncio
    async def test_bash_echo(self) -> None:
        tool = AgentToolExecutor().create_bash_tool()
        result = await tool.execute("t1", {"command": "echo hello"}, None, None)
        assert result is not None
        assert "hello" in result.content[0].text

    @requires_posix_shell
    @pytest.mark.asyncio
    async def test_bash_error(self) -> None:
        tool = AgentToolExecutor().create_bash_tool()
        with pytest.raises(RuntimeError, match="exited with code"):
            await tool.execute("t1", {"command": "false"}, None, None)

    @requires_posix_shell
    @pytest.mark.asyncio
    async def test_bash_no_output(self) -> None:
        tool = AgentToolExecutor().create_bash_tool()
        result = await tool.execute("t1", {"command": "true"}, None, None)
        assert "(no output)" in result.content[0].text

    @pytest.mark.asyncio
    async def test_bash_invalid_timeout(self) -> None:
        tool = AgentToolExecutor().create_bash_tool()
        with pytest.raises(ValueError, match="Invalid timeout"):
            await tool.execute("t1", {"command": "echo", "timeout": -1}, None, None)

    @pytest.mark.asyncio
    async def test_bash_missing_command(self) -> None:
        tool = AgentToolExecutor().create_bash_tool()
        result = await tool.execute("t1", {}, None, None)
        assert "requires a 'command' string argument" in result.content[0].text


class TestReadTool:
    @pytest.mark.asyncio
    async def test_read_text_file(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("line1\nline2\nline3")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_read_tool()
        result = await tool.execute("t1", {"path": "test.txt"}, None, None)
        assert "line1" in result.content[0].text
        assert "line3" in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_not_found(self, temp_dir: Path) -> None:
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_read_tool()
        with pytest.raises(FileNotFoundError):
            await tool.execute("t1", {"path": "nonexistent.txt"}, None, None)

    @pytest.mark.asyncio
    async def test_read_with_offset(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_read_tool()
        result = await tool.execute("t1", {"path": "test.txt", "offset": 2, "limit": 2}, None, None)
        text = result.content[0].text
        assert "line2" in text
        assert "line3" in text

    @pytest.mark.asyncio
    async def test_read_offset_beyond_end(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("one")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_read_tool()
        with pytest.raises(ValueError, match="beyond end of file"):
            await tool.execute("t1", {"path": "test.txt", "offset": 100}, None, None)


class TestWriteTool:
    @pytest.mark.asyncio
    async def test_write_creates_file(self, temp_dir: Path) -> None:
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_write_tool()
        result = await tool.execute("t1", {"path": "new.txt", "content": "hello"}, None, None)
        assert "Successfully wrote" in result.content[0].text
        assert (temp_dir / "new.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, temp_dir: Path) -> None:
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_write_tool()
        await tool.execute("t1", {"path": "sub/dir/new.txt", "content": "data"}, None, None)
        assert (temp_dir / "sub" / "dir" / "new.txt").read_text() == "data"

    @pytest.mark.asyncio
    async def test_write_overwrites(self, temp_dir: Path) -> None:
        f = temp_dir / "exists.txt"
        f.write_text("old")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_write_tool()
        await tool.execute("t1", {"path": "exists.txt", "content": "new"}, None, None)
        assert f.read_text() == "new"


class TestEditTool:
    @pytest.mark.asyncio
    async def test_edit_replace(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("hello world\nfoo bar")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_edit_tool()
        result = await tool.execute(
            "t1",
            {"path": "test.txt", "edits": [{"oldText": "hello world", "newText": "goodbye world"}]},
            None,
            None,
        )
        assert "Successfully replaced" in result.content[0].text
        assert "goodbye world" in f.read_text()

    @pytest.mark.asyncio
    async def test_edit_not_found(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("hello world")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_edit_tool()
        with pytest.raises(ValueError, match="oldText not found"):
            await tool.execute(
                "t1",
                {"path": "test.txt", "edits": [{"oldText": "nonexistent", "newText": "x"}]},
                None,
                None,
            )

    @pytest.mark.asyncio
    async def test_edit_not_unique(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("dup\ndup")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_edit_tool()
        with pytest.raises(ValueError, match="found 2 times"):
            await tool.execute(
                "t1",
                {"path": "test.txt", "edits": [{"oldText": "dup", "newText": "x"}]},
                None,
                None,
            )

    @pytest.mark.asyncio
    async def test_edit_legacy_format(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("hello world")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_edit_tool()
        # Legacy: oldText/newText at top level instead of in edits array
        result = await tool.execute(
            "t1",
            {"path": "test.txt", "oldText": "hello", "newText": "goodbye"},
            None,
            None,
        )
        assert "Successfully replaced" in result.content[0].text
        assert "goodbye world" in f.read_text()

    @pytest.mark.asyncio
    async def test_edit_empty_edits(self, temp_dir: Path) -> None:
        f = temp_dir / "test.txt"
        f.write_text("hello")
        tool = AgentToolExecutor(cwd=str(temp_dir)).create_edit_tool()
        with pytest.raises(ValueError, match="at least one replacement"):
            await tool.execute("t1", {"path": "test.txt", "edits": []}, None, None)


class TestAgentToolExecutor:
    def test_create_all_tools(self) -> None:
        executor = AgentToolExecutor()
        tools = executor.create_all_tools()
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"bash", "read", "write", "edit"}

    def test_create_individual_tools(self) -> None:
        executor = AgentToolExecutor()
        assert executor.create_bash_tool().name == "bash"
        assert executor.create_read_tool().name == "read"
        assert executor.create_write_tool().name == "write"
        assert executor.create_edit_tool().name == "edit"

    def test_tool_parameters_have_required(self) -> None:
        executor = AgentToolExecutor()
        bash = executor.create_bash_tool()
        assert "command" in bash.parameters.get("required", [])

        read = executor.create_read_tool()
        assert "path" in read.parameters.get("required", [])

        write = executor.create_write_tool()
        assert "path" in write.parameters.get("required", [])
        assert "content" in write.parameters.get("required", [])

        edit = executor.create_edit_tool()
        assert "path" in edit.parameters.get("required", [])
        assert "edits" in edit.parameters.get("required", [])
