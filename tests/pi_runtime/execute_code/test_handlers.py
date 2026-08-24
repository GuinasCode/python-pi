"""Tests for pi_runtime.execute_code.handlers — the parent-side RPC
handlers that wrap real pi_coding_agent tool functions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_runtime.execute_code.handlers import (
    DEFAULT_HANDLERS,
    fetch_url_handler,
    list_files_handler,
    read_file_handler,
    search_files_handler,
    terminal_handler,
)
from pi_runtime.execute_code.rpc import RpcError


class TestReadFileHandler:
    def test_reads_a_real_file_through_the_real_tool(self, tmp_path: Path) -> None:
        target = tmp_path / "log.txt"
        target.write_text("line1\nline2\n")

        result = asyncio.run(read_file_handler("read_file", {"path": str(target)}))
        assert "line1" in result
        assert "line2" in result

    def test_missing_path_argument_is_a_malformed_request(self) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(read_file_handler("read_file", {}))
        assert excinfo.value.error_type == "malformed_request"

    def test_nonexistent_file_raises_a_tool_error(self, tmp_path: Path) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(read_file_handler("read_file", {"path": str(tmp_path / "nope.txt")}))
        assert excinfo.value.error_type == "tool_error"

    def test_image_file_raises_a_clear_tool_error_not_garbage(self, tmp_path: Path) -> None:
        image_path = tmp_path / "pixel.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfakebytes")

        with pytest.raises(RpcError) as excinfo:
            asyncio.run(read_file_handler("read_file", {"path": str(image_path)}))
        assert excinfo.value.error_type == "tool_error"
        assert "image" in str(excinfo.value).lower() or "text" in str(excinfo.value).lower()

    def test_offset_and_limit_are_forwarded(self, tmp_path: Path) -> None:
        target = tmp_path / "log.txt"
        target.write_text("\n".join(f"line{i}" for i in range(100)))

        result = asyncio.run(read_file_handler("read_file", {"path": str(target), "offset": 0, "limit": 3}))
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 3


class TestSearchFilesHandler:
    def test_finds_matches_through_the_real_grep_tool(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def foo():\n    pass\n")
        (tmp_path / "b.py").write_text("def bar():\n    pass\n")

        result = asyncio.run(search_files_handler("search_files", {"pattern": "def foo", "path": str(tmp_path)}))
        assert "a.py" in result
        assert "b.py" not in result

    def test_missing_pattern_argument_is_a_malformed_request(self) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(search_files_handler("search_files", {}))
        assert excinfo.value.error_type == "malformed_request"

    def test_invalid_regex_is_a_tool_error(self, tmp_path: Path) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(search_files_handler("search_files", {"pattern": "(unclosed", "path": str(tmp_path)}))
        assert excinfo.value.error_type == "tool_error"


class TestListFilesHandler:
    def test_lists_a_real_directory(self, tmp_path: Path) -> None:
        (tmp_path / "one.txt").write_text("x")
        (tmp_path / "two.txt").write_text("y")

        result = asyncio.run(list_files_handler("list_files", {"path": str(tmp_path)}))
        assert "one.txt" in result
        assert "two.txt" in result

    def test_nonexistent_directory_is_a_tool_error(self, tmp_path: Path) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(list_files_handler("list_files", {"path": str(tmp_path / "nope")}))
        assert excinfo.value.error_type == "tool_error"


class TestTerminalHandler:
    def test_runs_a_real_command_through_the_real_bash_tool(self) -> None:
        result = asyncio.run(terminal_handler("terminal", {"command": "echo hello-from-terminal"}))
        assert "hello-from-terminal" in result

    def test_missing_command_argument_is_a_malformed_request(self) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(terminal_handler("terminal", {}))
        assert excinfo.value.error_type == "malformed_request"

    def test_requested_timeout_is_capped_not_trusted(self) -> None:
        """A child claiming timeout=999999 must not be able to tie up the
        parent's RPC handling for that long — the handler clamps it."""
        from pi_runtime.execute_code.handlers import _MAX_TERMINAL_TIMEOUT_SECONDS

        captured: dict[str, object] = {}

        def _fake_execute_bash(command: str, *, cwd: object = None, timeout: int = 60) -> object:
            captured["timeout"] = timeout
            from pi_coding_agent.tools import ToolResult

            return ToolResult(content=[{"type": "text", "text": "ok"}])

        import pi_runtime.execute_code.handlers as handlers_module

        original = handlers_module.execute_bash
        handlers_module.execute_bash = _fake_execute_bash  # type: ignore[assignment]
        try:
            asyncio.run(terminal_handler("terminal", {"command": "echo hi", "timeout": 999999}))
        finally:
            handlers_module.execute_bash = original  # type: ignore[assignment]

        assert captured["timeout"] == _MAX_TERMINAL_TIMEOUT_SECONDS


class TestFetchUrlHandler:
    def test_missing_url_argument_is_a_malformed_request(self) -> None:
        with pytest.raises(RpcError) as excinfo:
            asyncio.run(fetch_url_handler("fetch_url", {}))
        assert excinfo.value.error_type == "malformed_request"


class TestAllowlistExcludesDangerousTools:
    """Spec section 6/8: execute_code must never be reachable from
    inside a script (no recursion), and delegate_task must never be
    granted automatically (no uncontrolled subagent spawning)."""

    def test_execute_code_is_not_in_the_default_handlers(self) -> None:
        assert "execute_code" not in DEFAULT_HANDLERS

    def test_delegate_task_is_not_in_the_default_handlers(self) -> None:
        assert "delegate_task" not in DEFAULT_HANDLERS
