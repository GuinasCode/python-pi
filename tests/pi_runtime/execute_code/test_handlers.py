"""Tests for pi_runtime.execute_code.handlers — the parent-side RPC
handlers that wrap real pi_coding_agent tool functions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_runtime.execute_code.handlers import read_file_handler
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
