"""Tests for pi_coding_agent built-in tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pi_agent_core.shell import resolve_posix_shell
from pi_coding_agent.tools import (
    edit_file,
    execute_bash,
    grep_search,
    list_files,
    read_file,
    write_file,
)

requires_posix_shell = pytest.mark.skipif(
    os.name == "nt" and resolve_posix_shell() is None,
    reason="no POSIX-compatible shell (bash/sh) found on PATH",
)


class TestExecuteBash:
    def test_echo_command(self) -> None:
        result = execute_bash("echo hello")
        assert not result.is_error
        assert "hello" in result.content[0]["text"]

    def test_failing_command(self) -> None:
        result = execute_bash("exit 1")
        assert result.is_error

    @requires_posix_shell
    def test_timeout(self) -> None:
        result = execute_bash("sleep 10", timeout=1)
        assert result.is_error
        assert "timed out" in result.content[0]["text"]


class TestReadFile:
    def test_read_existing_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("line1\nline2\nline3\n")
        result = read_file(file_path)
        assert not result.is_error
        assert "line1" in result.content[0]["text"]
        assert "line3" in result.content[0]["text"]

    def test_read_nonexistent_file(self) -> None:
        result = read_file("/nonexistent/file.txt")
        assert result.is_error
        assert "not found" in result.content[0]["text"]

    def test_read_with_offset(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("a\nb\nc\nd\n")
        result = read_file(file_path, offset=2)
        assert not result.is_error
        text = result.content[0]["text"]
        assert "c" in text
        assert "a" not in text.split("\n")[0]

    def test_read_with_limit(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("\n".join(f"line{i}" for i in range(100)))
        result = read_file(file_path, limit=5)
        assert not result.is_error
        lines = result.content[0]["text"].split("\n")
        assert len(lines) == 5


class TestWriteFile:
    def test_write_new_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "new.txt"
        result = write_file(file_path, "hello world")
        assert not result.is_error
        assert file_path.read_text() == "hello world"

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        file_path = tmp_path / "sub" / "dir" / "file.txt"
        result = write_file(file_path, "content")
        assert not result.is_error
        assert file_path.read_text() == "content"


class TestEditFile:
    def test_edit_replace(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")
        result = edit_file(file_path, "hello", "goodbye")
        assert not result.is_error
        assert file_path.read_text() == "goodbye world"

    def test_edit_not_found(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello")
        result = edit_file(file_path, "nonexistent", "replacement")
        assert result.is_error

    def test_edit_multiple_without_replace_all(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("a\na\na\n")
        result = edit_file(file_path, "a", "b")
        assert result.is_error

    def test_edit_multiple_with_replace_all(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("a\na\na\n")
        result = edit_file(file_path, "a", "b", replace_all=True)
        assert not result.is_error
        assert file_path.read_text() == "b\nb\nb\n"

    def test_edit_nonexistent_file(self) -> None:
        result = edit_file("/nonexistent/file.txt", "old", "new")
        assert result.is_error


class TestGrepSearch:
    def test_grep_finds_matches(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.py"
        file_path.write_text("def hello():\n    pass\ndef world():\n    pass\n")
        result = grep_search("def", path=tmp_path, include="*.py")
        assert not result.is_error
        text = result.content[0]["text"]
        assert "hello" in text
        assert "world" in text

    def test_grep_no_matches(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")
        result = grep_search("nonexistent_pattern", path=tmp_path)
        assert not result.is_error
        assert "No matches" in result.content[0]["text"]

    def test_grep_invalid_regex(self) -> None:
        result = grep_search("[invalid")
        assert result.is_error
        assert "Invalid regex" in result.content[0]["text"]

    def test_grep_ignore_case(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("HELLO\nWorld\n")
        result = grep_search("hello", path=tmp_path, ignore_case=True)
        assert not result.is_error
        assert "HELLO" in result.content[0]["text"]


class TestListFiles:
    def test_list_files_in_directory(self, tmp_path: Path) -> None:
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        (tmp_path / "subdir").mkdir()
        result = list_files(tmp_path)
        assert not result.is_error
        text = result.content[0]["text"]
        assert "file1.txt" in text
        assert "file2.py" in text
        assert "subdir" in text

    def test_list_nonexistent_directory(self) -> None:
        result = list_files("/nonexistent/path")
        assert result.is_error
