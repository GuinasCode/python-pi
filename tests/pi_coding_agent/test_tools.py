"""Tests for pi_coding_agent built-in tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from pi_agent_core.shell import resolve_posix_shell
from pi_coding_agent.tools import (
    _PLAYWRIGHT_AVAILABLE,
    _html_to_text,
    browser_fetch_url,
    edit_file,
    execute_bash,
    fetch_url,
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


class TestHtmlToText:
    def test_strips_tags_and_keeps_text(self) -> None:
        markup = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>"
        text = _html_to_text(markup)
        assert "Title" in text
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_skips_script_and_style_content(self) -> None:
        markup = "<html><body><script>evil()</script><style>.x{}</style><p>real content</p></body></html>"
        text = _html_to_text(markup)
        assert "real content" in text
        assert "evil" not in text

    def test_unescapes_html_entities(self) -> None:
        text = _html_to_text("<p>Fish &amp; Chips &mdash; caf&eacute;</p>")
        assert "Fish & Chips" in text
        assert "café" in text


class TestFetchUrl:
    def test_returns_plain_text_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = httpx.Response(200, content=b"hello world", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)
        result = fetch_url("https://example.com")
        assert not result.is_error
        assert "hello world" in result.content[0]["text"]

    def test_strips_html_and_skips_scripts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        markup = b"<html><body><h1>Title</h1><script>evil()</script></body></html>"
        response = httpx.Response(200, content=markup, headers={"content-type": "text/html; charset=utf-8"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)
        result = fetch_url("https://example.com")
        assert not result.is_error
        text = result.content[0]["text"]
        assert "Title" in text
        assert "evil" not in text

    def test_reports_http_error_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = httpx.Response(404, content=b"not found", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)
        result = fetch_url("https://example.com/missing")
        assert result.is_error
        assert "404" in result.content[0]["text"]

    def test_reports_connection_errors_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: Any, **_k: Any) -> httpx.Response:
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "get", _raise)
        result = fetch_url("https://unreachable.example.com")
        assert result.is_error
        assert "boom" in result.content[0]["text"]

    def test_empty_response_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = httpx.Response(200, content=b"", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)
        result = fetch_url("https://example.com/empty")
        assert not result.is_error
        assert "empty response" in result.content[0]["text"]

    def test_truncates_long_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        long_text = "x" * 100_000
        response = httpx.Response(200, content=long_text.encode(), headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)
        result = fetch_url("https://example.com/huge")
        assert not result.is_error
        assert "truncated" in result.content[0]["text"]


class TestBrowserFetchUrl:
    def test_reports_missing_playwright_as_an_error_not_a_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_coding_agent.tools as tools_module

        monkeypatch.setattr(tools_module, "_PLAYWRIGHT_AVAILABLE", False)
        result = browser_fetch_url("https://example.com")
        assert result.is_error
        assert "Playwright" in result.content[0]["text"]
        assert "pip install" in result.content[0]["text"]

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason="requires the optional `browser` extra")
    def test_loads_a_real_page_when_playwright_is_installed(self) -> None:
        result = browser_fetch_url("https://example.com")
        assert not result.is_error
        assert "Example Domain" in result.content[0]["text"]
