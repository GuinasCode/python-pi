"""Tests for pi_coding_agent.git_info."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

from pi_coding_agent.git_info import get_git_repo_line


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestGetGitRepoLine:
    def test_returns_repo_and_branch(self) -> None:
        def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "--show-toplevel" in cmd:
                return _completed("/home/user/python-pi\n")
            if "--show-current" in cmd:
                return _completed("main\n")
            raise AssertionError(f"unexpected git command: {cmd}")

        with patch("subprocess.run", side_effect=fake_run):
            assert get_git_repo_line("/home/user/python-pi") == "(python-pi:main)"

    def test_returns_none_outside_a_repo(self) -> None:
        with patch("subprocess.run", return_value=_completed("", returncode=128)):
            assert get_git_repo_line("/tmp/not-a-repo") is None

    def test_returns_none_when_git_not_installed(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert get_git_repo_line("/tmp/whatever") is None

    def test_detached_head_falls_back_to_short_sha(self) -> None:
        def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "--show-toplevel" in cmd:
                return _completed("/home/user/python-pi\n")
            if "--show-current" in cmd:
                return _completed("\n")
            if "--short" in cmd:
                return _completed("abc1234\n")
            raise AssertionError(f"unexpected git command: {cmd}")

        with patch("subprocess.run", side_effect=fake_run):
            assert get_git_repo_line("/home/user/python-pi") == "(python-pi:detached:abc1234)"
