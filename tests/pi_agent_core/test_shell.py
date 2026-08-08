"""Tests for POSIX shell resolution on Windows."""

from __future__ import annotations

import pytest

from pi_agent_core.shell import _is_wsl_launcher_stub, resolve_posix_shell


class TestIsWslLauncherStub:
    def test_flags_system32_bash(self) -> None:
        assert _is_wsl_launcher_stub(r"C:\Windows\System32\bash.exe", r"C:\Windows")

    def test_flags_syswow64_bash(self) -> None:
        assert _is_wsl_launcher_stub(r"C:\Windows\SysWOW64\bash.exe", r"C:\Windows")

    def test_does_not_flag_git_bash(self) -> None:
        assert not _is_wsl_launcher_stub(r"C:\Program Files\Git\bin\bash.exe", r"C:\Windows")

    def test_does_not_flag_git_usr_bin_bash(self) -> None:
        assert not _is_wsl_launcher_stub(r"C:\Program Files\Git\usr\bin\bash.exe", r"C:\Windows")


class TestResolvePosixShell:
    def test_skips_wsl_stub_even_when_first_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """shutil.which("bash") returning the WSL launcher stub must not be used."""
        resolve_posix_shell.cache_clear()
        monkeypatch.setattr("pi_agent_core.shell.os.name", "nt")
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
        monkeypatch.delenv("ProgramFiles", raising=False)
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("ProgramW6432", raising=False)

        def fake_which(name: str) -> str | None:
            return r"C:\Windows\System32\bash.exe" if name == "bash" else None

        monkeypatch.setattr("pi_agent_core.shell.shutil.which", fake_which)
        monkeypatch.setattr("pi_agent_core.shell.os.path.isfile", lambda _p: True)

        try:
            assert resolve_posix_shell() is None
        finally:
            resolve_posix_shell.cache_clear()

    def test_falls_back_to_git_install_dir_when_not_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolve_posix_shell.cache_clear()
        monkeypatch.setattr("pi_agent_core.shell.os.name", "nt")
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
        monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("ProgramW6432", raising=False)

        monkeypatch.setattr("pi_agent_core.shell.shutil.which", lambda _name: None)
        expected = r"C:\Program Files\Git\bin\bash.exe"
        monkeypatch.setattr("pi_agent_core.shell.os.path.isfile", lambda p: p == expected)

        try:
            assert resolve_posix_shell() == expected
        finally:
            resolve_posix_shell.cache_clear()
