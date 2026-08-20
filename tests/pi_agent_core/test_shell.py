"""Tests for POSIX shell resolution on Windows."""

from __future__ import annotations

import ntpath
import os as real_os
from collections.abc import Callable

import pytest

from pi_agent_core.shell import _is_wsl_launcher_stub, resolve_posix_shell


class _FakePath:
    """A stand-in for ``os.path`` while ``os.name`` is faked as "nt".

    Delegates to ``ntpath`` (real Windows path semantics: backslash joins,
    drive letters, ...) rather than the host's actual ``os.path`` module —
    which is what production code gets for free when ``os.name`` really is
    "nt", but isn't true here since the host running this test may well be
    Linux. ``isfile`` is overridden per-test since it needs to be faked
    regardless of platform.
    """

    def __init__(self, isfile: Callable[[str], bool]) -> None:
        self.isfile = isfile

    def __getattr__(self, attr: str) -> object:
        return getattr(ntpath, attr)


class _FakeOs:
    """Proxies the real ``os`` module except ``.name`` (and its own ``.path``).

    ``monkeypatch.setattr("pi_agent_core.shell.os.name", "nt")`` looks
    scoped, but ``os`` is a singleton module — that call actually mutates
    the real, process-wide ``os.name`` for the test's duration. That's
    dangerous here specifically: ``pathlib.Path()`` reads ``os.name`` on
    every call to decide whether to build a ``WindowsPath`` or
    ``PosixPath``, so forcing it to "nt" mid-test risks any concurrent
    pytest internals that construct a ``Path`` during that window ending up
    with a real ``WindowsPath`` baked into long-lived state — which then
    blows up later on a non-Windows host. Rebinding the ``os`` name *within
    shell.py's module namespace only* (via this proxy) avoids that
    entirely: everything but ``.name``/``.path.isfile`` still delegates to
    the real module.
    """

    def __init__(self, name: str, isfile: Callable[[str], bool] | None = None) -> None:
        self.name = name
        self.path = _FakePath(isfile or real_os.path.isfile)

    def __getattr__(self, attr: str) -> object:
        return getattr(real_os, attr)


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
        monkeypatch.setattr("pi_agent_core.shell.os", _FakeOs("nt", isfile=lambda _p: True))
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
        monkeypatch.delenv("ProgramFiles", raising=False)
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("ProgramW6432", raising=False)

        def fake_which(name: str) -> str | None:
            return r"C:\Windows\System32\bash.exe" if name == "bash" else None

        monkeypatch.setattr("pi_agent_core.shell.shutil.which", fake_which)

        try:
            assert resolve_posix_shell() is None
        finally:
            resolve_posix_shell.cache_clear()

    def test_falls_back_to_git_install_dir_when_not_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolve_posix_shell.cache_clear()
        expected = r"C:\Program Files\Git\bin\bash.exe"
        monkeypatch.setattr("pi_agent_core.shell.os", _FakeOs("nt", isfile=lambda p: p == expected))
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
        monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.delenv("ProgramW6432", raising=False)

        monkeypatch.setattr("pi_agent_core.shell.shutil.which", lambda _name: None)

        try:
            assert resolve_posix_shell() == expected
        finally:
            resolve_posix_shell.cache_clear()
