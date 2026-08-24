"""Tests for pi_runtime.environments. Covers Fase 11's acceptance
criteria from plan.md section 15:

- tools podem executar dentro de um backend
- paths são normalizados
- processos são encerrados (timeout)
- timeouts funcionam
- output é capturado
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_runtime.environments import (
    CommandResult,
    DockerExecutionBackend,
    LocalExecutionBackend,
    SandboxExecutionBackend,
    SshExecutionBackend,
    normalize_path,
)


class TestNormalizePath:
    def test_collapses_dot_segments(self) -> None:
        assert normalize_path("a/./b/../c") == "a/c"

    def test_normalizes_windows_separators(self) -> None:
        assert normalize_path("a\\b\\c") == "a/b/c"

    def test_leaves_a_clean_absolute_path_alone(self) -> None:
        assert normalize_path("/usr/local/bin") == "/usr/local/bin"


class TestLocalExecutionBackendRunsCommands:
    def test_command_output_is_captured(self, tmp_path: Path) -> None:
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        result = backend.run("echo hello")
        assert result.ok
        assert "hello" in result.stdout

    def test_nonzero_exit_is_reflected(self, tmp_path: Path) -> None:
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        result = backend.run("exit 1")
        assert not result.ok
        assert result.exit_code != 0

    def test_cwd_is_reported(self, tmp_path: Path) -> None:
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        assert backend.cwd() == str(tmp_path)


class TestLocalExecutionBackendTimeout:
    @pytest.mark.skipif(__import__("os").name == "nt", reason="requires a POSIX sleep-compatible shell")
    def test_timeout_is_enforced_and_reported(self, tmp_path: Path) -> None:
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        result = backend.run("sleep 5", timeout=0.5)
        assert result.timed_out
        assert not result.ok


class TestLocalExecutionBackendFiles:
    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        """read_file() goes through pi_coding_agent.tools.read_file,
        unchanged — which formats output with line numbers by design
        (see its own tests in tests/pi_coding_agent/test_tools.py), so
        the round trip is checked by containment, not exact equality."""
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        target = str(tmp_path / "file.txt")
        backend.write_file(target, "hello world")
        assert "hello world" in backend.read_file(target)

    def test_reading_a_missing_file_raises(self, tmp_path: Path) -> None:
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            backend.read_file(str(tmp_path / "nope.txt"))

    def test_tools_run_through_a_backend_not_bypassing_it(self, tmp_path: Path) -> None:
        """Fase 11 acceptance criterion: "tools podem executar dentro de
        um backend" — proves the read/write actually goes through the
        backend's own file, not some other path."""
        backend = LocalExecutionBackend(_cwd=str(tmp_path))
        target = tmp_path / "via_backend.txt"
        backend.write_file(str(target), "written via backend")
        assert target.read_text() == "written via backend"


class TestDockerExecutionBackendWhenUnavailable:
    """Assumes `docker` isn't necessarily on PATH in the test
    environment — these confirm the explicit-unavailable path, not real
    container execution (that would need a real Docker daemon, out of
    scope for a unit test)."""

    def test_missing_binary_is_reported_not_crashed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = DockerExecutionBackend(container="test-container")
        monkeypatch.setattr(backend, "_binary_available", lambda: False)
        result = backend.run("echo hi")
        assert not result.ok
        assert "docker" in result.stderr.lower()

    def test_read_file_raises_when_backend_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = DockerExecutionBackend(container="test-container")
        monkeypatch.setattr(backend, "_binary_available", lambda: False)
        with pytest.raises(FileNotFoundError):
            backend.read_file("/some/path")


class TestSshExecutionBackendWhenUnavailable:
    def test_missing_binary_is_reported_not_crashed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = SshExecutionBackend(host="example.com")
        monkeypatch.setattr(backend, "_binary_available", lambda: False)
        result = backend.run("echo hi")
        assert not result.ok
        assert "ssh" in result.stderr.lower()


class TestSandboxIsAnExplicitTodoNotFaked:
    def test_instantiation_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            SandboxExecutionBackend()


class TestCommandResult:
    def test_ok_requires_zero_exit_and_no_timeout(self) -> None:
        assert CommandResult(stdout="", stderr="", exit_code=0).ok
        assert not CommandResult(stdout="", stderr="", exit_code=1).ok
        assert not CommandResult(stdout="", stderr="", exit_code=0, timed_out=True).ok
