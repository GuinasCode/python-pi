"""Tests for pi_runtime.execute_code.runner.CodeExecutor — Slice A1.

Covers: código simples, código inválido, exit code, stderr, timeout,
cancellation, output bounded (spec section 19 "Core" + "Output").
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_runtime.execute_code.result import ExecutionStatus
from pi_runtime.execute_code.runner import CodeExecutor


def _executor(tmp_path: Path) -> CodeExecutor:
    return CodeExecutor(artifacts_root=tmp_path)


class TestSimpleCode:
    def test_prints_are_captured(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print('hello world')"))
        assert result.ok
        assert result.status == ExecutionStatus.SUCCESS
        assert "hello world" in result.stdout.preview
        assert result.exit_code == 0

    def test_duration_is_recorded(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("pass"))
        assert result.duration_ms >= 0

    def test_artifacts_dir_contains_the_script(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print(1)"))
        assert result.artifacts_dir is not None
        assert (Path(result.artifacts_dir) / "script.py").read_text() == "print(1)"


class TestInvalidCode:
    def test_syntax_error_is_classified_not_run(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("def broken(:\n"))
        assert result.status == ExecutionStatus.INVALID_CODE
        assert result.exit_code is None
        assert result.error_message

    def test_invalid_code_never_spawns_a_process(self, tmp_path: Path) -> None:
        """No stdout.log/stderr.log should exist — the subprocess step
        is never reached."""
        result = asyncio.run(_executor(tmp_path).execute("this is not python !!"))
        assert result.status == ExecutionStatus.INVALID_CODE
        assert not (Path(result.artifacts_dir) / "stdout.log").exists()  # type: ignore[arg-type]


class TestExitCode:
    def test_nonzero_exit_is_classified(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("import sys; sys.exit(3)"))
        assert result.status == ExecutionStatus.NONZERO_EXIT
        assert result.exit_code == 3
        assert not result.ok

    def test_uncaught_exception_is_nonzero_exit(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("raise ValueError('boom')"))
        assert result.status == ExecutionStatus.NONZERO_EXIT
        assert result.exit_code != 0


class TestStderr:
    def test_stderr_is_captured_separately_from_stdout(self, tmp_path: Path) -> None:
        code = "import sys; print('out'); print('err', file=sys.stderr)"
        result = asyncio.run(_executor(tmp_path).execute(code))
        assert "out" in result.stdout.preview
        assert "err" in result.stderr.preview
        assert "err" not in result.stdout.preview


class TestTimeout:
    def test_long_running_code_times_out(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("import time; time.sleep(30)", timeout=0.3))
        assert result.status == ExecutionStatus.TIMEOUT
        assert result.error_message is not None
        assert "timeout" in result.error_message.lower()

    def test_timeout_actually_kills_the_child(self, tmp_path: Path) -> None:
        """A quick regression guard: a killed process's exit code must
        not read back as a clean 0."""
        result = asyncio.run(_executor(tmp_path).execute("import time; time.sleep(30)", timeout=0.3))
        assert result.exit_code != 0


class TestCancellation:
    def test_cancelling_the_awaiting_task_kills_the_child(self, tmp_path: Path) -> None:
        async def _run() -> None:
            task = asyncio.ensure_future(_executor(tmp_path).execute("import time; time.sleep(30)"))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run())


class TestOutputBounded:
    def test_large_output_is_truncated_with_a_small_preview(self, tmp_path: Path) -> None:
        # ~1MB of output, well past the default head/tail bounds.
        code = "for i in range(200_000):\n    print('x' * 20)\n"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=30))
        assert result.ok
        assert result.stdout.truncated
        assert len(result.stdout.preview) < result.stdout.total_bytes
        assert result.stdout.total_bytes > 1_000_000

    def test_full_output_is_always_available_as_an_artifact(self, tmp_path: Path) -> None:
        code = "for i in range(200_000):\n    print('x' * 20)\n"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=30))
        assert result.stdout.artifact_path is not None
        artifact = Path(result.stdout.artifact_path)
        assert artifact.exists()
        assert artifact.stat().st_size == result.stdout.total_bytes

    def test_small_output_is_not_truncated(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print('small')"))
        assert not result.stdout.truncated
        assert result.stdout.preview.strip() == "small"

    def test_sha256_matches_the_artifact_content(self, tmp_path: Path) -> None:
        import hashlib

        result = asyncio.run(_executor(tmp_path).execute("print('hello')"))
        artifact_bytes = Path(result.stdout.artifact_path).read_bytes()  # type: ignore[arg-type]
        assert hashlib.sha256(artifact_bytes).hexdigest() == result.stdout.sha256

    def test_line_count_is_accurate(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("for i in range(50): print(i)"))
        assert result.stdout.total_lines == 50
