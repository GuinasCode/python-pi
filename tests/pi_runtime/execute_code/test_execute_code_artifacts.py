"""Slice A5 — artifact directory convention, metadata.json, output modes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pi_runtime.execute_code.artifacts import apply_output_mode, runs_artifacts_root
from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS
from pi_runtime.execute_code.result import ExecutionStatus, OutputCapture
from pi_runtime.execute_code.runner import CodeExecutor


def _executor(tmp_path: Path) -> CodeExecutor:
    return CodeExecutor(artifacts_root=tmp_path / "artifacts")


class TestRunsArtifactsRoot:
    def test_matches_the_dot_pi_runs_convention(self, tmp_path: Path) -> None:
        root = runs_artifacts_root(run_id="abc123", base_dir=tmp_path)
        assert root == tmp_path / ".pi" / "runs" / "abc123" / "execute-code"

    def test_code_executor_uses_it_when_only_run_id_is_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        executor = CodeExecutor(run_id="myrun")
        result = asyncio.run(executor.execute("print('x')", timeout=15))
        assert result.status == ExecutionStatus.SUCCESS
        assert result.artifacts_dir is not None
        assert ".pi" in result.artifacts_dir
        assert "myrun" in result.artifacts_dir
        assert "execute-code" in result.artifacts_dir


class TestMetadataFile:
    def test_metadata_json_is_written_with_status_and_timing(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print('hello')", timeout=15))
        metadata_path = Path(result.artifacts_dir) / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["status"] == "success"
        assert metadata["exit_code"] == 0
        assert metadata["duration_ms"] > 0

    def test_metadata_records_the_rpc_trace(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_text("hi")
        code = f"""
from pi_tools import read_file
read_file({str(target)!r})
"""
        result = asyncio.run(_executor(tmp_path).execute(code, rpc_handlers=DEFAULT_HANDLERS, timeout=15))
        metadata = json.loads((Path(result.artifacts_dir) / "metadata.json").read_text())
        assert len(metadata["rpc_trace"]) == 1
        assert metadata["rpc_trace"][0]["tool"] == "read_file"
        assert metadata["rpc_trace"][0]["status"] == "success"

    def test_all_expected_artifact_files_are_present(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print('x')", timeout=15))
        artifacts_dir = Path(result.artifacts_dir)
        assert (artifacts_dir / "script.py").exists()
        assert (artifacts_dir / "stdout.log").exists()
        assert (artifacts_dir / "stderr.log").exists()
        assert (artifacts_dir / "metadata.json").exists()


class TestOutputModeArtifact:
    def test_artifact_mode_forces_an_empty_preview_but_keeps_the_pointer(self, tmp_path: Path) -> None:
        result = asyncio.run(_executor(tmp_path).execute("print('some content')", timeout=15, output_mode="artifact"))
        assert result.stdout.preview == ""
        assert result.stdout.artifact_path is not None
        assert Path(result.stdout.artifact_path).read_text().strip() == "some content"


class TestOutputModeFull:
    def test_full_mode_returns_the_entire_small_stream_uncut(self, tmp_path: Path) -> None:
        code = "for i in range(50):\n    print(f'line{i}')\n"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, output_mode="full"))
        assert "line0" in result.stdout.preview
        assert "line49" in result.stdout.preview
        assert not result.stdout.truncated

    def test_full_mode_falls_back_to_bounded_preview_above_the_hard_cap(self, tmp_path: Path) -> None:
        from pi_runtime.execute_code import artifacts as artifacts_module

        original_cap = artifacts_module._FULL_MODE_HARD_CAP_BYTES
        artifacts_module._FULL_MODE_HARD_CAP_BYTES = 100
        try:
            code = "print('x' * 10_000)\n"
            result = asyncio.run(_executor(tmp_path).execute(code, timeout=15, output_mode="full"))
        finally:
            artifacts_module._FULL_MODE_HARD_CAP_BYTES = original_cap
        assert len(result.stdout.preview) < 10_000


class TestOutputModeHeadTailIsDefault:
    def test_default_output_mode_is_head_tail_bounded(self, tmp_path: Path) -> None:
        code = "for i in range(200_000):\n    print('x' * 20)\n"
        result = asyncio.run(_executor(tmp_path).execute(code, timeout=30))
        assert result.stdout.truncated
        assert len(result.stdout.preview) < result.stdout.total_bytes


class TestApplyOutputModeUnit:
    def _capture(self) -> OutputCapture:
        return OutputCapture(
            preview="head...tail", truncated=True, total_bytes=1000, total_lines=10, artifact_path=None, sha256="x"
        )

    def test_summary_and_head_tail_are_equivalent(self) -> None:
        capture = self._capture()
        assert apply_output_mode(capture, output_mode="summary") == apply_output_mode(capture, output_mode="head_tail")

    def test_full_mode_without_an_artifact_path_falls_back_safely(self) -> None:
        capture = self._capture()
        result = apply_output_mode(capture, output_mode="full")
        assert result == capture
