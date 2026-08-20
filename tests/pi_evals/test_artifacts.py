"""Tests for pi_evals.artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pi_evals.artifacts import EvalArtifactWriter


class TestEvalArtifactWriter:
    def test_write_run_creates_runs_jsonl(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        record = writer.write_run(name="smoke", harness_name="baseline", output="Paris", passed=True)

        runs_path = tmp_path / ".eval" / "runs.jsonl"
        assert runs_path.is_file()
        lines = runs_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["name"] == "smoke"
        assert row["harnessName"] == "baseline"
        assert row["output"] == "Paris"
        assert row["passed"] is True
        assert row["runId"] == record.run_id

    def test_write_run_appends_without_overwriting(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        writer.write_run(name="a")
        writer.write_run(name="b")

        runs = writer.read_runs()
        assert [r["name"] for r in runs] == ["a", "b"]

    def test_session_snapshot_written_under_sessions_dir(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        record = writer.write_run(name="smoke", session_snapshot='{"role": "user", "content": "hi"}')

        assert record.session_path is not None
        assert Path(record.session_path) == Path("sessions") / f"{record.run_id}.jsonl"
        session_file = tmp_path / ".eval" / "sessions" / f"{record.run_id}.jsonl"
        assert session_file.is_file()
        assert json.loads(session_file.read_text(encoding="utf-8"))["content"] == "hi"

    def test_no_session_snapshot_means_no_session_path(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        record = writer.write_run(name="smoke")
        assert record.session_path is None
        assert not (tmp_path / ".eval" / "sessions").exists()

    def test_read_runs_returns_empty_list_when_nothing_written(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        assert writer.read_runs() == []

    def test_each_run_gets_a_unique_run_id(self, tmp_path: Path) -> None:
        writer = EvalArtifactWriter(tmp_path / ".eval")
        r1 = writer.write_run(name="a")
        r2 = writer.write_run(name="a")
        assert r1.run_id != r2.run_id
