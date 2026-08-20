"""Writes ``.eval/`` run artifacts, mirroring the original TS harness's
directory convention: ``runs.jsonl`` indexes every completed run, and each
run's native Pi session transcript is snapshotted under ``sessions/``.

Deliberately decoupled from :mod:`pi_evals.pi_harness` (accepts plain
dicts, not ``PiHarnessResult``) so this stays a small, independently
testable utility — callers pass ``dataclasses.asdict(result.usage)`` etc.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_EVAL_DIR", "EvalArtifactWriter", "RunRecord"]

DEFAULT_EVAL_DIR = ".eval"


@dataclass
class RunRecord:
    """One row of ``runs.jsonl``."""

    run_id: str
    name: str
    timestamp: int
    harness_name: str | None = None
    output: Any = None
    usage: dict[str, Any] | None = None
    total_ms: float | None = None
    passed: bool | None = None
    session_path: str | None = None


class EvalArtifactWriter:
    """Appends run records to ``<root>/runs.jsonl`` and, when a session
    snapshot is provided, writes it to ``<root>/sessions/<run_id>.jsonl``.
    """

    def __init__(self, root: str | Path = DEFAULT_EVAL_DIR) -> None:
        self._root = Path(root)
        self._sessions_dir = self._root / "sessions"
        self._runs_path = self._root / "runs.jsonl"

    @property
    def root(self) -> Path:
        return self._root

    def write_run(
        self,
        *,
        name: str,
        harness_name: str | None = None,
        output: Any = None,
        usage: dict[str, Any] | None = None,
        total_ms: float | None = None,
        passed: bool | None = None,
        session_snapshot: str | None = None,
    ) -> RunRecord:
        """Append one run's outcome to runs.jsonl, snapshotting its native
        session transcript alongside it when provided. Returns the record
        that was written (with the generated run_id and session_path)."""
        self._root.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())

        session_path: str | None = None
        if session_snapshot is not None:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            session_file = self._sessions_dir / f"{run_id}.jsonl"
            session_file.write_text(session_snapshot, encoding="utf-8")
            # Forward slashes regardless of OS: this path is written into a
            # JSON artifact that may be read back on a different platform.
            session_path = (Path("sessions") / session_file.name).as_posix()

        record = RunRecord(
            run_id=run_id,
            name=name,
            timestamp=int(time.time() * 1000),
            harness_name=harness_name,
            output=output,
            usage=usage,
            total_ms=total_ms,
            passed=passed,
            session_path=session_path,
        )
        with open(self._runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_record_to_dict(record)) + "\n")
        return record

    def read_runs(self) -> list[dict[str, Any]]:
        """Read back every record written so far (empty list if runs.jsonl doesn't exist yet)."""
        if not self._runs_path.is_file():
            return []
        lines = self._runs_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def _record_to_dict(record: RunRecord) -> dict[str, Any]:
    return {
        "runId": record.run_id,
        "name": record.name,
        "timestamp": record.timestamp,
        "harnessName": record.harness_name,
        "output": record.output,
        "usage": record.usage,
        "totalMs": record.total_ms,
        "passed": record.passed,
        "sessionPath": record.session_path,
    }
