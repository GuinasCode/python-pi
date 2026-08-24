"""Slice A5 — artifact directory convention, metadata, and output modes.

Spec section 14: `.pi/runs/<run-id>/execute-code/<execution-id>/` holds
everything one execution produced (script.py, stdout.log, stderr.log —
already written by capture.py/runner.py — plus metadata.json here) so a
session can be inspected after the fact without re-running anything.

Spec section 13: output_mode controls what `OutputCapture.preview`
contains, layered on top of the Slice A1 bounded capture that always
happens regardless of mode (the artifact file and hash are never
optional — only how much of it also rides along in the *preview* field
changes):

  - "head_tail" (default): the existing bounded head+tail preview —
    safe regardless of actual stream size.
  - "summary": alias for "head_tail" — the safety guarantee is identical;
    the distinction is about intent (the script prints its own summary),
    not a different capture mechanism.
  - "full": preview becomes the entire captured stream — only up to
    `_FULL_MODE_HARD_CAP_BYTES`, even here, so an explicit opt-in still
    can't blow the model's context by literal accident; genuinely large
    output still requires reading the artifact file.
  - "artifact": preview is forced empty — the caller only wants the
    artifact_path pointer and the stats (bytes/lines/hash), not any
    inline text at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from pi_runtime.execute_code.result import OutputCapture

if TYPE_CHECKING:
    from pi_runtime.execute_code.result import ExecuteCodeResult
    from pi_runtime.execute_code.rpc import RpcCallRecord

_FULL_MODE_HARD_CAP_BYTES = 5 * 1024 * 1024


def runs_artifacts_root(*, run_id: str, base_dir: Path | None = None) -> Path:
    """`.pi/runs/<run_id>/execute-code/` under `base_dir` (defaults to
    the current working directory — the project root in `mode="project"`
    execute_code calls, same convention `mode="strict"` calls also use
    for their artifacts even though their *script* cwd differs)."""
    root = base_dir if base_dir is not None else Path.cwd()
    return root / ".pi" / "runs" / run_id / "execute-code"


def apply_output_mode(capture: OutputCapture, *, output_mode: str) -> OutputCapture:
    if output_mode == "artifact":
        return OutputCapture(
            preview="",
            truncated=capture.truncated,
            total_bytes=capture.total_bytes,
            total_lines=capture.total_lines,
            artifact_path=capture.artifact_path,
            sha256=capture.sha256,
        )
    if output_mode == "full":
        if capture.artifact_path is not None and capture.total_bytes <= _FULL_MODE_HARD_CAP_BYTES:
            full_text = Path(capture.artifact_path).read_text(encoding="utf-8", errors="replace")
            return OutputCapture(
                preview=full_text,
                truncated=False,
                total_bytes=capture.total_bytes,
                total_lines=capture.total_lines,
                artifact_path=capture.artifact_path,
                sha256=capture.sha256,
            )
        # over the hard cap: fall through to the safe bounded preview
        # rather than silently loading megabytes into the model's context.
        return capture
    # "head_tail" and "summary" both mean: use the bounded preview as-is.
    return capture


def write_metadata(
    artifacts_dir: Path,
    result: ExecuteCodeResult,
    *,
    rpc_trace: list[RpcCallRecord],
) -> None:
    """Spec section 14's "metadata; timing; exit code; tool RPC trace
    resumido" — one JSON file alongside script.py/stdout.log/stderr.log,
    written after execution completes so it can capture the final
    status/timing rather than needing a second write pass."""
    metadata = {
        "status": result.status.value,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
        "mode": result.mode,
        "error_message": result.error_message,
        "rpc_call_count": result.rpc_call_count,
        "rpc_trace": [asdict(record) for record in rpc_trace],
        "stdout": {
            "truncated": result.stdout.truncated,
            "total_bytes": result.stdout.total_bytes,
            "total_lines": result.stdout.total_lines,
            "artifact_path": result.stdout.artifact_path,
            "sha256": result.stdout.sha256,
        },
        "stderr": {
            "truncated": result.stderr.truncated,
            "total_bytes": result.stderr.total_bytes,
            "total_lines": result.stderr.total_lines,
            "artifact_path": result.stderr.artifact_path,
            "sha256": result.stderr.sha256,
        },
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = ["apply_output_mode", "runs_artifacts_root", "write_metadata"]
