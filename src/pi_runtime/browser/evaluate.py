"""Slice B4 — browser_evaluate, with output bounds "da mesma família do
execute_code" (spec section 29): never let a JS return value explode
directly into the model's context. Small results ride inline as a JSON
preview; anything past the cap is written to an artifact file, mirroring
`pi_runtime.execute_code`'s own bounded-output philosophy without
reusing its byte-stream-oriented `BoundedStreamCapture` (an evaluate
result is a single already-in-memory value, not a stream to pump)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pi_runtime.browser.interactions import InteractionStatus

_MAX_PREVIEW_CHARS = 20_000


@dataclass
class EvaluateResult:
    status: InteractionStatus
    preview: str
    truncated: bool
    total_chars: int
    artifact_path: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == InteractionStatus.SUCCESS


def bound_evaluate_result(value: object, *, artifacts_dir: Path | None) -> EvaluateResult:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)

    total_chars = len(text)
    if total_chars <= _MAX_PREVIEW_CHARS:
        return EvaluateResult(status=InteractionStatus.SUCCESS, preview=text, truncated=False, total_chars=total_chars)

    artifact_path: str | None = None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        target = artifacts_dir / f"evaluate-{digest}.json"
        target.write_text(text, encoding="utf-8")
        artifact_path = str(target)

    preview = text[:_MAX_PREVIEW_CHARS] + "...(truncated)"
    return EvaluateResult(
        status=InteractionStatus.SUCCESS,
        preview=preview,
        truncated=True,
        total_chars=total_chars,
        artifact_path=artifact_path,
    )


__all__ = ["EvaluateResult", "bound_evaluate_result"]
