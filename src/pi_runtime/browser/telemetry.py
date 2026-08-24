"""Slice B5 — browser action telemetry (spec section 37).

One record per action: session, page, action, target, duration, status,
error, url_before/after, artifact_refs. `BrowserManager` calls
`emit()` after every action when a sink is configured — opt-in (no sink,
no overhead, matching the rest of this codebase's "don't build a
mechanism with no consumer" stance). Field values that could hold
user-typed text (fill/type's `text` argument) are never included as-is:
spec section 37's "não registrar valores sensíveis de campos digitados
sem necessidade" — since a generic browser harness has no reliable way
to classify which typed field is a password, the safe default is to
redact *every* typed value, not attempt (and inevitably get wrong)
sensitivity detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

_REDACTED = "[value redacted]"


@dataclass
class BrowserTelemetryRecord:
    session_id: str
    action: str
    status: str
    duration_ms: float
    page_id: str | None = None
    target: str | None = None
    error: str | None = None
    url_before: str | None = None
    url_after: str | None = None
    artifact_refs: list[str] = field(default_factory=list)


class BrowserTelemetrySink(Protocol):
    def __call__(self, record: BrowserTelemetryRecord) -> None: ...


def redact_typed_value(_value: str) -> str:
    return _REDACTED


__all__ = ["BrowserTelemetryRecord", "BrowserTelemetrySink", "redact_typed_value"]
