"""Slice B3 — result/status contracts for click/type/fill/press/select/
scroll/wait. The interactions themselves live as BrowserManager methods
(manager.py) — same object surface as navigate/snapshot, not a parallel
API a caller has to juggle separately.

Spec section 41: don't collapse timeout/stale_ref/not_found/error into
one indistinguishable string — `InteractionStatus` lets a caller decide
retry/repair/replan/ask/stop from the status alone, the same contract
`pi_runtime.execute_code.result.ExecutionStatus` already established
for GAP A.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InteractionStatus(str, Enum):
    SUCCESS = "success"
    STALE_REF = "stale_ref"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    POLICY_DENIED = "policy_denied"
    ERROR = "error"


@dataclass
class InteractionResult:
    status: InteractionStatus
    action: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == InteractionStatus.SUCCESS


def classify_playwright_error(exc: Exception) -> InteractionStatus:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    if isinstance(exc, PlaywrightTimeoutError):
        return InteractionStatus.TIMEOUT
    message = str(exc).lower()
    if "not found" in message or "no element" in message or "resolved to 0 elements" in message:
        return InteractionStatus.NOT_FOUND
    return InteractionStatus.ERROR


__all__ = ["InteractionResult", "InteractionStatus", "classify_playwright_error"]
