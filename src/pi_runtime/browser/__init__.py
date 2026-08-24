"""pi_runtime.browser — persistent Playwright browser harness (GAP B).

Replaces the previous one-shot browser_fetch_url wrapper: sessions here
hold a real Playwright BrowserContext/Page across multiple tool calls
(spec section 21) instead of opening and closing a browser per call.
"""

from __future__ import annotations

from pi_runtime.browser.backend import BrowserBackend, BrowserBackendConfigError
from pi_runtime.browser.downloads import DownloadResult
from pi_runtime.browser.evaluate import EvaluateResult
from pi_runtime.browser.interactions import InteractionResult, InteractionStatus
from pi_runtime.browser.manager import BrowserManager, NavigationResult
from pi_runtime.browser.session import BrowserSession
from pi_runtime.browser.snapshot import PageSnapshot, StaleRefError
from pi_runtime.browser.telemetry import BrowserTelemetryRecord

__all__ = [
    "BrowserBackend",
    "BrowserBackendConfigError",
    "BrowserManager",
    "BrowserSession",
    "BrowserTelemetryRecord",
    "DownloadResult",
    "EvaluateResult",
    "InteractionResult",
    "InteractionStatus",
    "NavigationResult",
    "PageSnapshot",
    "StaleRefError",
]
