"""Slice B6 — pluggable browser backend (spec section 33).

Not CDP-by-default: `BrowserBackend.PLAYWRIGHT_LOCAL` (Playwright
launches and owns its own Chromium process — everything B1-B5 already
use) stays the default. `BrowserBackend.CDP` is an explicit opt-in that
connects to an *already-running* browser over the Chrome DevTools
Protocol instead of launching one — real, not a stub: verified against
an actual Chromium process started with `--remote-debugging-port` and
connected to via Playwright's own `connect_over_cdp`.

`executable_path` (a real, honored pass-through to
`BrowserType.launch(executable_path=...)`) lets `PLAYWRIGHT_LOCAL` use a
non-bundled Chromium/Chrome binary.

`browser.profile` (a persistent, on-disk Chrome user-data-dir) is
explicitly NOT implemented here — it would require a fundamentally
different lifecycle (`launch_persistent_context` returns one browser
process bound to exactly one context, incompatible with this module's
"one shared Browser backs every isolated-context session" model) and
building a second, parallel session model just for this one config
value is more architecture than the spec's own "fallback razoável"
framing asks for. Documented as a known gap (spec section 11's honesty
principle extends here too), not silently unsupported.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright


class BrowserBackend(str, Enum):
    PLAYWRIGHT_LOCAL = "playwright_local"
    CDP = "cdp"


class BrowserBackendConfigError(ValueError):
    """Raised for a backend configuration that can't possibly work —
    e.g. BrowserBackend.CDP with no cdp_url — caught at BrowserManager
    construction/first-use time, not deep inside a random tool call."""


async def launch_browser(
    playwright: Playwright,
    *,
    backend: BrowserBackend,
    headless: bool,
    cdp_url: str | None,
    executable_path: str | None,
) -> Browser:
    if backend == BrowserBackend.CDP:
        if not cdp_url:
            raise BrowserBackendConfigError("BrowserBackend.CDP requires cdp_url to be set")
        return await playwright.chromium.connect_over_cdp(cdp_url)

    launch_kwargs: dict[str, object] = {"headless": headless}
    if executable_path is not None:
        launch_kwargs["executable_path"] = executable_path
    return await playwright.chromium.launch(**launch_kwargs)  # type: ignore[arg-type]


__all__ = ["BrowserBackend", "BrowserBackendConfigError", "launch_browser"]
