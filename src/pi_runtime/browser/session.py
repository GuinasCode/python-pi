"""BrowserSession — Slice B1: a genuinely persistent Playwright session.

The previous pi_runtime.browser.BrowserManager wrapped a one-shot
browser_fetch_url call (open browser -> goto -> extract -> close, every
single tool call) with session bookkeeping around it. That's not a
persistent session — a real `browser_click` after a real
`browser_navigate` needs the *same* Page still open, with the DOM state
that navigation left behind. This module holds the real thing: one
Playwright `BrowserContext` (isolated cookies/storage per session, spec
section 32) and its `Page`s, kept alive across calls until the session
is explicitly closed or expires.

One shared `Browser` process backs every session in a `BrowserManager`
(launching a full browser process per session would be wasteful);
per-session isolation comes from Playwright's own `BrowserContext`
boundary, which already gives each session its own cookie jar and
storage — matching spec section 32's "session isolated, credentials not
persisted unless explicitly enabled" without needing a second OS
process per session.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

    from pi_runtime.browser.snapshot import RefTarget


@dataclass
class BrowserSession:
    """Owns one BrowserContext and its Pages. `active_page_id` names
    which page interactions default to when a tool call doesn't specify
    one explicitly (spec section 24: "active page", never assume there's
    only one tab)."""

    session_id: str
    context: BrowserContext
    pages: dict[str, Page] = field(default_factory=dict)
    active_page_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0
    closed: bool = False
    navigations: list[str] = field(default_factory=list)
    # Replaced wholesale by every browser_snapshot call (see
    # pi_runtime.browser.snapshot) — a ref only resolves against the
    # most recent snapshot, on purpose (spec section 27).
    ref_map: dict[str, RefTarget] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_used_at = time.time()

    def is_expired(self) -> bool:
        return not self.closed and (time.time() - self.last_used_at) > self.timeout_seconds

    def add_page(self, page: Page, *, make_active: bool = True) -> str:
        page_id = uuid.uuid4().hex[:8]
        self.pages[page_id] = page
        if make_active or self.active_page_id is None:
            self.active_page_id = page_id
        return page_id

    def get_page(self, page_id: str | None = None) -> Page | None:
        """Returns the named page, or the active page when `page_id` is
        None — the common case, since most tool calls act on "whatever
        page is currently in front"."""
        target = page_id if page_id is not None else self.active_page_id
        if target is None:
            return None
        return self.pages.get(target)

    def remove_page(self, page_id: str) -> None:
        self.pages.pop(page_id, None)
        if self.active_page_id == page_id:
            self.active_page_id = next(iter(self.pages), None)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.context.close()


__all__ = ["BrowserSession"]
