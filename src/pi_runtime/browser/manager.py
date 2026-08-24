"""BrowserManager — Slice B1: real session lifecycle around a single
shared Playwright Browser process.

```text
BrowserManager
   |
   +-- BrowserSession (per browser_session_id)
   |      +-- BrowserContext (Playwright)  <- isolation boundary
   |      +-- Page[]
   |      +-- active_page
   |      +-- metadata
   |
   +-- persistence: sessions survive across calls until closed/expired
   +-- policy: open_session() goes through PolicyEngine.evaluate("browser")
```

Lifecycle (spec section 23): create -> attach -> reuse -> detach ->
close -> cleanup.

  - **create**: `open_session()` — launches the shared Browser on first
    use, opens a fresh isolated Context + Page.
  - **attach**: `attach_session(session_id)` — retrieves an existing,
    still-open session and marks it touched (used).
  - **reuse**: not a separate method — every session-scoped call
    (`navigate`, or a future `click`/`type`/...) *is* reuse: it acts on
    the same Context/Page `open_session` created, never a fresh one.
  - **detach**: `detach_session(session_id)` — removes a session from
    this manager's registry *without* closing its browser resources,
    handing ownership to the caller. A real, distinct operation from
    close (the underlying Context/Pages stay alive), not an alias.
  - **close**: `close_session(session_id)` — closes the Context (and
    every Page in it) and marks the session closed.
  - **cleanup**: `cleanup_expired()` reaps sessions past their
    `timeout_seconds` (call periodically, or at loop-iteration
    boundaries — the same "checked at boundaries, not via a background
    timer" pattern `pi_runtime.state.Budget` already uses); `close_all()`
    tears down every live session plus the shared Browser/Playwright
    driver itself — call it at task end, on a fatal error, or from a
    `finally`. `BrowserManager` also works as an async context manager
    (`async with BrowserManager() as mgr: ...`) for exactly that.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from pi_runtime.browser.session import BrowserSession
from pi_runtime.browser.snapshot import PageSnapshot, StaleRefError, capture_snapshot, resolve_locator
from pi_runtime.research import Evidence
from pi_runtime.tools import PolicyEngine

if TYPE_CHECKING:
    from playwright.async_api import Browser, Locator, Playwright

_DEFAULT_TIMEOUT_SECONDS = 300.0
_BODY_TEXT_EXCERPT_CHARS = 500


class NavigationResult:
    """Kept as a plain-field result object (not a dataclass with
    Evidence as a required positional field) so `ok=False` results don't
    need a dummy Evidence — mirrors the Slice A result contracts'
    "distinguish outcomes explicitly" principle."""

    __slots__ = ("error", "evidence", "ok", "page_id", "session_id", "url")

    def __init__(
        self,
        *,
        session_id: str,
        url: str,
        ok: bool,
        evidence: Evidence | None = None,
        error: str | None = None,
        page_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.url = url
        self.ok = ok
        self.evidence = evidence
        self.error = error
        self.page_id = page_id


class BrowserManager:
    def __init__(
        self,
        *,
        policy_engine: PolicyEngine | None = None,
        default_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        headless: bool = True,
    ) -> None:
        self._policy_engine = policy_engine
        self._default_timeout_seconds = default_timeout_seconds
        self._headless = headless
        self._sessions: dict[str, BrowserSession] = {}
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> BrowserManager:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close_all()

    async def _ensure_browser(self) -> Browser:
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
        return self._browser

    async def open_session(self, *, timeout_seconds: float | None = None) -> BrowserSession:
        if self._policy_engine is not None:
            self._policy_engine.evaluate("browser")
        browser = await self._ensure_browser()
        context = await browser.new_context()
        session = BrowserSession(
            session_id=uuid.uuid4().hex[:8],
            context=context,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds,
        )
        page = await context.new_page()
        session.add_page(page)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> BrowserSession | None:
        return self._sessions.get(session_id)

    def attach_session(self, session_id: str) -> BrowserSession | None:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return None
        session.touch()
        return session

    def detach_session(self, session_id: str) -> BrowserSession | None:
        """Removes the session from this manager's bookkeeping without
        closing its Context/Pages — the caller now owns cleanup."""
        return self._sessions.pop(session_id, None)

    async def close_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return False
        await session.close()
        return True

    async def cleanup_expired(self) -> list[str]:
        expired_ids = [sid for sid, session in self._sessions.items() if session.is_expired()]
        for session_id in expired_ids:
            await self._sessions[session_id].close()
        return expired_ids

    async def close_all(self) -> None:
        for session in list(self._sessions.values()):
            if not session.closed:
                await session.close()
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def navigate(self, session_id: str, url: str, *, timeout: float = 30.0) -> NavigationResult:
        """Spec section 21's opening example's first step. Reuses the
        session's existing active Page (never opens a fresh browser per
        call) — failures (bad session, closed/expired session, a real
        Playwright navigation error) all become `NavigationResult.error`
        instead of raising, same failure-isolation invariant the
        one-shot predecessor had."""
        session = self._sessions.get(session_id)
        if session is None:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="no such browser session")
        if session.closed:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="browser session is closed")
        if session.is_expired():
            await session.close()
            return NavigationResult(session_id=session_id, url=url, ok=False, error="browser session timed out")

        session.touch()
        page = session.get_page()
        if page is None:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="session has no active page")

        try:
            await page.goto(url, timeout=timeout * 1000, wait_until="load")
        except Exception as exc:
            session.navigations.append(url)
            return NavigationResult(session_id=session_id, url=url, ok=False, error=str(exc))

        session.navigations.append(url)
        session.ref_map.clear()  # any refs from before this navigation are no longer valid
        evidence = await _page_to_evidence(page, url)
        return NavigationResult(
            session_id=session_id, url=url, ok=True, evidence=evidence, page_id=session.active_page_id
        )

    async def snapshot(self, session_id: str, *, page_id: str | None = None) -> PageSnapshot:
        """Spec section 26: a bounded accessibility-tree representation,
        never raw HTML. Replaces the session's ref map wholesale — refs
        from a previous snapshot stop resolving after this call."""
        session = self._require_open_session(session_id)
        page = session.get_page(page_id)
        if page is None:
            raise ValueError(f"session {session_id!r} has no page {page_id!r}")
        target_page_id = page_id if page_id is not None else session.active_page_id
        assert target_page_id is not None

        page_snapshot, ref_map = await capture_snapshot(page, page_id=target_page_id)
        session.ref_map = ref_map
        session.touch()
        return page_snapshot

    def resolve_ref(self, session_id: str, ref: str) -> Locator:
        """Spec section 27's invariant: an unknown or stale ref fails
        clearly (`StaleRefError`), never silently resolves to a
        different element."""
        session = self._require_open_session(session_id)
        target = session.ref_map.get(ref)
        if target is None:
            raise StaleRefError(f"ref {ref!r} is not from the most recent snapshot of session {session_id!r}")
        page = session.get_page(target.page_id)
        if page is None:
            raise StaleRefError(f"ref {ref!r} refers to a page that is no longer open")
        return resolve_locator(page, target)

    def _require_open_session(self, session_id: str) -> BrowserSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"no such browser session: {session_id!r}")
        if session.closed:
            raise ValueError(f"browser session {session_id!r} is closed")
        return session


async def _page_to_evidence(page: Any, requested_url: str) -> Evidence:
    title = await page.title()
    try:
        text = await page.inner_text("body")
    except Exception:
        text = ""
    return Evidence(
        source_id=page.url or requested_url,
        url=page.url or requested_url,
        title=title,
        excerpt=text[:_BODY_TEXT_EXCERPT_CHARS],
        retrieved_at=time.time(),
        extraction_method="browser",
    )


__all__ = ["BrowserManager", "BrowserSession", "NavigationResult"]
