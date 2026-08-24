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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi_runtime.browser.backend import BrowserBackend, launch_browser
from pi_runtime.browser.downloads import DownloadResult, save_download
from pi_runtime.browser.evaluate import EvaluateResult, bound_evaluate_result
from pi_runtime.browser.interactions import InteractionResult, InteractionStatus, classify_playwright_error
from pi_runtime.browser.session import BrowserSession
from pi_runtime.browser.snapshot import PageSnapshot, StaleRefError, capture_snapshot, resolve_locator
from pi_runtime.browser.telemetry import BrowserTelemetryRecord, redact_typed_value
from pi_runtime.research import Evidence
from pi_runtime.tools import PolicyEngine, PolicyViolation

if TYPE_CHECKING:
    from playwright.async_api import Browser, Locator, Playwright

    from pi_runtime.browser.telemetry import BrowserTelemetrySink

_DEFAULT_ACTION_TIMEOUT_SECONDS = 5.0

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
        telemetry_sink: BrowserTelemetrySink | None = None,
        backend: BrowserBackend = BrowserBackend.PLAYWRIGHT_LOCAL,
        cdp_url: str | None = None,
        executable_path: str | None = None,
    ) -> None:
        self._policy_engine = policy_engine
        self._default_timeout_seconds = default_timeout_seconds
        self._headless = headless
        self._telemetry_sink = telemetry_sink
        self._backend = backend
        self._cdp_url = cdp_url
        self._executable_path = executable_path
        self._sessions: dict[str, BrowserSession] = {}
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    async def __aenter__(self) -> BrowserManager:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close_all()

    def _emit(
        self,
        *,
        session_id: str,
        action: str,
        status: str,
        duration_ms: float,
        page_id: str | None = None,
        target: str | None = None,
        error: str | None = None,
        url_before: str | None = None,
        url_after: str | None = None,
        artifact_refs: list[str] | None = None,
    ) -> None:
        if self._telemetry_sink is None:
            return
        self._telemetry_sink(
            BrowserTelemetryRecord(
                session_id=session_id,
                action=action,
                status=status,
                duration_ms=duration_ms,
                page_id=page_id,
                target=target,
                error=error,
                url_before=url_before,
                url_after=url_after,
                artifact_refs=artifact_refs or [],
            )
        )

    def _check_policy(self, tool_name: str) -> str | None:
        """Returns an error message if policy denies the action, else
        None. Never raises — every caller converts this into its own
        typed "denied" outcome instead of propagating PolicyViolation,
        so a policy denial looks like any other reported failure, not a
        crash (same failure-isolation invariant as the rest of this
        module)."""
        if self._policy_engine is None:
            return None
        try:
            self._policy_engine.evaluate(tool_name)
        except PolicyViolation as exc:
            return str(exc)
        return None

    async def _ensure_browser(self) -> Browser:
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await launch_browser(
                self._playwright,
                backend=self._backend,
                headless=self._headless,
                cdp_url=self._cdp_url,
                executable_path=self._executable_path,
            )
        return self._browser

    async def open_session(
        self, *, timeout_seconds: float | None = None, storage_state_path: str | None = None
    ) -> BrowserSession:
        """`storage_state_path` (spec section 32's "persistent profile
        opcional"): when given and the file exists, the new context
        loads cookies/localStorage from it — otherwise the session is
        fully ephemeral, the default. Nothing is written back
        automatically; call `save_storage_state` explicitly when you
        want this session's state persisted. Loading one goes through
        the higher-risk "browser_persistent_profile" policy check, not
        just "browser"."""
        if self._policy_engine is not None:
            self._policy_engine.evaluate("browser")
            if storage_state_path is not None:
                self._policy_engine.evaluate("browser_persistent_profile")
        browser = await self._ensure_browser()
        if storage_state_path is not None and Path(storage_state_path).exists():
            context = await browser.new_context(storage_state=storage_state_path)
        else:
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

    async def save_storage_state(self, session_id: str, path: str) -> None:
        """Explicit opt-in persistence — never automatic (spec section
        32: "credentials not persisted unless explicitly enabled")."""
        session = self._require_open_session(session_id)
        await session.context.storage_state(path=path)

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

        denial = self._check_policy("browser_navigate")
        if denial is not None:
            return NavigationResult(session_id=session_id, url=url, ok=False, error=f"policy_denied: {denial}")

        session.touch()
        page = session.get_page()
        if page is None:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="session has no active page")

        url_before = page.url
        started = time.monotonic()
        try:
            await page.goto(url, timeout=timeout * 1000, wait_until="load")
        except Exception as exc:
            session.navigations.append(url)
            self._emit(
                session_id=session_id,
                action="navigate",
                status="error",
                duration_ms=(time.monotonic() - started) * 1000,
                target=url,
                error=str(exc),
                url_before=url_before,
            )
            return NavigationResult(session_id=session_id, url=url, ok=False, error=str(exc))

        session.navigations.append(url)
        session.ref_map.clear()  # any refs from before this navigation are no longer valid
        evidence = await _page_to_evidence(page, url)
        self._emit(
            session_id=session_id,
            action="navigate",
            status="success",
            duration_ms=(time.monotonic() - started) * 1000,
            page_id=session.active_page_id,
            target=url,
            url_before=url_before,
            url_after=page.url,
        )
        return NavigationResult(
            session_id=session_id, url=url, ok=True, evidence=evidence, page_id=session.active_page_id
        )

    async def snapshot(self, session_id: str, *, page_id: str | None = None) -> PageSnapshot:
        """Spec section 26: a bounded accessibility-tree representation,
        never raw HTML. Replaces the session's ref map wholesale — refs
        from a previous snapshot stop resolving after this call."""
        session = self._require_open_session(session_id)
        denial = self._check_policy("browser_snapshot")
        if denial is not None:
            raise PermissionError(f"policy_denied: {denial}")
        page = session.get_page(page_id)
        if page is None:
            raise ValueError(f"session {session_id!r} has no page {page_id!r}")
        target_page_id = page_id if page_id is not None else session.active_page_id
        assert target_page_id is not None

        started = time.monotonic()
        page_snapshot, ref_map = await capture_snapshot(page, page_id=target_page_id)
        session.ref_map = ref_map
        session.touch()
        self._emit(
            session_id=session_id,
            action="snapshot",
            status="success",
            duration_ms=(time.monotonic() - started) * 1000,
            page_id=target_page_id,
            url_after=page_snapshot.url,
        )
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

    async def _run_ref_action(
        self,
        session_id: str,
        ref: str,
        action: str,
        call: Callable[[Locator], Awaitable[object]],
        *,
        redacted_note: str | None = None,
    ) -> InteractionResult:
        denial = self._check_policy(f"browser_{action}")
        if denial is not None:
            return InteractionResult(status=InteractionStatus.POLICY_DENIED, action=action, error=denial)

        target_label = redacted_note if redacted_note is not None else ref
        session = self._sessions.get(session_id)
        started = time.monotonic()
        try:
            locator = self.resolve_ref(session_id, ref)
        except StaleRefError as exc:
            self._emit(
                session_id=session_id,
                action=action,
                status="stale_ref",
                duration_ms=(time.monotonic() - started) * 1000,
                target=target_label,
                error=str(exc),
            )
            return InteractionResult(status=InteractionStatus.STALE_REF, action=action, error=str(exc))
        except ValueError as exc:
            self._emit(
                session_id=session_id,
                action=action,
                status="not_found",
                duration_ms=(time.monotonic() - started) * 1000,
                target=target_label,
                error=str(exc),
            )
            return InteractionResult(status=InteractionStatus.NOT_FOUND, action=action, error=str(exc))

        try:
            await call(locator)
        except Exception as exc:
            status = classify_playwright_error(exc)
            self._emit(
                session_id=session_id,
                action=action,
                status=status.value,
                duration_ms=(time.monotonic() - started) * 1000,
                target=target_label,
                error=str(exc),
            )
            return InteractionResult(status=status, action=action, error=str(exc))

        self._emit(
            session_id=session_id,
            action=action,
            status="success",
            duration_ms=(time.monotonic() - started) * 1000,
            target=target_label,
            page_id=session.active_page_id if session is not None else None,
        )
        return InteractionResult(status=InteractionStatus.SUCCESS, action=action)

    async def click(
        self, session_id: str, ref: str, *, timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS
    ) -> InteractionResult:
        return await self._run_ref_action(
            session_id, ref, "click", lambda locator: locator.click(timeout=timeout * 1000)
        )

    async def type_text(
        self, session_id: str, ref: str, text: str, *, timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS
    ) -> InteractionResult:
        """Types character-by-character (dispatches real key events) —
        for a plain input/textarea/contenteditable value set, prefer
        `fill`, which is faster and doesn't rely on key-event handlers."""
        return await self._run_ref_action(
            session_id,
            ref,
            "type",
            lambda locator: locator.press_sequentially(text, timeout=timeout * 1000),
            redacted_note=redact_typed_value(text),
        )

    async def fill(
        self, session_id: str, ref: str, text: str, *, timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS
    ) -> InteractionResult:
        return await self._run_ref_action(
            session_id,
            ref,
            "fill",
            lambda locator: locator.fill(text, timeout=timeout * 1000),
            redacted_note=redact_typed_value(text),
        )

    async def press(
        self, session_id: str, ref: str, key: str, *, timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS
    ) -> InteractionResult:
        """`key` is a Playwright key name/combo: "Enter", "Tab",
        "Escape", "Control+A", etc."""
        return await self._run_ref_action(
            session_id, ref, "press", lambda locator: locator.press(key, timeout=timeout * 1000)
        )

    async def select_option(
        self,
        session_id: str,
        ref: str,
        *,
        value: str | None = None,
        label: str | None = None,
        timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS,
    ) -> InteractionResult:
        if value is None and label is None:
            return InteractionResult(status=InteractionStatus.ERROR, action="select", error="need value or label")

        async def _select(locator: Locator) -> object:
            if value is not None:
                return await locator.select_option(value=value, timeout=timeout * 1000)
            return await locator.select_option(label=label, timeout=timeout * 1000)

        return await self._run_ref_action(session_id, ref, "select", _select)

    async def scroll_into_view(
        self, session_id: str, ref: str, *, timeout: float = _DEFAULT_ACTION_TIMEOUT_SECONDS
    ) -> InteractionResult:
        return await self._run_ref_action(
            session_id, ref, "scroll", lambda locator: locator.scroll_into_view_if_needed(timeout=timeout * 1000)
        )

    async def wait_for(
        self,
        session_id: str,
        *,
        url_contains: str | None = None,
        text: str | None = None,
        load_state: str | None = None,
        timeout: float = 30.0,
    ) -> InteractionResult:
        """Spec section 30: wait on a real signal (URL substring, text
        appearing, load state) instead of an arbitrary sleep()."""
        session = self._require_open_session(session_id)
        denial = self._check_policy("browser_wait")
        if denial is not None:
            return InteractionResult(status=InteractionStatus.POLICY_DENIED, action="wait", error=denial)
        page = session.get_page()
        if page is None:
            return InteractionResult(status=InteractionStatus.NOT_FOUND, action="wait", error="no active page")
        try:
            if url_contains is not None:
                await page.wait_for_url(f"**{url_contains}**", timeout=timeout * 1000)
            elif text is not None:
                await page.get_by_text(text).first.wait_for(timeout=timeout * 1000)
            elif load_state is not None:
                await page.wait_for_load_state(load_state, timeout=timeout * 1000)  # type: ignore[arg-type]
            else:
                return InteractionResult(
                    status=InteractionStatus.ERROR, action="wait", error="need url_contains, text, or load_state"
                )
        except Exception as exc:
            return InteractionResult(status=classify_playwright_error(exc), action="wait", error=str(exc))
        return InteractionResult(status=InteractionStatus.SUCCESS, action="wait")

    async def evaluate(
        self,
        session_id: str,
        script: str,
        *,
        page_id: str | None = None,
        timeout: float = 30.0,
        artifacts_dir: Path | None = None,
    ) -> EvaluateResult:
        """Spec section 29: JS evaluation, output-bounded the same way
        execute_code bounds Python output — a large return value gets a
        truncated preview plus an artifact pointer, never a raw dump."""
        import asyncio

        session = self._require_open_session(session_id)
        denial = self._check_policy("browser_evaluate")
        if denial is not None:
            return EvaluateResult(
                status=InteractionStatus.POLICY_DENIED, preview="", truncated=False, total_chars=0, error=denial
            )
        page = session.get_page(page_id)
        if page is None:
            return EvaluateResult(status=InteractionStatus.NOT_FOUND, preview="", truncated=False, total_chars=0)
        try:
            result = await asyncio.wait_for(page.evaluate(script), timeout=timeout)
        except TimeoutError:
            return EvaluateResult(
                status=InteractionStatus.TIMEOUT, preview="", truncated=False, total_chars=0, error="evaluate timed out"
            )
        except Exception as exc:
            return EvaluateResult(
                status=classify_playwright_error(exc), preview="", truncated=False, total_chars=0, error=str(exc)
            )
        return bound_evaluate_result(result, artifacts_dir=artifacts_dir)

    async def upload(self, session_id: str, ref: str, file_paths: list[str]) -> InteractionResult:
        return await self._run_ref_action(
            session_id, ref, "upload", lambda locator: locator.set_input_files(file_paths)
        )

    async def download_via_click(
        self, session_id: str, ref: str, *, artifacts_dir: Path, timeout: float = 30.0
    ) -> DownloadResult:
        """Spec section 31: clicking `ref` is expected to trigger a
        browser download — waits for it and saves it as a real artifact
        with provenance (path/filename/mime/size/sha256), rather than
        trusting anything the page claims about the file."""
        session = self._require_open_session(session_id)
        denial = self._check_policy("browser_download")
        if denial is not None:
            return DownloadResult(ok=False, error=f"policy_denied: {denial}")
        page = session.get_page()
        if page is None:
            return DownloadResult(ok=False, error="no active page")
        try:
            locator = self.resolve_ref(session_id, ref)
        except (StaleRefError, ValueError) as exc:
            return DownloadResult(ok=False, error=str(exc))

        try:
            async with page.expect_download(timeout=timeout * 1000) as download_info:
                await locator.click()
            download = await download_info.value
        except Exception as exc:
            return DownloadResult(ok=False, error=str(exc))

        return await save_download(download, artifacts_dir=artifacts_dir)

    def list_pages(self, session_id: str) -> list[dict[str, str | bool]]:
        session = self._require_open_session(session_id)
        return [
            {"page_id": page_id, "url": page.url, "active": page_id == session.active_page_id}
            for page_id, page in session.pages.items()
        ]

    async def new_page(self, session_id: str, *, url: str | None = None) -> str:
        session = self._require_open_session(session_id)
        page = await session.context.new_page()
        page_id = session.add_page(page, make_active=True)
        if url is not None:
            await page.goto(url)
        session.touch()
        return page_id

    def switch_page(self, session_id: str, page_id: str) -> bool:
        session = self._require_open_session(session_id)
        if page_id not in session.pages:
            return False
        session.active_page_id = page_id
        session.touch()
        return True

    async def close_page(self, session_id: str, page_id: str) -> bool:
        session = self._require_open_session(session_id)
        page = session.pages.get(page_id)
        if page is None:
            return False
        await page.close()
        session.remove_page(page_id)
        return True


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
