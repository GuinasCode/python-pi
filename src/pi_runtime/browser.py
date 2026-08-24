"""Browser Runtime — Fase 5 of the research-first-runtime plan.

Wraps the existing, already-tested browser_fetch_url tool (Playwright)
with session bookkeeping, timeouts, PolicyEngine validation (Fase 3), and
Evidence conversion (Fase 4) — "separate search/extraction/browser
automation" at the scope that's real today.

browser_fetch_url is a one-shot open -> goto -> extract -> close call,
not a persistent Playwright page kept alive across multiple tool calls.
Real click/type/submit interaction on a *live* page needs exactly that —
a session object holding an open Playwright page across calls, with the
thread-safety/async-sync bridging that requires — which does not exist
anywhere in this codebase yet. Building that is a real, separate lift;
faking "interactive" browsing on top of a one-shot fetch would violate
Regra 1.3 ("não use mocks como produto"). This phase delivers real
session bookkeeping (creation, timeout, closure, navigation history,
page -> Evidence, failure isolation) around the one-shot path. A
persistent interactive session and download tracking (browser_fetch_url
has no download capability at all today) are explicit TODOs (Regra 1.5),
not faked.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.tools import ToolResult, browser_fetch_url
from pi_runtime.research import Evidence
from pi_runtime.tools import PolicyEngine


@dataclass
class BrowserSessionInfo:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    navigations: list[str] = field(default_factory=list)
    closed: bool = False
    timeout_seconds: float = 120.0

    def touch(self) -> None:
        self.last_used_at = time.time()

    def is_expired(self) -> bool:
        return not self.closed and (time.time() - self.last_used_at) > self.timeout_seconds


@dataclass
class NavigationResult:
    session_id: str
    url: str
    ok: bool
    evidence: Evidence | None = None
    error: str | None = None
    screenshot_captured: bool = False


class BrowserManager:
    """Session bookkeeping + policy + failure isolation around the
    existing one-shot browser_fetch_url tool. A page that loads
    successfully becomes Evidence (Fase 4's contract, not a second
    content shape) with extraction_method="browser"."""

    def __init__(self, *, policy_engine: PolicyEngine | None = None, default_timeout_seconds: float = 120.0) -> None:
        self._policy_engine = policy_engine
        self._default_timeout_seconds = default_timeout_seconds
        self._sessions: dict[str, BrowserSessionInfo] = {}

    def open_session(self, *, timeout_seconds: float | None = None) -> BrowserSessionInfo:
        """Fase 5 acceptance criterion 1: "browser pode ser criado". Goes
        through PolicyEngine when one is configured (Regra: "browser deve
        obedecer ao PolicyEngine") — opt-in, same pattern as
        pi_runtime.loop.Executor's policy check."""
        if self._policy_engine is not None:
            self._policy_engine.evaluate("browser")
        session = BrowserSessionInfo(
            session_id=uuid.uuid4().hex[:8],
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds,
        )
        self._sessions[session.session_id] = session
        return session

    def close_session(self, session_id: str) -> bool:
        """Fase 5 acceptance criterion 1: "...e encerrado"."""
        session = self._sessions.get(session_id)
        if session is None or session.closed:
            return False
        session.closed = True
        return True

    def get_session(self, session_id: str) -> BrowserSessionInfo | None:
        return self._sessions.get(session_id)

    def navigate(
        self, session_id: str, url: str, *, screenshot: bool = False, timeout: float = 30.0
    ) -> NavigationResult:
        """Fase 5 acceptance criteria 2-4:

        - navegação é observável: every call (success or failure) is
          recorded on the session's `navigations` list before the fetch
          even runs.
        - páginas podem virar Evidence: a successful load returns one.
        - falhas não derrubam todo o agente: nothing here raises — a bad
          session, a closed/expired session, a Playwright exception, or
          browser_fetch_url reporting an error all become
          NavigationResult.error instead. This is also how "browser deve
          ser opcional" is honored: browser_fetch_url already reports a
          missing Playwright install as an error ToolResult rather than
          crashing, and that flows straight into NavigationResult.error
          the same way any other failure does.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="no such browser session")
        if session.closed:
            return NavigationResult(session_id=session_id, url=url, ok=False, error="browser session is closed")
        if session.is_expired():
            session.closed = True
            return NavigationResult(session_id=session_id, url=url, ok=False, error="browser session timed out")

        session.touch()
        session.navigations.append(url)

        try:
            result: ToolResult = browser_fetch_url(url, timeout=timeout, screenshot=screenshot)
        except Exception as exc:  # Playwright can raise outside ToolResult's own try/except too
            return NavigationResult(session_id=session_id, url=url, ok=False, error=str(exc))

        if result.is_error:
            error_text = result.content[0].get("text", "") if result.content else "browser navigation failed"
            return NavigationResult(session_id=session_id, url=url, ok=False, error=error_text)

        return NavigationResult(
            session_id=session_id,
            url=url,
            ok=True,
            evidence=_result_to_evidence(url, result),
            screenshot_captured=any(block.get("type") == "image" for block in result.content),
        )


def _result_to_evidence(url: str, result: ToolResult) -> Evidence:
    text = "".join(block.get("text", "") for block in result.content if block.get("type") == "text")
    details: dict[str, Any] = result.details or {}
    title = str(details.get("title") or url)
    return Evidence(
        source_id=url,
        url=url,
        title=title,
        excerpt=text[:500],
        retrieved_at=time.time(),
        extraction_method="browser",
    )


__all__ = ["BrowserManager", "BrowserSessionInfo", "NavigationResult"]
