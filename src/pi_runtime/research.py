"""Research Engine — Fase 4 of the research-first-runtime plan.

Smallest real vertical slice, per PROMPT.md section 6 / plan.md Passo C
("não implemente 'Research Platform' inteira... primeiro ResearchTask ->
provider -> evidence -> result. Depois evolua"):

    ResearchTask -> ExtractProvider -> Evidence -> ResearchResult

ExtractProvider wraps the existing, already-tested
pi_coding_agent.tools.fetch_url — not a second HTTP client. There is no
real web-search API configured anywhere in this repo (no search API key),
so a query planner/search provider that turns a question into URLs would
have to fabricate results to demonstrate anything — that's exactly what
Regra 1.3 ("não use mocks como produto") forbids. This phase's scope is
therefore evidence-gathering with real provenance and honest coverage
reporting (including recognizing "não há evidência suficiente" instead
of inventing certainty), taking a caller-supplied list of URLs; a real
SearchProvider that decomposes a question into queries is a TODO for
when a real search backend is configured (Regra 1.5 — registered here,
not silently built as a fake).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from pi_coding_agent.tools import ToolResult, fetch_url
from pi_runtime.state import VerificationResult


@dataclass
class Evidence:
    """plan.md section 8 — minimum fields for provenance."""

    source_id: str
    url: str
    title: str
    excerpt: str
    retrieved_at: float
    relevance: float = 1.0
    reliability: float = 1.0
    extraction_method: str = "fetch_url"


@dataclass
class Claim:
    """plan.md section 8. Populated by whatever turns evidence into
    claims (an LLM synthesis step, needing a real model call) — this
    phase's ResearchEngine doesn't synthesize claims itself (see module
    docstring); Claim exists here so ResearchVerifier has a real contract
    to check once something does produce them."""

    text: str
    claim_type: str = "factual"
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supported: bool = False
    contradicted: bool = False
    unresolved: bool = True


@dataclass
class ResearchTask:
    question: str
    urls: list[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    question: str
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    failed_urls: list[str] = field(default_factory=list)
    coverage_note: str = ""


class ExtractProvider(Protocol):
    def extract(self, url: str) -> Evidence | None: ...


class ToolExtractProvider:
    """The real extraction path: wraps pi_coding_agent.tools.fetch_url
    (already tested against httpx.MockTransport-style fixtures in
    tests/pi_coding_agent/test_tools.py) rather than reimplementing HTTP
    fetching. Returns None on any fetch error — the caller (ResearchEngine)
    tracks that as a failed URL, not a crash."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def extract(self, url: str) -> Evidence | None:
        result: ToolResult = fetch_url(url, timeout=self._timeout)
        if result.is_error or not result.content:
            return None
        text = result.content[0].get("text", "")
        return Evidence(
            source_id=url,
            url=url,
            title=url,
            excerpt=text[:500],
            retrieved_at=time.time(),
            extraction_method="fetch_url",
        )


class ResearchEngine:
    """Question -> [Extract per URL] -> Assess coverage -> Report. The
    full plan.md pipeline (Decompose -> Plan -> Search -> Extract ->
    Assess -> Identify gaps -> Search again -> Synthesize -> Verify ->
    Report) needs a real search provider and an LLM synthesis step,
    neither of which exist yet here — this implements the Extract/Assess/
    Report slice of it for a caller-supplied URL list, honestly."""

    def __init__(self, *, extract_provider: ExtractProvider | None = None) -> None:
        self._extract_provider = extract_provider or ToolExtractProvider()

    def research(self, task: ResearchTask) -> ResearchResult:
        evidence: list[Evidence] = []
        failed: list[str] = []
        for url in task.urls:
            item = self._extract_provider.extract(url)
            if item is None:
                failed.append(url)
            else:
                evidence.append(item)

        if not evidence:
            coverage = "no evidence gathered — insufficient information to answer the question"
        elif failed:
            coverage = f"evidence gathered from {len(evidence)}/{len(task.urls)} requested sources"
        else:
            coverage = f"evidence gathered from all {len(evidence)} requested sources"

        return ResearchResult(question=task.question, evidence=evidence, failed_urls=failed, coverage_note=coverage)


class ResearchVerifier:
    """plan.md section 10 (research verifier) and section 6's rule:
    "Nenhuma afirmação factual importante no resultado final deve ser
    marcada como suportada sem evidência." Reuses
    pi_runtime.state.VerificationResult (Fase 1) rather than a second
    verification concept — same pattern as Fase 3's verify_tool_result."""

    def verify(self, result: ResearchResult) -> VerificationResult:
        unsupported = [claim.text for claim in result.claims if claim.supported and not claim.evidence_refs]
        if unsupported:
            return VerificationResult(
                passed=False,
                score=0.0,
                unsupported_claims=unsupported,
                recommended_repair="attach evidence to every supported claim, or mark it unresolved instead",
            )
        if not result.evidence and not result.claims:
            # Honestly reporting "no evidence" is not a failure — inventing
            # certainty in its place would be (plan.md section 6).
            return VerificationResult(passed=True, score=0.3, missing_requirements=["no evidence gathered"])
        return VerificationResult(passed=True, score=1.0)


__all__ = [
    "Claim",
    "Evidence",
    "ExtractProvider",
    "ResearchEngine",
    "ResearchResult",
    "ResearchTask",
    "ResearchVerifier",
    "ToolExtractProvider",
]
