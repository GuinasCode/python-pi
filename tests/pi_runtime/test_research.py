"""Tests for pi_runtime.research. Covers Fase 4's acceptance criteria
from plan.md section 8 that apply to this phase's real scope (evidence
gathering with provenance; claim/citation/contradiction machinery is a
TODO pending a real search+synthesis backend — see module docstring):

- fontes são preservadas
- pesquisa termina por cobertura/limite (reports coverage explicitly)
- relatório final expõe incerteza ("não há evidência suficiente")
- unsupported claims are rejected by the verifier
"""

from __future__ import annotations

import httpx
import pytest

from pi_coding_agent.tools import fetch_url
from pi_runtime.research import (
    Claim,
    Evidence,
    ResearchEngine,
    ResearchResult,
    ResearchTask,
    ResearchVerifier,
    ToolExtractProvider,
)


class _StubExtractProvider:
    """Deterministic stub for tests that don't need a real HTTP fetch —
    ToolExtractProvider itself (the real path) is exercised separately
    below via httpx mocking, same pattern tests/pi_coding_agent/
    test_tools.py already uses for fetch_url."""

    def __init__(self, answers: dict[str, Evidence | None]) -> None:
        self._answers = answers

    def extract(self, url: str) -> Evidence | None:
        return self._answers.get(url)


class TestToolExtractProvider:
    def test_wraps_the_real_fetch_url_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = httpx.Response(200, content=b"real page content", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)

        provider = ToolExtractProvider()
        evidence = provider.extract("https://example.com")

        assert evidence is not None
        assert evidence.url == "https://example.com"
        assert "real page content" in evidence.excerpt
        assert evidence.extraction_method == "fetch_url"

    def test_returns_none_on_fetch_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = httpx.Response(404, content=b"not found", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)

        provider = ToolExtractProvider()
        assert provider.extract("https://example.com/missing") is None

    def test_actually_calls_fetch_url_not_a_reimplementation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []
        original = fetch_url

        def _spy(url: str, **kwargs: object) -> object:
            calls.append(url)
            return original(url, **kwargs)  # type: ignore[arg-type]

        import pi_runtime.research as research_module

        monkeypatch.setattr(research_module, "fetch_url", _spy)
        response = httpx.Response(200, content=b"x", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)

        ToolExtractProvider().extract("https://example.com")
        assert calls == ["https://example.com"]


class TestResearchEngine:
    def test_sources_are_preserved_with_provenance(self) -> None:
        evidence = Evidence(source_id="s1", url="https://a.example", title="A", excerpt="fact one", retrieved_at=123.0)
        engine = ResearchEngine(extract_provider=_StubExtractProvider({"https://a.example": evidence}))
        result = engine.research(ResearchTask(question="what is A?", urls=["https://a.example"]))

        assert len(result.evidence) == 1
        assert result.evidence[0].url == "https://a.example"
        assert result.evidence[0].retrieved_at == 123.0

    def test_full_coverage_when_every_url_succeeds(self) -> None:
        provider = _StubExtractProvider(
            {
                "https://a.example": Evidence(
                    source_id="a", url="https://a.example", title="A", excerpt="x", retrieved_at=1.0
                ),
                "https://b.example": Evidence(
                    source_id="b", url="https://b.example", title="B", excerpt="y", retrieved_at=1.0
                ),
            }
        )
        engine = ResearchEngine(extract_provider=provider)
        result = engine.research(ResearchTask(question="q", urls=["https://a.example", "https://b.example"]))

        assert len(result.evidence) == 2
        assert result.failed_urls == []
        assert "all 2" in result.coverage_note

    def test_partial_coverage_reports_failed_urls(self) -> None:
        provider = _StubExtractProvider(
            {
                "https://a.example": Evidence(
                    source_id="a", url="https://a.example", title="A", excerpt="x", retrieved_at=1.0
                ),
                "https://broken.example": None,
            }
        )
        engine = ResearchEngine(extract_provider=provider)
        result = engine.research(ResearchTask(question="q", urls=["https://a.example", "https://broken.example"]))

        assert len(result.evidence) == 1
        assert result.failed_urls == ["https://broken.example"]
        assert "1/2" in result.coverage_note

    def test_no_evidence_is_reported_honestly_not_invented(self) -> None:
        """plan.md section 6: recognize "não há evidência suficiente"
        instead of fabricating certainty."""
        provider = _StubExtractProvider({"https://broken.example": None})
        engine = ResearchEngine(extract_provider=provider)
        result = engine.research(ResearchTask(question="q", urls=["https://broken.example"]))

        assert result.evidence == []
        assert "insufficient" in result.coverage_note

    def test_no_urls_at_all_is_also_reported_as_insufficient(self) -> None:
        engine = ResearchEngine(extract_provider=_StubExtractProvider({}))
        result = engine.research(ResearchTask(question="q", urls=[]))
        assert result.evidence == []
        assert "insufficient" in result.coverage_note


class TestResearchVerifier:
    def test_no_evidence_passes_but_flags_missing_requirement(self) -> None:
        """Honestly reporting no evidence is not a verification failure —
        inventing certainty in its place would be."""
        verifier = ResearchVerifier()
        result = ResearchResult(question="q")
        verification = verifier.verify(result)
        assert verification.passed
        assert verification.missing_requirements

    def test_supported_claim_without_evidence_refs_fails(self) -> None:
        verifier = ResearchVerifier()
        result = ResearchResult(
            question="q",
            evidence=[Evidence(source_id="s", url="u", title="t", excerpt="e", retrieved_at=1.0)],
            claims=[Claim(text="X is true", supported=True, evidence_refs=[])],
        )
        verification = verifier.verify(result)
        assert not verification.passed
        assert "X is true" in verification.unsupported_claims

    def test_supported_claim_with_evidence_refs_passes(self) -> None:
        verifier = ResearchVerifier()
        result = ResearchResult(
            question="q",
            evidence=[Evidence(source_id="s1", url="u", title="t", excerpt="e", retrieved_at=1.0)],
            claims=[Claim(text="X is true", supported=True, evidence_refs=["s1"])],
        )
        verification = verifier.verify(result)
        assert verification.passed

    def test_unresolved_claim_without_evidence_is_not_penalized(self) -> None:
        """An unresolved (not claimed as supported) claim with no
        evidence is honest, not a violation."""
        verifier = ResearchVerifier()
        result = ResearchResult(
            question="q",
            claims=[Claim(text="unclear whether X", supported=False, unresolved=True, evidence_refs=[])],
        )
        verification = verifier.verify(result)
        assert verification.passed
