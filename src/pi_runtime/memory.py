"""Cognitive Memory — Fase 7 of the research-first-runtime plan.

Does not replace pi_memory (Regra: "não substituir pi_memory sem
necessidade") — it's already a real, tested system: SQLite+FTS5+
sqlite-vec hybrid search with lexical fallback when embeddings are
unavailable, secret detection on every write, Soul-specific dedupe
(find_overlapping_soul) and semantic dedupe (find_similar). This module
adds the cognitive layer plan.md's Fase 7 asks for on top of it: a
Working/Episodic/Semantic/Procedural/User/Project taxonomy mapped onto
pi_memory's existing MemoryType values (not a competing type system), a
write policy that enforces "memória não deve ser registrada
indiscriminadamente" (a confidence gate, plus dedupe-before-write reusing
find_similar), and freshness-aware reranking of pi_memory's own search
results.

Working memory is intentionally NOT persisted here — cognitive
architectures treat working memory as short-term/session-scoped, and
that role is already filled by pi_runtime.state.AgentState.working_memory
(Fase 1); persisting it into pi_memory would blur "session scratch
space" with "curated facts across sessions," which is exactly the
distinction pi_memory's own module docstring already draws.

`confidence` is not a stored/retrievable field on MemoryRecord (adding
one would need a schema migration, the same kind of change Soul's
_migrate_add_soul_type() made — a bigger, riskier lift than this vertical
slice needs). It's used here only as a write-time gate: a memory below
the confidence threshold is refused before it ever reaches
MemoryStore.write(), which is real enforcement, not fake metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from pi_memory.store import MemoryRecord, MemoryStore, MemoryType

_DEFAULT_MIN_CONFIDENCE = 0.5


class CognitiveMemoryType(str, Enum):
    """plan.md section 11. WORKING has no storage mapping — see module
    docstring — attempting to persist it raises rather than silently
    doing something else."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    USER = "user"
    PROJECT = "project"


# Maps the cognitive taxonomy onto pi_memory's existing, unchanged
# MemoryType values — same database, same rows, just a second vocabulary
# for talking about *why* a memory exists. PROCEDURAL -> SOUL is the most
# meaningful mapping: Soul is explicitly documented (pi_memory.store) as
# "stable, high-priority, low-churn principles", which is exactly what
# procedural memory is in cognitive-architecture terms.
_COGNITIVE_TO_STORAGE_TYPE: dict[CognitiveMemoryType, MemoryType] = {
    CognitiveMemoryType.EPISODIC: MemoryType.DECISION,
    CognitiveMemoryType.SEMANTIC: MemoryType.REFERENCE,
    CognitiveMemoryType.PROCEDURAL: MemoryType.SOUL,
    CognitiveMemoryType.USER: MemoryType.USER,
    CognitiveMemoryType.PROJECT: MemoryType.PROJECT,
}


class WorkingMemoryNotPersistable(Exception):
    """Raised by write_with_policy() for CognitiveMemoryType.WORKING —
    working memory belongs on AgentState.working_memory (Fase 1), not in
    pi_memory. Refusing explicitly beats silently writing it as some
    other type would."""


@dataclass
class WriteDecision:
    """What write_with_policy() actually did, and why — so a caller (or
    a test) can tell "written", "rejected: low confidence", and
    "skipped: duplicate of #N" apart instead of just getting None back
    for every non-write outcome."""

    written: bool
    record: MemoryRecord | None = None
    reason: str = ""
    duplicate_of: MemoryRecord | None = None


def write_with_policy(
    store: MemoryStore,
    *,
    cognitive_type: CognitiveMemoryType,
    title: str,
    content: str,
    confidence: float,
    source: str = "auto",
    project_cwd: str | None = None,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
) -> WriteDecision:
    """plan.md section 11: "toda memória nova precisa: source, type,
    confidence, dedupe, sensitivity check." `source`/`type` are
    MemoryStore.write()'s existing required parameters; `sensitivity
    check` is MemoryStore.write()'s existing secret-detection guard
    (unchanged, still runs — SecretDetectedError propagates out of this
    function too, it isn't swallowed); `confidence` and `dedupe` are what
    this function adds.

    Never registers indiscriminately (plan.md's explicit rule): a memory
    below `min_confidence` is refused before any write is attempted, and
    a memory that's a close semantic duplicate of an existing one (via
    MemoryStore.find_similar, unchanged) is skipped rather than
    duplicated.
    """
    if cognitive_type is CognitiveMemoryType.WORKING:
        raise WorkingMemoryNotPersistable(
            "working memory is session-scoped (AgentState.working_memory) and is never written to pi_memory"
        )
    if confidence < min_confidence:
        return WriteDecision(written=False, reason=f"confidence {confidence:.2f} below minimum {min_confidence:.2f}")

    duplicate = store.find_similar(title, content, project_cwd=project_cwd)
    if duplicate is not None:
        existing, _distance = duplicate
        return WriteDecision(written=False, reason="duplicate of an existing memory", duplicate_of=existing)

    storage_type = _COGNITIVE_TO_STORAGE_TYPE[cognitive_type]
    record = store.write(type=storage_type, title=title, content=content, project_cwd=project_cwd, source=source)
    return WriteDecision(written=True, record=record)


@dataclass
class RankedMemory:
    record: MemoryRecord
    freshness: float
    combined_score: float


def retrieve_ranked(
    store: MemoryStore,
    query: str,
    *,
    project_cwd: str | None = None,
    top_k: int = 5,
    freshness_half_life_days: float = 30.0,
) -> list[RankedMemory]:
    """Wraps MemoryStore.search() (unchanged — still hybrid FTS+vector
    with automatic lexical fallback when embeddings are unavailable,
    "falha do semantic search degrada para lexical sem derrubar a
    sessão" is search()'s existing, already-tested behavior, not
    something this adds) with an explicit freshness score, so a stale
    match ranked highly by text/semantic relevance alone doesn't
    dominate the context (plan.md: "memória irrelevante não domina o
    contexto") — recency is folded into the final ordering instead of
    being ignored entirely."""
    now_ms = int(time.time() * 1000)
    results = store.search(query, top_k=top_k, project_cwd=project_cwd)

    half_life_ms = freshness_half_life_days * 24 * 60 * 60 * 1000
    ranked: list[RankedMemory] = []
    for position, record in enumerate(results):
        age_ms = max(0, now_ms - record.updated_at)
        freshness = 0.5 ** (age_ms / half_life_ms) if half_life_ms > 0 else 1.0
        # search() already returned these ranked by relevance (RRF) —
        # position encodes that ordering; freshness adjusts it rather
        # than replacing it.
        relevance_rank_score = 1.0 - (position / max(1, len(results)))
        ranked.append(
            RankedMemory(
                record=record, freshness=freshness, combined_score=relevance_rank_score * 0.7 + freshness * 0.3
            )
        )
    ranked.sort(key=lambda r: r.combined_score, reverse=True)
    return ranked


__all__ = [
    "CognitiveMemoryType",
    "RankedMemory",
    "WorkingMemoryNotPersistable",
    "WriteDecision",
    "retrieve_ranked",
    "write_with_policy",
]
