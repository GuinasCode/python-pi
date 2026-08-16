"""Reciprocal Rank Fusion for combining ranked result lists.

Used to merge FTS5 (keyword) and sqlite-vec (semantic) search results
into a single ranking without needing to normalize incomparable scores
(BM25 vs. cosine/L2 distance).
"""

from __future__ import annotations

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = DEFAULT_K,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked id lists into one, ordered by fused score desc.

    Each input list is ids ordered best-to-worst. An id's score is the sum
    of ``1 / (k + rank)`` (1-indexed rank) over every list it appears in.
    ``k=60`` is the standard constant from Cormack et al., chosen so a
    single first-place ranking doesn't dominate the fused result.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
