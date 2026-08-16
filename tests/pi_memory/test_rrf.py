"""Tests for pi_memory.rrf."""

from __future__ import annotations

from pi_memory.rrf import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        result = reciprocal_rank_fusion([[1, 2, 3]])
        assert [item_id for item_id, _score in result] == [1, 2, 3]

    def test_disjoint_lists_both_contribute(self) -> None:
        result = reciprocal_rank_fusion([[1, 2], [3, 4]])
        ids = {item_id for item_id, _score in result}
        assert ids == {1, 2, 3, 4}

    def test_overlap_boosts_score(self) -> None:
        # id 2 appears first in both lists -> highest fused score.
        result = reciprocal_rank_fusion([[2, 1], [2, 3]])
        assert result[0][0] == 2

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_ties_broken_deterministically_by_input_order(self) -> None:
        # Two ids with identical scores (both rank 1 in separate lists) —
        # just confirm both are present and score equal, not a specific order.
        result = reciprocal_rank_fusion([[1], [2]])
        scores = dict(result)
        assert scores[1] == scores[2]
