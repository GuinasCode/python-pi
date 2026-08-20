"""Tests for pi_evals.judges."""

from __future__ import annotations

import asyncio

import pytest

from pi_evals.judges import JudgeScore, create_judge


class TestJudgeScoring:
    def test_sync_scorer_returning_float(self) -> None:
        judge = create_judge("exact_match", lambda output, expected: 1.0 if output == expected else 0.0)
        result = asyncio.run(judge(output="paris", expected="paris"))
        assert result.score == 1.0
        assert result.reasoning is None

    def test_sync_scorer_returning_judge_score_with_reasoning(self) -> None:
        judge = create_judge("with_reasoning", lambda **_: JudgeScore(score=0.5, reasoning="partial match"))
        result = asyncio.run(judge(output="x"))
        assert result.score == 0.5
        assert result.reasoning == "partial match"

    def test_async_scorer_is_awaited(self) -> None:
        async def _score(**_: object) -> float:
            return 0.75

        judge = create_judge("async_judge", _score)
        result = asyncio.run(judge())
        assert result.score == 0.75


class TestJudgeThreshold:
    def test_check_passes_when_score_meets_threshold(self) -> None:
        judge = create_judge("t", lambda **_: 1.0, threshold=0.8)
        judge.check(JudgeScore(score=0.9))  # does not raise

    def test_check_raises_when_score_below_threshold(self) -> None:
        judge = create_judge("t", lambda **_: 0.0, threshold=0.8)
        with pytest.raises(AssertionError, match=r"scored 0\.500 < threshold 0\.800"):
            judge.check(JudgeScore(score=0.5))

    def test_check_error_includes_reasoning(self) -> None:
        judge = create_judge("t", lambda **_: 0.0, threshold=0.8)
        with pytest.raises(AssertionError, match="too vague"):
            judge.check(JudgeScore(score=0.1, reasoning="too vague"))

    def test_threshold_none_is_observation_only_and_never_raises(self) -> None:
        judge = create_judge("observational", lambda **_: 0.0, threshold=None)
        judge.check(JudgeScore(score=0.0))  # does not raise despite score 0


class TestScoreAndCheck:
    def test_returns_score_and_raises_on_failure(self) -> None:
        judge = create_judge("t", lambda **_: 0.2, threshold=0.8)
        with pytest.raises(AssertionError):
            asyncio.run(judge.score_and_check(output="x"))

    def test_returns_score_when_passing(self) -> None:
        judge = create_judge("t", lambda **_: 0.9, threshold=0.8)
        result = asyncio.run(judge.score_and_check(output="x"))
        assert result.score == 0.9
