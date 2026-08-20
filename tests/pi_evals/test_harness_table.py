"""Tests for pi_evals.harness_table."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pi_evals.harness_table import compute_pass_rate_lift, harness_table, passed_from_score


class TestHarnessTable:
    def test_single_candidate_grid(self) -> None:
        rows = harness_table(baseline="B", candidate="C", repetitions=2)
        names_reps = [(r.name, r.repetition) for r in rows]
        assert names_reps == [
            ("baseline", 0),
            ("candidate", 0),
            ("baseline", 1),
            ("candidate", 1),
        ]
        assert all(r.harness in ("B", "C") for r in rows)

    def test_multiple_named_candidates(self) -> None:
        rows = harness_table(baseline="B", candidates={"gpt": "G", "opus": "O"}, repetitions=1)
        names = {r.name for r in rows}
        assert names == {"baseline", "gpt", "opus"}
        assert len(rows) == 3

    def test_default_repetitions_is_one(self) -> None:
        rows = harness_table(baseline="B", candidate="C")
        assert len(rows) == 2

    def test_rejects_both_candidate_and_candidates(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            harness_table(baseline="B", candidate="C", candidates={"x": "X"})

    def test_rejects_neither_candidate_nor_candidates(self) -> None:
        with pytest.raises(ValueError, match="at least one candidate"):
            harness_table(baseline="B")

    def test_rejects_zero_repetitions(self) -> None:
        with pytest.raises(ValueError, match="repetitions must be"):
            harness_table(baseline="B", candidate="C", repetitions=0)


class TestPassedFromScore:
    def test_score_at_or_above_one_passes(self) -> None:
        assert passed_from_score(1.0) is True
        assert passed_from_score(1.5) is True

    def test_score_below_one_fails(self) -> None:
        assert passed_from_score(0.99) is False
        assert passed_from_score(0.0) is False


@dataclass
class _Row:
    harness: str
    score: float
    tokens: float | None = None
    latency_ms: float | None = None
    cost: float | None = None


class TestComputePassRateLift:
    def test_no_baseline_rows_returns_empty(self) -> None:
        rows = [_Row(harness="candidate", score=1.0)]
        lifts = compute_pass_rate_lift(rows, harness_name=lambda r: r.harness, passed=lambda r: r.score >= 1)
        assert lifts == []

    def test_candidate_beats_baseline(self) -> None:
        rows = [
            _Row(harness="baseline", score=0.0),
            _Row(harness="baseline", score=1.0),
            _Row(harness="candidate", score=1.0),
            _Row(harness="candidate", score=1.0),
        ]
        lifts = compute_pass_rate_lift(rows, harness_name=lambda r: r.harness, passed=lambda r: r.score >= 1)
        assert len(lifts) == 1
        lift = lifts[0]
        assert lift.candidate == "candidate"
        assert lift.baseline_pass_rate == 0.5
        assert lift.candidate_pass_rate == 1.0
        assert lift.lift_pp == pytest.approx(50.0)
        assert lift.n_baseline == 2
        assert lift.n_candidate == 2

    def test_multiple_candidates_each_get_a_row(self) -> None:
        rows = [
            _Row(harness="baseline", score=0.0),
            _Row(harness="gpt", score=1.0),
            _Row(harness="opus", score=0.0),
        ]
        lifts = compute_pass_rate_lift(rows, harness_name=lambda r: r.harness, passed=lambda r: r.score >= 1)
        by_name = {lift.candidate: lift for lift in lifts}
        assert by_name["gpt"].lift_pp == pytest.approx(100.0)
        assert by_name["opus"].lift_pp == pytest.approx(0.0)

    def test_token_latency_cost_deltas_are_unpaired_averages(self) -> None:
        rows = [
            _Row(harness="baseline", score=1.0, tokens=100, latency_ms=500, cost=0.01),
            _Row(harness="baseline", score=1.0, tokens=200, latency_ms=700, cost=0.02),
            _Row(harness="candidate", score=1.0, tokens=150, latency_ms=400, cost=0.005),
        ]
        lifts = compute_pass_rate_lift(
            rows,
            harness_name=lambda r: r.harness,
            passed=lambda r: r.score >= 1,
            tokens=lambda r: r.tokens,
            latency_ms=lambda r: r.latency_ms,
            cost=lambda r: r.cost,
        )
        lift = lifts[0]
        assert lift.token_delta == pytest.approx(150 - 150)  # candidate avg 150, baseline avg 150
        assert lift.latency_ms_delta == pytest.approx(400 - 600)
        assert lift.cost_delta == pytest.approx(0.005 - 0.015)

    def test_deltas_are_none_when_accessor_not_provided(self) -> None:
        rows = [
            _Row(harness="baseline", score=1.0),
            _Row(harness="candidate", score=1.0),
        ]
        lifts = compute_pass_rate_lift(rows, harness_name=lambda r: r.harness, passed=lambda r: r.score >= 1)
        assert lifts[0].token_delta is None
        assert lifts[0].latency_ms_delta is None
        assert lifts[0].cost_delta is None

    def test_missing_telemetry_values_are_skipped_not_treated_as_zero(self) -> None:
        rows = [
            _Row(harness="baseline", score=1.0, tokens=None),
            _Row(harness="baseline", score=1.0, tokens=100),
            _Row(harness="candidate", score=1.0, tokens=50),
        ]
        lifts = compute_pass_rate_lift(
            rows, harness_name=lambda r: r.harness, passed=lambda r: r.score >= 1, tokens=lambda r: r.tokens
        )
        # baseline avg should be 100 (the None is excluded), not 50 (which
        # treating None as 0 would produce).
        assert lifts[0].token_delta == pytest.approx(50 - 100)
