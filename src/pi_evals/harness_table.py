"""Comparative baseline/candidate eval sets, mirroring vitest-evals'
``evalHarnessTable`` + the pass-rate lift reporting built on top of it.

Two independent pieces, matching how the original TS harness split them:

- :func:`harness_table` — a pure grid generator. Produces one row per
  (harness x repetition) combination; feed it straight into
  ``@pytest.mark.parametrize`` so pytest-evals' eval phase runs every row as
  its own case, collecting results into ``eval_bag``.
- :func:`compute_pass_rate_lift` — run in the ``@pytest.mark.eval_analysis``
  phase over ``eval_results``, once every row has finished. Computes each
  candidate's pass-rate lift over the baseline (candidate minus baseline, in
  percentage points), plus average token/latency/cost deltas.

Accessors (``harness_name``, ``passed``, ``tokens``, ...) are plain callables
rather than a fixed result type, so this stays agnostic of whatever shape a
caller's ``eval_bag``/``eval_results`` rows happen to have.

Simplification vs. the original: deltas here are *unpaired* — average of the
candidate's values minus average of the baseline's, not a per-matched-input
paired delta. A paired version needs a stable per-input grouping key threaded
through eval_bag, which is left for a caller that needs that precision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CandidateLift",
    "HarnessTableRow",
    "compute_pass_rate_lift",
    "harness_table",
    "passed_from_score",
]


@dataclass
class HarnessTableRow:
    """One row of a comparative eval grid: run `harness` (named `name`) for
    the given `repetition` index."""

    name: str
    harness: Any
    repetition: int


def harness_table(
    *,
    baseline: Any,
    candidate: Any | None = None,
    candidates: dict[str, Any] | None = None,
    repetitions: int = 1,
) -> list[HarnessTableRow]:
    """Build the (harness x repetition) grid for a comparative eval set.

    Pass exactly one of `candidate` (single treatment, named "candidate") or
    `candidates` (named treatments, e.g. `{"gpt-5": h1, "opus": h2}`) —
    mirrors evalHarnessTable's "Use candidate for one treatment or
    candidates for multiple treatments."
    """
    if candidate is not None and candidates is not None:
        raise ValueError("Pass either candidate= (one) or candidates= (many), not both.")
    resolved_candidates = candidates or ({"candidate": candidate} if candidate is not None else {})
    if not resolved_candidates:
        raise ValueError("harness_table requires at least one candidate (via candidate= or candidates=).")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1.")

    rows: list[HarnessTableRow] = []
    for repetition in range(repetitions):
        rows.append(HarnessTableRow(name="baseline", harness=baseline, repetition=repetition))
        for name, harness in resolved_candidates.items():
            rows.append(HarnessTableRow(name=name, harness=harness, repetition=repetition))
    return rows


def passed_from_score(score: float) -> bool:
    """A judge score of at least 1 counts as passing — matches the original
    methodology's 'treating a score of at least 1 as passing'."""
    return score >= 1


@dataclass
class CandidateLift:
    candidate: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    lift_pp: float
    n_baseline: int
    n_candidate: int
    token_delta: float | None = None
    latency_ms_delta: float | None = None
    cost_delta: float | None = None


def _pass_rate(rows: Sequence[Any], passed: Callable[[Any], bool]) -> float:
    return sum(1 for r in rows if passed(r)) / len(rows)


def _avg_delta(
    baseline_rows: Sequence[Any],
    candidate_rows: Sequence[Any],
    accessor: Callable[[Any], float | None] | None,
) -> float | None:
    if accessor is None:
        return None
    baseline_values = [v for v in (accessor(r) for r in baseline_rows) if v is not None]
    candidate_values = [v for v in (accessor(r) for r in candidate_rows) if v is not None]
    if not baseline_values or not candidate_values:
        return None
    return (sum(candidate_values) / len(candidate_values)) - (sum(baseline_values) / len(baseline_values))


def compute_pass_rate_lift(
    rows: Sequence[Any],
    *,
    harness_name: Callable[[Any], str],
    passed: Callable[[Any], bool],
    baseline_name: str = "baseline",
    tokens: Callable[[Any], float | None] | None = None,
    latency_ms: Callable[[Any], float | None] | None = None,
    cost: Callable[[Any], float | None] | None = None,
) -> list[CandidateLift]:
    """Compute each candidate's pass-rate lift over the baseline from a flat
    sequence of result rows (typically ``eval_results`` in the analysis
    phase). Returns one :class:`CandidateLift` per non-baseline harness name
    present in `rows`; empty list if there's no baseline data to compare
    against.
    """
    by_harness: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_harness[harness_name(row)].append(row)

    baseline_rows = by_harness.get(baseline_name, [])
    if not baseline_rows:
        return []
    baseline_pass_rate = _pass_rate(baseline_rows, passed)

    lifts: list[CandidateLift] = []
    for name in sorted(by_harness):
        if name == baseline_name:
            continue
        candidate_rows = by_harness[name]
        if not candidate_rows:
            continue
        candidate_pass_rate = _pass_rate(candidate_rows, passed)
        lifts.append(
            CandidateLift(
                candidate=name,
                baseline_pass_rate=baseline_pass_rate,
                candidate_pass_rate=candidate_pass_rate,
                lift_pp=(candidate_pass_rate - baseline_pass_rate) * 100,
                n_baseline=len(baseline_rows),
                n_candidate=len(candidate_rows),
                token_delta=_avg_delta(baseline_rows, candidate_rows, tokens),
                latency_ms_delta=_avg_delta(baseline_rows, candidate_rows, latency_ms),
                cost_delta=_avg_delta(baseline_rows, candidate_rows, cost),
            )
        )
    return lifts
