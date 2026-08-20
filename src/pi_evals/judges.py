"""LLM-as-judge scoring primitive, mirroring vitest-evals' ``createJudge``.

Framework-agnostic (no ``AgentSession``/``pi_harness`` dependency) — pairs
naturally with a ``pytest-evals`` ``@pytest.mark.eval`` test, but doesn't
require it. Candidate for eventual upstream contribution to pytest-evals,
since nothing here is Pi-specific; kept local until it's proven out.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["Judge", "JudgeScore", "ScoreFn", "create_judge"]


@dataclass
class JudgeScore:
    """A judge's verdict: a 0..1 score plus optional free-text reasoning
    (e.g. the judge model's own explanation, useful in failure output)."""

    score: float
    reasoning: str | None = None


ScoreFn = Callable[..., "JudgeScore | float | Awaitable[JudgeScore | float]"]


@dataclass
class Judge:
    """A named, reusable scorer with a pass/fail threshold.

    ``threshold=None`` makes this an *observation* judge (vitest-evals'
    ``judgeThreshold: null``): :meth:`check` never raises, so a low score is
    recorded without failing the eval — appropriate for comparative eval
    sets where a candidate is expected to sometimes lose to the baseline.
    """

    name: str
    score_fn: ScoreFn
    threshold: float | None = 1.0

    async def __call__(self, **context: Any) -> JudgeScore:
        """Run the scorer against keyword context (e.g. output=, expected=, input=)."""
        result = self.score_fn(**context)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, JudgeScore):
            return result
        return JudgeScore(score=float(result))

    def check(self, result: JudgeScore) -> None:
        """Raise AssertionError if result.score is below threshold.

        No-op when threshold is None — call this from a hard-assertion test
        for suite invariants; skip it (just record the score) for
        observational/comparative measurements.
        """
        if self.threshold is None:
            return
        if result.score < self.threshold:
            detail = f": {result.reasoning}" if result.reasoning else ""
            raise AssertionError(
                f"Judge {self.name!r} scored {result.score:.3f} < threshold {self.threshold:.3f}{detail}"
            )

    async def score_and_check(self, **context: Any) -> JudgeScore:
        """Convenience: score, then check() the result. Returns the score either way."""
        result = await self(**context)
        self.check(result)
        return result


def create_judge(name: str, score_fn: ScoreFn, *, threshold: float | None = 1.0) -> Judge:
    """Create a :class:`Judge` — mirrors ``createJudge(name, scoreFn)`` from vitest-evals."""
    return Judge(name=name, score_fn=score_fn, threshold=threshold)
