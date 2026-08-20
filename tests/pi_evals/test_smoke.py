"""Smoke eval: port of packages/evals/src/smoke.eval.ts.

Runs a real prompt against a real LLM — unlike the rest of the test suite
(hermetic via the faux provider), this needs PI_PROVIDER/PI_MODEL set to an
actually-configured model and costs a real API call. pytest-evals only
collects @pytest.mark.eval tests when invoked with --run-eval (and skips
the rest of the suite in that mode), so a plain `pytest`/`uv run pytest`
run never triggers this — use `pi-evals --provider ... --model ...` (or
`pytest --run-eval --run-eval-analysis`) to actually run it.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from pi_evals import create_pi_coding_agent_harness

# noTools: "all" in the original — this harness never sees a tool call, so
# the smoke test only exercises the model round-trip, not tool execution.
_harness = create_pi_coding_agent_harness(no_tools=True)


@pytest.mark.eval(name="pi_coding_agent_smoke")
def test_runs_a_basic_prompt_end_to_end(eval_bag: Any) -> None:
    result = asyncio.run(_harness.run("What's the capital of France? Respond with only the city name."))

    eval_bag.output = result.output
    eval_bag.usage = result.usage
    eval_bag.passed = result.output.strip() == "Paris"

    assert result.output.strip() == "Paris"
    assert result.usage.provider == os.environ.get("PI_PROVIDER")
    assert result.usage.model == os.environ.get("PI_MODEL")
    assert result.usage.total_tokens > 0


@pytest.mark.eval_analysis(name="pi_coding_agent_smoke")
def test_pi_coding_agent_smoke_analysis(eval_results: Any) -> None:
    assert len(eval_results) > 0
    passed = sum(1 for r in eval_results if r.result.passed)
    print(f"Pi coding agent smoke: {passed}/{len(eval_results)} passed")
    assert passed == len(eval_results)
