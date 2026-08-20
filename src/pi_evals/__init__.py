"""Behavioral, model-backed evals for Pi workflows.

Port of ``packages/evals`` (TypeScript, built on ``vitest-evals``) onto
``pytest-evals``. See :mod:`pi_evals.pi_harness` for the AgentSession
adapter, :mod:`pi_evals.judges` for LLM-as-judge scoring, and
:mod:`pi_evals.harness_table` for comparative baseline/candidate eval sets.
"""

from __future__ import annotations

from pi_evals.artifacts import DEFAULT_EVAL_DIR, EvalArtifactWriter, RunRecord
from pi_evals.harness_table import (
    CandidateLift,
    HarnessTableRow,
    compute_pass_rate_lift,
    harness_table,
    passed_from_score,
)
from pi_evals.judges import Judge, JudgeScore, create_judge
from pi_evals.pi_harness import (
    PiCodingAgentHarness,
    PiCodingAgentHarnessOptions,
    PiEvalInput,
    PiHarnessResult,
    PiHarnessUsage,
    create_pi_coding_agent_harness,
    resolve_model_selection,
)

__all__ = [
    "DEFAULT_EVAL_DIR",
    "CandidateLift",
    "EvalArtifactWriter",
    "HarnessTableRow",
    "Judge",
    "JudgeScore",
    "PiCodingAgentHarness",
    "PiCodingAgentHarnessOptions",
    "PiEvalInput",
    "PiHarnessResult",
    "PiHarnessUsage",
    "RunRecord",
    "compute_pass_rate_lift",
    "create_judge",
    "create_pi_coding_agent_harness",
    "harness_table",
    "passed_from_score",
    "resolve_model_selection",
]
