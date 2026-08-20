"""Behavioral, model-backed evals for Pi workflows.

Port of ``packages/evals`` (TypeScript, built on ``vitest-evals``) onto
``pytest-evals``. See :mod:`pi_evals.pi_harness` for the AgentSession
adapter, :mod:`pi_evals.judges` for LLM-as-judge scoring, and
:mod:`pi_evals.harness_table` for comparative baseline/candidate eval sets.
"""

from __future__ import annotations

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
    "PiCodingAgentHarness",
    "PiCodingAgentHarnessOptions",
    "PiEvalInput",
    "PiHarnessResult",
    "PiHarnessUsage",
    "create_pi_coding_agent_harness",
    "resolve_model_selection",
]
