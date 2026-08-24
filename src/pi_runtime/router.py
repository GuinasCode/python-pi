"""Model Router — Fase 10 of the research-first-runtime plan.

Separates "modelo disponível" from "modelo apropriado" (plan.md section
14). Routes by TaskType -> a desired capability Tier, picking from
whatever's actually registered in a real MutableModels (unchanged — this
is not a second model registry) with deterministic fallback: desired
tier unavailable -> next tier down (a fixed table, never random/timing
dependent) -> an explicit "no provider available" decision, never a bare
None with no explanation.

Credential pooling and provider-failure retry are not reinvented here —
pi_ai.models already has CredentialStore/InMemoryCredentialStore, and
pi_ai.providers.nvidia_models' `nvidia/auto` fallback chain already
retries the next model on a connection failure before any content has
streamed back (see that module's docstring). This module's job is
purely the missing piece: deciding *which* tier a task needs before any
of that machinery runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pi_ai import Model
from pi_ai.models import MutableModels
from pi_runtime.state import Budget


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    PLANNING = "planning"
    RESEARCH = "research"
    VERIFICATION = "verification"
    SUBAGENT = "subagent"
    CODING = "coding"
    GENERAL = "general"


class Tier(str, Enum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    STRONG = "strong"


# plan.md's own examples: classification->cheap, planning->medium/strong,
# research->strong, verification->strong, subagents->cheap/fast.
_TASK_TIER: dict[TaskType, Tier] = {
    TaskType.CLASSIFICATION: Tier.CHEAP,
    TaskType.SUBAGENT: Tier.CHEAP,
    TaskType.GENERAL: Tier.MEDIUM,
    TaskType.CODING: Tier.MEDIUM,
    TaskType.PLANNING: Tier.STRONG,
    TaskType.RESEARCH: Tier.STRONG,
    TaskType.VERIFICATION: Tier.STRONG,
}

# Fixed fallback order once the desired tier has nothing registered —
# "fallback é determinístico" (Fase 10 acceptance criterion 2).
_TIER_FALLBACK: dict[Tier, list[Tier]] = {
    Tier.STRONG: [Tier.MEDIUM, Tier.CHEAP],
    Tier.MEDIUM: [Tier.CHEAP, Tier.STRONG],
    Tier.CHEAP: [Tier.MEDIUM, Tier.STRONG],
}

# A model without reasoning and a modest context window is CHEAP; one
# with both reasoning and a large context window is STRONG; everything
# else is MEDIUM. Heuristic over Model's own declared fields (reasoning,
# context_window — set per-model already in every provider), not a
# per-call guess.
_STRONG_CONTEXT_WINDOW = 100_000


def classify_model_tier(model: Model) -> Tier:
    if model.reasoning and model.context_window >= _STRONG_CONTEXT_WINDOW:
        return Tier.STRONG
    if model.reasoning or model.context_window >= _STRONG_CONTEXT_WINDOW:
        return Tier.MEDIUM
    return Tier.CHEAP


def estimate_cost(model: Model, *, input_tokens: int, output_tokens: int) -> float:
    """Model.cost rates are $ per 1M tokens — the convention already used
    throughout pi_ai's providers (e.g. pi_ai.providers.openai.
    openai_provider's cost_input=2.50 default), reused here rather than
    redefined."""
    return (input_tokens / 1_000_000) * model.cost.input + (output_tokens / 1_000_000) * model.cost.output


@dataclass
class RoutingDecision:
    task_type: TaskType
    requested_tier: Tier
    model: Model | None = None
    actual_tier: Tier | None = None
    fallback_used: bool = False
    unavailable_reason: str | None = None
    estimated_cost: float = 0.0


class ModelRouter:
    """Deterministic routing — no network calls, no LLM-based
    classification (that would make routing itself untestable without a
    real call, which Fase 10 acceptance criterion 1 explicitly forbids)."""

    def __init__(self, models: MutableModels, *, task_tiers: dict[TaskType, Tier] | None = None) -> None:
        self._models = models
        self._task_tiers = task_tiers or dict(_TASK_TIER)

    def _models_by_tier(self, tier: Tier) -> list[Model]:
        return [model for model in self._models.get_models() if classify_model_tier(model) == tier]

    def route(
        self,
        task_type: TaskType,
        *,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
        budget: Budget | None = None,
    ) -> RoutingDecision:
        """Fase 10 acceptance criteria:

        - routing pode ser testado sem chamadas reais: a pure function
          over MutableModels' already-registered models, no I/O.
        - fallback é determinístico: _TIER_FALLBACK is a fixed table.
        - budget impede overspend: a candidate whose estimated cost would
          push consumed_cost past the budget's max_cost is skipped in
          favor of a cheaper one, in the same deterministic fallback
          order.
        - provider indisponível é classificado: no matching model at any
          tier (or budget) returns a RoutingDecision with
          unavailable_reason set, never a bare None with no explanation.
        """
        desired_tier = self._task_tiers.get(task_type, Tier.MEDIUM)
        tiers_to_try = [desired_tier, *_TIER_FALLBACK.get(desired_tier, [])]

        for tier in tiers_to_try:
            for model in self._models_by_tier(tier):
                cost = estimate_cost(model, input_tokens=estimated_input_tokens, output_tokens=estimated_output_tokens)
                if budget is not None and budget.max_cost is not None and budget.consumed_cost + cost > budget.max_cost:
                    continue
                return RoutingDecision(
                    task_type=task_type,
                    requested_tier=desired_tier,
                    model=model,
                    actual_tier=tier,
                    fallback_used=(tier != desired_tier),
                    estimated_cost=cost,
                )

        reason = "no registered model available" if not self._models.get_models() else "no model fits within budget"
        return RoutingDecision(task_type=task_type, requested_tier=desired_tier, unavailable_reason=reason)


__all__ = ["ModelRouter", "RoutingDecision", "TaskType", "Tier", "classify_model_tier", "estimate_cost"]
