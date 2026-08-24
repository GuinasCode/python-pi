"""Tests for pi_runtime.router.ModelRouter. Covers Fase 10's acceptance
criteria from plan.md section 14:

- routing pode ser testado sem chamadas reais
- fallback é determinístico
- budget impede overspend
- provider indisponível é classificado
"""

from __future__ import annotations

from pi_ai import Model, ModelCost
from pi_ai.models import MutableModels, Provider
from pi_runtime.router import ModelRouter, TaskType, Tier, classify_model_tier, estimate_cost
from pi_runtime.state import Budget


def _cheap_model(model_id: str = "cheap-1") -> Model:
    return Model(
        id=model_id, provider="test", reasoning=False, context_window=8_000, cost=ModelCost(input=0.1, output=0.2)
    )


def _strong_model(model_id: str = "strong-1") -> Model:
    return Model(
        id=model_id, provider="test", reasoning=True, context_window=200_000, cost=ModelCost(input=5.0, output=15.0)
    )


def _medium_model(model_id: str = "medium-1") -> Model:
    return Model(
        id=model_id, provider="test", reasoning=True, context_window=32_000, cost=ModelCost(input=1.0, output=2.0)
    )


def _models_with(*models: Model) -> MutableModels:
    mm = MutableModels()
    mm.set_provider(Provider(id="test", name="Test", models=list(models)))
    return mm


class TestClassifyModelTier:
    def test_no_reasoning_small_context_is_cheap(self) -> None:
        assert classify_model_tier(_cheap_model()) == Tier.CHEAP

    def test_reasoning_and_large_context_is_strong(self) -> None:
        assert classify_model_tier(_strong_model()) == Tier.STRONG

    def test_reasoning_alone_is_medium(self) -> None:
        assert classify_model_tier(_medium_model()) == Tier.MEDIUM


class TestEstimateCost:
    def test_zero_tokens_costs_nothing(self) -> None:
        assert estimate_cost(_strong_model(), input_tokens=0, output_tokens=0) == 0.0

    def test_cost_scales_with_tokens_and_rate(self) -> None:
        model = Model(id="x", cost=ModelCost(input=2.0, output=4.0))
        cost = estimate_cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 6.0


class TestRoutingWithoutRealCalls:
    """Fase 10 acceptance criterion 1 — every test here is a pure
    function over MutableModels populated with plain Model objects, no
    network/provider call anywhere."""

    def test_classification_routes_to_a_cheap_model(self) -> None:
        models = _models_with(_cheap_model(), _strong_model())
        router = ModelRouter(models)
        decision = router.route(TaskType.CLASSIFICATION)
        assert decision.model is not None
        assert decision.actual_tier == Tier.CHEAP
        assert not decision.fallback_used

    def test_research_routes_to_a_strong_model(self) -> None:
        models = _models_with(_cheap_model(), _strong_model())
        router = ModelRouter(models)
        decision = router.route(TaskType.RESEARCH)
        assert decision.model is not None
        assert decision.actual_tier == Tier.STRONG


class TestDeterministicFallback:
    def test_missing_strong_tier_falls_back_to_medium(self) -> None:
        models = _models_with(_medium_model())
        router = ModelRouter(models)
        decision = router.route(TaskType.RESEARCH)  # wants STRONG
        assert decision.model is not None
        assert decision.actual_tier == Tier.MEDIUM
        assert decision.fallback_used

    def test_fallback_order_is_stable_across_repeated_calls(self) -> None:
        models = _models_with(_cheap_model(), _medium_model())
        router = ModelRouter(models)
        decisions = [router.route(TaskType.RESEARCH) for _ in range(5)]
        tiers = {d.actual_tier for d in decisions}
        assert tiers == {Tier.MEDIUM}  # always picks the same fallback tier, never varies


class TestProviderUnavailableIsClassified:
    def test_no_models_registered_at_all(self) -> None:
        router = ModelRouter(MutableModels())
        decision = router.route(TaskType.RESEARCH)
        assert decision.model is None
        assert decision.unavailable_reason is not None
        assert "no registered model" in decision.unavailable_reason

    def test_reason_is_never_silently_none_when_model_is_none(self) -> None:
        router = ModelRouter(MutableModels())
        decision = router.route(TaskType.CLASSIFICATION)
        assert decision.model is None
        assert decision.unavailable_reason


class TestBudgetPreventsOverspend:
    def test_expensive_model_skipped_when_over_budget(self) -> None:
        models = _models_with(_strong_model(), _cheap_model())
        router = ModelRouter(models)
        budget = Budget(max_cost=0.01)

        # strong model @ these token counts costs 0.2 (over budget); the
        # cheap model costs 0.003 (comfortably within it).
        decision = router.route(
            TaskType.RESEARCH, estimated_input_tokens=10_000, estimated_output_tokens=10_000, budget=budget
        )

        # research wants STRONG, but the strong model would blow the
        # budget — must fall back to the cheap one instead of overspending
        assert decision.model is not None
        assert decision.actual_tier == Tier.CHEAP
        assert decision.estimated_cost <= budget.max_cost

    def test_no_model_fits_the_budget_is_explicit(self) -> None:
        models = _models_with(_strong_model())
        router = ModelRouter(models)
        budget = Budget(max_cost=0.0001)

        decision = router.route(
            TaskType.RESEARCH, estimated_input_tokens=1_000_000, estimated_output_tokens=1_000_000, budget=budget
        )
        assert decision.model is None
        assert decision.unavailable_reason is not None
        assert "budget" in decision.unavailable_reason

    def test_already_consumed_cost_counts_toward_the_check(self) -> None:
        models = _models_with(_cheap_model())
        router = ModelRouter(models)
        budget = Budget(max_cost=0.001)
        budget.record_usage(cost=0.001)  # already exhausted

        decision = router.route(TaskType.CLASSIFICATION, estimated_input_tokens=1000, budget=budget)
        assert decision.model is None
        assert decision.unavailable_reason is not None

    def test_no_budget_given_never_blocks_a_selection(self) -> None:
        models = _models_with(_strong_model())
        router = ModelRouter(models)
        decision = router.route(
            TaskType.RESEARCH, estimated_input_tokens=10_000_000, estimated_output_tokens=10_000_000
        )
        assert decision.model is not None


class TestCustomTaskTiers:
    def test_caller_can_override_the_default_routing_table(self) -> None:
        models = _models_with(_cheap_model(), _strong_model())
        router = ModelRouter(models, task_tiers={TaskType.GENERAL: Tier.STRONG})
        decision = router.route(TaskType.GENERAL)
        assert decision.actual_tier == Tier.STRONG
