"""Tests for pi_coding_agent.model_resolver module."""

from __future__ import annotations

import pytest

from pi_ai import Model
from pi_coding_agent.model_resolver import (
    DEFAULT_THINKING_LEVEL,
    ScopedModel,
    build_fallback_model,
    find_exact_model_reference_match,
    find_initial_model,
    is_alias,
    models_are_equal,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope,
    resolve_model_scope_with_diagnostics,
    restore_model_from_session,
    try_match_model,
)

# --- Test fixtures ---


def make_model(
    model_id: str,
    *,
    provider: str = "anthropic",
    name: str = "",
    api: str = "anthropic-messages",
    reasoning: bool = False,
) -> Model:
    return Model(
        id=model_id,
        name=name or model_id,
        api=api,
        provider=provider,
        reasoning=reasoning,
    )


class MockModelRuntime:
    """Mock model runtime for testing."""

    def __init__(
        self,
        models: list[Model] | None = None,
        configured_providers: set[str] | None = None,
    ) -> None:
        self._models = models or []
        self._configured = configured_providers or set()

    def get_available(self) -> list[Model]:
        return [m for m in self._models if m.provider in self._configured]

    def get_models(self) -> list[Model]:
        return list(self._models)

    def has_configured_auth(self, provider: str) -> bool:
        return provider in self._configured

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return next((m for m in self._models if m.provider == provider and m.id == model_id), None)


@pytest.fixture
def sample_models() -> list[Model]:
    return [
        make_model("claude-opus-4-8", provider="anthropic"),
        make_model("claude-sonnet-4-5", provider="anthropic"),
        make_model("claude-sonnet-4-5-20250929", provider="anthropic"),
        make_model("gpt-5.5", provider="openai"),
        make_model("gpt-5.5-mini", provider="openai"),
        make_model("glm-5.1", provider="zai"),
        make_model("kimi-k2.6", provider="openrouter", name="Kimi K2.6"),
    ]


@pytest.fixture
def configured_runtime(sample_models: list[Model]) -> MockModelRuntime:
    return MockModelRuntime(sample_models, {"anthropic", "openai", "zai", "openrouter"})


# --- models_are_equal ---


class TestModelsAreEqual:
    def test_equal(self) -> None:
        a = make_model("gpt-4", provider="openai")
        b = make_model("gpt-4", provider="openai")
        assert models_are_equal(a, b)

    def test_different_id(self) -> None:
        a = make_model("gpt-4", provider="openai")
        b = make_model("gpt-5", provider="openai")
        assert not models_are_equal(a, b)

    def test_different_provider(self) -> None:
        a = make_model("gpt-4", provider="openai")
        b = make_model("gpt-4", provider="azure")
        assert not models_are_equal(a, b)


# --- is_alias ---


class TestIsAlias:
    def test_latest_is_alias(self) -> None:
        assert is_alias("claude-sonnet-4-5-latest")

    def test_dated_is_not_alias(self) -> None:
        assert not is_alias("claude-sonnet-4-5-20250929")

    def test_plain_name_is_alias(self) -> None:
        assert is_alias("gpt-5.5")

    def test_plain_name_with_suffix(self) -> None:
        assert is_alias("claude-opus-4-8")


# --- find_exact_model_reference_match ---


class TestFindExactModelReference:
    def test_bare_id(self, sample_models: list[Model]) -> None:
        result = find_exact_model_reference_match("claude-opus-4-8", sample_models)
        assert result is not None
        assert result.id == "claude-opus-4-8"

    def test_canonical_reference(self, sample_models: list[Model]) -> None:
        result = find_exact_model_reference_match("anthropic/claude-opus-4-8", sample_models)
        assert result is not None
        assert result.id == "claude-opus-4-8"
        assert result.provider == "anthropic"

    def test_case_insensitive(self, sample_models: list[Model]) -> None:
        result = find_exact_model_reference_match("Anthropic/Claude-Opus-4-8", sample_models)
        assert result is not None
        assert result.id == "claude-opus-4-8"

    def test_no_match(self, sample_models: list[Model]) -> None:
        result = find_exact_model_reference_match("nonexistent-model", sample_models)
        assert result is None

    def test_empty_input(self, sample_models: list[Model]) -> None:
        assert find_exact_model_reference_match("", sample_models) is None
        assert find_exact_model_reference_match("   ", sample_models) is None


# --- try_match_model ---


class TestTryMatchModel:
    def test_exact_match(self, sample_models: list[Model]) -> None:
        result = try_match_model("claude-opus-4-8", sample_models)
        assert result is not None
        assert result.id == "claude-opus-4-8"

    def test_partial_match_prefers_alias(self, sample_models: list[Model]) -> None:
        result = try_match_model("claude-sonnet", sample_models)
        assert result is not None
        assert result.id == "claude-sonnet-4-5"  # alias preferred over dated

    def test_partial_match_by_name(self, sample_models: list[Model]) -> None:
        result = try_match_model("Kimi", sample_models)
        assert result is not None
        assert result.id == "kimi-k2.6"

    def test_no_match(self, sample_models: list[Model]) -> None:
        result = try_match_model("nonexistent", sample_models)
        assert result is None


# --- parse_model_pattern ---


class TestParseModelPattern:
    def test_exact_match(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern("claude-opus-4-8", sample_models)
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.thinking_level is None
        assert result.warning is None

    def test_thinking_level_suffix(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern("claude-opus-4-8:high", sample_models)
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.thinking_level == "high"

    def test_off_thinking_level(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern("claude-opus-4-8:off", sample_models)
        assert result.model is not None
        assert result.thinking_level == "off"

    def test_invalid_thinking_level_warns(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern("claude-opus-4-8:invalid", sample_models)
        assert result.model is not None
        assert result.thinking_level is None
        assert "Invalid thinking level" in (result.warning or "")

    def test_no_match(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern("nonexistent", sample_models)
        assert result.model is None
        assert result.warning is None

    def test_strict_mode_no_fallback(self, sample_models: list[Model]) -> None:
        result = parse_model_pattern(
            "claude-opus-4-8:invalid",
            sample_models,
            {"allowInvalidThinkingLevelFallback": False},
        )
        # In strict mode, invalid suffix means the whole pattern doesn't match
        assert result.model is None


# --- resolve_model_scope ---


class TestResolveModelScope:
    def test_simple_pattern(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["claude-opus-4-8"], configured_runtime)
        assert len(result) == 1
        assert result[0].model.id == "claude-opus-4-8"

    def test_multiple_patterns(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["claude-opus-4-8", "gpt-5.5"], configured_runtime)
        assert len(result) == 2
        assert result[0].model.id == "claude-opus-4-8"
        assert result[1].model.id == "gpt-5.5"

    def test_glob_pattern(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["anthropic/*"], configured_runtime)
        # Should match all anthropic models
        assert len(result) == 3
        providers = {sm.model.provider for sm in result}
        assert providers == {"anthropic"}

    def test_glob_with_thinking_level(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["anthropic/*:high"], configured_runtime)
        assert len(result) > 0
        assert all(sm.thinking_level == "high" for sm in result)

    def test_thinking_level_suffix(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["claude-opus-4-8:medium"], configured_runtime)
        assert len(result) == 1
        assert result[0].thinking_level == "medium"

    def test_no_match_diagnostic(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope_with_diagnostics(["nonexistent"], configured_runtime)
        assert len(result.scoped_models) == 0
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].code == "no-match"

    def test_duplicate_models_deduped(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["claude-opus-4-8", "claude-opus-4-8"], configured_runtime)
        assert len(result) == 1

    def test_partial_match(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_model_scope(["claude-sonnet"], configured_runtime)
        assert len(result) == 1
        assert "sonnet" in result[0].model.id


# --- resolve_cli_model ---


class TestResolveCliModel:
    def test_provider_and_model(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_provider="anthropic",
            cli_model="claude-opus-4-8",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.error is None

    def test_provider_model_slash_format(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_model="anthropic/claude-opus-4-8",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.model.provider == "anthropic"

    def test_case_insensitive_provider(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_provider="Anthropic",
            cli_model="claude-opus-4-8",
            model_runtime=configured_runtime,
        )
        assert result.model is not None

    def test_unknown_provider(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_provider="unknown-provider",
            cli_model="some-model",
            model_runtime=configured_runtime,
        )
        assert result.model is None
        assert "Unknown provider" in (result.error or "")

    def test_no_model_specified(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(model_runtime=configured_runtime)
        assert result.model is None
        assert result.error is None

    def test_no_models_available(self) -> None:
        runtime = MockModelRuntime([], set())
        result = resolve_cli_model(cli_model="gpt-4", model_runtime=runtime)
        assert result.model is None
        assert "No models available" in (result.error or "")

    def test_model_not_found_with_provider(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_provider="anthropic",
            cli_model="nonexistent-model",
            model_runtime=configured_runtime,
        )
        # Should build a fallback model
        assert result.model is not None
        assert result.model.id == "nonexistent-model"
        assert result.model.provider == "anthropic"
        assert "not found" in (result.warning or "")

    def test_model_not_found_no_provider(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_model="nonexistent-model",
            model_runtime=configured_runtime,
        )
        assert result.model is None
        assert "not found" in (result.error or "")

    def test_thinking_level_suffix(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_model="claude-opus-4-8:high",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.thinking_level == "high"

    def test_exact_id_match_without_provider(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_model="gpt-5.5",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "gpt-5.5"

    def test_provider_prefix_tolerated(self, configured_runtime: MockModelRuntime) -> None:
        result = resolve_cli_model(
            cli_provider="anthropic",
            cli_model="anthropic/claude-opus-4-8",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"


# --- build_fallback_model ---


class TestBuildFallbackModel:
    def test_builds_model_with_custom_id(self, sample_models: list[Model]) -> None:
        result = build_fallback_model("anthropic", "custom-model", sample_models)
        assert result is not None
        assert result.id == "custom-model"
        assert result.name == "custom-model"
        assert result.provider == "anthropic"

    def test_no_provider_models(self, sample_models: list[Model]) -> None:
        result = build_fallback_model("unknown", "custom-model", sample_models)
        assert result is None


# --- find_initial_model ---


class TestFindInitialModel:
    def test_cli_args_priority(self, configured_runtime: MockModelRuntime) -> None:
        result = find_initial_model(
            cli_provider="anthropic",
            cli_model="claude-opus-4-8",
            scoped_models=[],
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"

    def test_cli_args_error_raises(self, configured_runtime: MockModelRuntime) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            find_initial_model(
                cli_provider="unknown",
                cli_model="some-model",
                scoped_models=[],
                model_runtime=configured_runtime,
            )

    def test_scoped_models_when_not_continuing(self, configured_runtime: MockModelRuntime) -> None:
        scoped = [ScopedModel(model=make_model("claude-opus-4-8"), thinking_level="high")]
        result = find_initial_model(
            scoped_models=scoped,
            is_continuing=False,
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.thinking_level == "high"

    def test_scoped_models_skipped_when_continuing(self, configured_runtime: MockModelRuntime) -> None:
        scoped = [ScopedModel(model=make_model("claude-opus-4-8"), thinking_level="high")]
        result = find_initial_model(
            scoped_models=scoped,
            is_continuing=True,
            model_runtime=configured_runtime,
        )
        # Should skip scoped models and fall through to available
        assert result.model is not None

    def test_default_from_settings(self, configured_runtime: MockModelRuntime) -> None:
        result = find_initial_model(
            scoped_models=[],
            default_provider="anthropic",
            default_model_id="claude-opus-4-8",
            default_thinking_level="low",
            model_runtime=configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.thinking_level == "low"

    def test_first_available_fallback(self, configured_runtime: MockModelRuntime) -> None:
        result = find_initial_model(
            scoped_models=[],
            model_runtime=configured_runtime,
        )
        assert result.model is not None

    def test_no_model_available(self) -> None:
        runtime = MockModelRuntime([], set())
        result = find_initial_model(
            scoped_models=[],
            model_runtime=runtime,
        )
        assert result.model is None
        assert result.thinking_level == DEFAULT_THINKING_LEVEL


# --- restore_model_from_session ---


class TestRestoreModelFromSession:
    def test_restore_success(self, configured_runtime: MockModelRuntime) -> None:
        result = restore_model_from_session(
            "anthropic",
            "claude-opus-4-8",
            None,
            False,
            configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "claude-opus-4-8"
        assert result.fallback_message is None

    def test_restore_with_current_model_fallback(self, configured_runtime: MockModelRuntime) -> None:
        current = make_model("gpt-5.5", provider="openai")
        result = restore_model_from_session(
            "unknown-provider",
            "nonexistent",
            current,
            False,
            configured_runtime,
        )
        assert result.model is not None
        assert result.model.id == "gpt-5.5"
        assert result.fallback_message is not None
        assert "Could not restore" in result.fallback_message

    def test_restore_finds_available(self, configured_runtime: MockModelRuntime) -> None:
        result = restore_model_from_session(
            "unknown-provider",
            "nonexistent",
            None,
            False,
            configured_runtime,
        )
        assert result.model is not None
        assert result.fallback_message is not None

    def test_restore_no_models_available(self) -> None:
        runtime = MockModelRuntime([], set())
        result = restore_model_from_session(
            "unknown",
            "nonexistent",
            None,
            False,
            runtime,
        )
        assert result.model is None
