"""Model resolution, scoping, and initial selection.

Mirrors packages/coding-agent/src/core/model-resolver.ts.

Resolves which model/provider to use from settings, CLI args, and env vars.
Supports pattern matching with glob patterns, thinking level suffixes, and
canonical ``provider/modelId`` references.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pi_ai import Model
from pi_coding_agent import is_valid_thinking_level

DEFAULT_THINKING_LEVEL = "medium"

# Default model IDs for each known provider (subset relevant to the Python port)
default_model_per_provider: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure-openai-responses": "gpt-5.4",
    "openai-codex": "gpt-5.5",
    "radius": "auto",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.5",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.7",
    "zai": "glm-5.1",
    "zai-coding-cn": "glm-5.1",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k2p6",
    "together": "moonshotai/Kimi-K2.6",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k2.6",
    "kimi-coding": "kimi-for-coding",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
}


# --- Model equality ---


def models_are_equal(a: Model, b: Model) -> bool:
    """Check if two models are equal (same provider and id)."""
    return a.provider == b.provider and a.id == b.id


# --- Helpers ---


def is_alias(model_id: str) -> bool:
    """Check if a model ID looks like an alias (no date suffix).

    Dates are typically in the format -20241022 or -20250929.
    """
    if model_id.endswith("-latest"):
        return True
    # Check if ID ends with a date pattern (-YYYYMMDD)
    return not bool(re.search(r"-\d{8}$", model_id))


def find_exact_model_reference_match(
    model_reference: str,
    available_models: list[Model],
) -> Model | None:
    """Find an exact model reference match.

    Supports either a bare model id or a canonical provider/modelId reference.
    When matching by bare id, ambiguous matches across providers are rejected.
    """
    trimmed = model_reference.strip()
    if not trimmed:
        return None
    normalized = trimmed.lower()

    # Try canonical provider/modelId match
    canonical_matches = [m for m in available_models if f"{m.provider}/{m.id}".lower() == normalized]
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        return None

    # Try provider/modelId split
    slash_index = trimmed.find("/")
    if slash_index != -1:
        provider = trimmed[:slash_index].strip()
        model_id = trimmed[slash_index + 1 :].strip()
        if provider and model_id:
            provider_matches = [
                m
                for m in available_models
                if m.provider.lower() == provider.lower() and m.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    # Try bare id match
    id_matches = [m for m in available_models if m.id.lower() == normalized]
    return id_matches[0] if len(id_matches) == 1 else None


def try_match_model(model_pattern: str, available_models: list[Model]) -> Model | None:
    """Try to match a pattern to a model from available models.

    Returns the matched model or None if no match found.
    Falls back to partial matching on id or name.
    """
    exact = find_exact_model_reference_match(model_pattern, available_models)
    if exact:
        return exact

    pattern_lower = model_pattern.lower()
    matches = [
        m
        for m in available_models
        if m.id.lower().__contains__(pattern_lower) or (m.name or "").lower().__contains__(pattern_lower)
    ]
    if not matches:
        return None

    aliases = [m for m in matches if is_alias(m.id)]
    dated = [m for m in matches if not is_alias(m.id)]

    if aliases:
        aliases.sort(key=lambda m: m.id, reverse=True)
        return aliases[0]
    dated.sort(key=lambda m: m.id, reverse=True)
    return dated[0]


# --- Parsed model result ---


@dataclass
class ParsedModelResult:
    """Result from parsing a model pattern."""

    model: Model | None = None
    thinking_level: str | None = None
    warning: str | None = None


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    options: dict[str, Any] | None = None,
) -> ParsedModelResult:
    """Parse a pattern to extract model and thinking level.

    Handles models with colons in their IDs (e.g., OpenRouter's :exacto suffix).

    Algorithm:
    1. Try to match full pattern as a model
    2. If found, return it with no explicit thinking level
    3. If not found and has colons, split on last colon:
       - If suffix is valid thinking level, use it and recurse on prefix
       - If suffix is invalid, warn and recurse on prefix
    """
    # Try exact match first
    exact_match = try_match_model(pattern, available_models)
    if exact_match:
        return ParsedModelResult(model=exact_match, thinking_level=None, warning=None)

    # No match - try splitting on last colon if present
    last_colon_index = pattern.rfind(":")
    if last_colon_index == -1:
        return ParsedModelResult(model=None, thinking_level=None, warning=None)

    prefix = pattern[:last_colon_index]
    suffix = pattern[last_colon_index + 1 :]

    if is_valid_thinking_level(suffix):
        # Valid thinking level - recurse on prefix and use this level
        result = parse_model_pattern(prefix, available_models, options)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None if result.warning else suffix,
                warning=result.warning,
            )
        return result

    # Invalid suffix
    allow_fallback = (options or {}).get("allowInvalidThinkingLevelFallback", True)
    if not allow_fallback:
        return ParsedModelResult(model=None, thinking_level=None, warning=None)

    # Scope mode: recurse on prefix and warn
    result = parse_model_pattern(prefix, available_models, options)
    if result.model:
        return ParsedModelResult(
            model=result.model,
            thinking_level=None,
            warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


# --- Model scope resolution ---


@dataclass
class ScopedModel:
    """A model resolved from a pattern, with optional thinking level."""

    model: Model
    thinking_level: str | None = None


@dataclass
class ModelScopeDiagnostic:
    """A diagnostic from model scope resolution."""

    type: str  # always "warning"
    code: str  # "no-match" | "invalid-thinking-level"
    message: str
    pattern: str


@dataclass
class ResolveModelScopeResult:
    """Result from resolving model scope patterns."""

    scoped_models: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


def _has_glob_chars(pattern: str) -> bool:
    """Check if a pattern contains glob characters."""
    return "*" in pattern or "?" in pattern or "[" in pattern


class ModelRuntimeLike(Protocol):
    """Protocol for the model runtime needed by model resolution."""

    def get_available(self) -> list[Model]: ...
    def get_models(self) -> list[Model]: ...
    def has_configured_auth(self, provider: str) -> bool: ...
    def get_model(self, provider: str, model_id: str) -> Model | None: ...


def resolve_model_scope_with_diagnostics(
    patterns: list[str],
    model_runtime: ModelRuntimeLike,
) -> ResolveModelScopeResult:
    """Resolve model patterns to ScopedModel objects with diagnostics."""
    available_models = list(model_runtime.get_available())
    scoped_models: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    for pattern in patterns:
        # Check if pattern contains glob characters
        if _has_glob_chars(pattern):
            # Extract optional thinking level suffix
            colon_idx = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: str | None = None

            if colon_idx != -1:
                suffix = pattern[colon_idx + 1 :]
                if is_valid_thinking_level(suffix):
                    thinking_level = suffix
                    glob_pattern = pattern[:colon_idx]

            exact_match = find_exact_model_reference_match(glob_pattern, available_models)
            if exact_match:
                if not any(models_are_equal(sm.model, exact_match) for sm in scoped_models):
                    scoped_models.append(ScopedModel(model=exact_match, thinking_level=thinking_level))
                continue

            # Match against "provider/modelId" format OR just model ID
            matching_models = [
                m
                for m in available_models
                if fnmatch.fnmatch(f"{m.provider}/{m.id}", glob_pattern) or fnmatch.fnmatch(m.id, glob_pattern)
            ]

            if not matching_models:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        type="warning",
                        code="no-match",
                        message=f'No models match pattern "{pattern}"',
                        pattern=pattern,
                    )
                )
                continue

            for model in matching_models:
                if not any(models_are_equal(sm.model, model) for sm in scoped_models):
                    scoped_models.append(ScopedModel(model=model, thinking_level=thinking_level))
            continue

        result = parse_model_pattern(pattern, available_models)

        if result.warning:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type="warning",
                    code="invalid-thinking-level",
                    message=result.warning,
                    pattern=pattern,
                )
            )

        if not result.model:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type="warning",
                    code="no-match",
                    message=f'No models match pattern "{pattern}"',
                    pattern=pattern,
                )
            )
            continue

        # Avoid duplicates
        if not any(models_are_equal(sm.model, result.model) for sm in scoped_models):
            scoped_models.append(ScopedModel(model=result.model, thinking_level=result.thinking_level))

    return ResolveModelScopeResult(scoped_models=scoped_models, diagnostics=diagnostics)


def resolve_model_scope(patterns: list[str], model_runtime: ModelRuntimeLike) -> list[ScopedModel]:
    """Resolve model patterns to ScopedModel objects, printing warnings to stderr."""
    result = resolve_model_scope_with_diagnostics(patterns, model_runtime)
    for diagnostic in result.diagnostics:
        import sys

        print(f"Warning: {diagnostic.message}", file=sys.stderr)
    return result.scoped_models


# --- CLI model resolution ---


@dataclass
class ResolveCliModelResult:
    """Result from resolving a single model from CLI flags."""

    model: Model | None = None
    thinking_level: str | None = None
    warning: str | None = None
    error: str | None = None


def build_fallback_model(
    provider: str,
    model_id: str,
    available_models: list[Model],
) -> Model | None:
    """Build a fallback model for a provider with a custom model id."""
    provider_models = [m for m in available_models if m.provider == provider]
    if not provider_models:
        return None

    default_id = default_model_per_provider.get(provider)
    if default_id:
        base_model = next((m for m in provider_models if m.id == default_id), provider_models[0])
    else:
        base_model = provider_models[0]

    # Return a copy with the custom id
    return Model(
        id=model_id,
        name=model_id,
        api=base_model.api,
        provider=base_model.provider,
        base_url=base_model.base_url,
        reasoning=base_model.reasoning,
        thinking_level_map=base_model.thinking_level_map,
        input=list(base_model.input),
        cost=base_model.cost,
        context_window=base_model.context_window,
        max_tokens=base_model.max_tokens,
        headers=dict(base_model.headers) if base_model.headers else None,
        compat=base_model.compat,
    )


def resolve_cli_model(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_thinking: str | None = None,
    model_runtime: ModelRuntimeLike,
) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags.

    Supports:
    - --provider <provider> --model <pattern>
    - --model <provider>/<pattern>
    - Fuzzy matching (same rules as model scoping)
    """
    if not cli_model:
        return ResolveCliModelResult()

    # Use all models here, not just models with pre-configured auth
    available_models = list(model_runtime.get_models())
    if not available_models:
        return ResolveCliModelResult(
            error="No models available. Check your installation or add models to models.json.",
        )

    # Build canonical provider lookup (case-insensitive)
    provider_map: dict[str, str] = {}
    for m in available_models:
        provider_map[m.provider.lower()] = m.provider

    provider = provider_map.get(cli_provider.lower()) if cli_provider else None
    if cli_provider and not provider:
        return ResolveCliModelResult(
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    # If no explicit --provider, try to interpret "provider/model" format
    pattern = cli_model
    inferred_provider = False

    if not provider:
        slash_index = cli_model.find("/")
        if slash_index != -1:
            maybe_provider = cli_model[:slash_index]
            canonical = provider_map.get(maybe_provider.lower())
            if canonical:
                provider = canonical
                pattern = cli_model[slash_index + 1 :]
                inferred_provider = True

    # If no provider inferred, try exact matches without provider inference
    if not provider:
        lower = cli_model.lower()
        exact = next(
            (m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact:
            return ResolveCliModelResult(model=exact, thinking_level=None, warning=None, error=None)

    if cli_provider and provider:
        # Tolerate --model <provider>/<pattern> by stripping the provider prefix
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix) :]

    candidates = [m for m in available_models if m.provider == provider] if provider else available_models
    result = parse_model_pattern(
        pattern,
        candidates,
        {"allowInvalidThinkingLevelFallback": False},
    )

    if result.model:
        # If provider inference matched an unauthenticated provider, prefer an exact raw model-id
        # match that is authenticated
        if inferred_provider:
            raw_exact_matches = [
                m
                for m in available_models
                if m.id.lower() == cli_model.lower() and not models_are_equal(m, result.model)
            ]
            if raw_exact_matches and not model_runtime.has_configured_auth(result.model.provider):
                authenticated_raw = [m for m in raw_exact_matches if model_runtime.has_configured_auth(m.provider)]
                if len(authenticated_raw) == 1:
                    return ResolveCliModelResult(
                        model=authenticated_raw[0], thinking_level=None, warning=None, error=None
                    )
        return ResolveCliModelResult(
            model=result.model, thinking_level=result.thinking_level, warning=result.warning, error=None
        )

    # If we inferred a provider from the slash but found no match, fall back to full input
    if inferred_provider:
        lower = cli_model.lower()
        exact = next(
            (m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact:
            return ResolveCliModelResult(model=exact, thinking_level=None, warning=None, error=None)
        fallback = parse_model_pattern(cli_model, available_models, {"allowInvalidThinkingLevelFallback": False})
        if fallback.model:
            return ResolveCliModelResult(
                model=fallback.model,
                thinking_level=fallback.thinking_level,
                warning=fallback.warning,
                error=None,
            )

    if provider:
        # Parse thinking level suffix from the pattern before building fallback
        fallback_pattern = pattern
        fallback_thinking: str | None = None
        if not cli_thinking:
            last_colon = pattern.rfind(":")
            if last_colon != -1:
                suffix = pattern[last_colon + 1 :]
                if is_valid_thinking_level(suffix):
                    fallback_pattern = pattern[:last_colon]
                    fallback_thinking = suffix

        fallback_model = build_fallback_model(provider, fallback_pattern, available_models)
        if fallback_model:
            requested_thinking = cli_thinking or fallback_thinking
            if requested_thinking and requested_thinking != "off":
                fallback_model.reasoning = True
            fallback_warning = (
                f'{result.warning} Model "{fallback_pattern}" not found for provider "{provider}". '
                f"Using custom model id."
                if result.warning
                else f'Model "{fallback_pattern}" not found for provider "{provider}". Using custom model id.'
            )
            return ResolveCliModelResult(
                model=fallback_model,
                thinking_level=fallback_thinking,
                warning=fallback_warning,
                error=None,
            )

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        model=None,
        thinking_level=None,
        warning=result.warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


# --- Initial model selection ---


@dataclass
class InitialModelResult:
    """Result from finding the initial model."""

    model: Model | None = None
    thinking_level: str = DEFAULT_THINKING_LEVEL
    fallback_message: str | None = None


def find_initial_model(
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    scoped_models: list[ScopedModel],
    is_continuing: bool = False,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: str | None = None,
    model_runtime: ModelRuntimeLike,
) -> InitialModelResult:
    """Find the initial model to use based on priority.

    Priority order:
    1. CLI args (provider + model)
    2. First model from scoped models (if not continuing/resuming)
    3. Saved default from settings
    4. First available model with valid API key
    """

    # 1. CLI args take priority
    if cli_provider and cli_model:
        resolved = resolve_cli_model(
            cli_provider=cli_provider,
            cli_model=cli_model,
            model_runtime=model_runtime,
        )
        if resolved.error:
            raise ValueError(resolved.error)
        if resolved.model:
            return InitialModelResult(
                model=resolved.model,
                thinking_level=DEFAULT_THINKING_LEVEL,
                fallback_message=None,
            )

    # 2. Use first model from scoped models (skip if continuing/resuming)
    if scoped_models and not is_continuing:
        return InitialModelResult(
            model=scoped_models[0].model,
            thinking_level=scoped_models[0].thinking_level or default_thinking_level or DEFAULT_THINKING_LEVEL,
            fallback_message=None,
        )

    # 3. Try saved default from settings if auth is configured
    if default_provider and default_model_id:
        found = model_runtime.get_model(default_provider, default_model_id)
        if found and model_runtime.has_configured_auth(found.provider):
            return InitialModelResult(
                model=found,
                thinking_level=default_thinking_level or DEFAULT_THINKING_LEVEL
                if default_thinking_level
                else DEFAULT_THINKING_LEVEL,
                fallback_message=None,
            )

    # 4. Try first available model with valid API key
    available_models = list(model_runtime.get_available())
    if available_models:
        # Try to find a default model from known providers
        for provider, default_id in default_model_per_provider.items():
            match = next((m for m in available_models if m.provider == provider and m.id == default_id), None)
            if match:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None)

        # If no default found, use first available
        return InitialModelResult(
            model=available_models[0], thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None
        )

    # 5. No model found
    return InitialModelResult(model=None, thinking_level=DEFAULT_THINKING_LEVEL, fallback_message=None)


# --- Restore model from session ---


@dataclass
class RestoreModelResult:
    """Result from restoring a model from a session."""

    model: Model | None = None
    fallback_message: str | None = None


def restore_model_from_session(
    saved_provider: str,
    saved_model_id: str,
    current_model: Model | None,
    should_print_messages: bool,
    model_runtime: ModelRuntimeLike,
) -> RestoreModelResult:
    """Restore model from session, with fallback to available models."""
    restored = model_runtime.get_model(saved_provider, saved_model_id)
    has_auth = model_runtime.has_configured_auth(restored.provider) if restored else False

    if restored and has_auth:
        if should_print_messages:
            import sys

            print(f"Restored model: {saved_provider}/{saved_model_id}", file=sys.stderr)
        return RestoreModelResult(model=restored, fallback_message=None)

    # Model not found or no API key - fall back
    reason = "model no longer exists" if not restored else "no auth configured"
    if should_print_messages:
        import sys

        print(
            f"Warning: Could not restore model {saved_provider}/{saved_model_id} ({reason}).",
            file=sys.stderr,
        )

    # If we already have a model, use it as fallback
    if current_model:
        if should_print_messages:
            import sys

            print(f"Falling back to: {current_model.provider}/{current_model.id}", file=sys.stderr)
        return RestoreModelResult(
            model=current_model,
            fallback_message=(
                f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
                f"Using {current_model.provider}/{current_model.id}."
            ),
        )

    # Try to find any available model
    available_models = list(model_runtime.get_available())
    if available_models:
        fallback_model: Model | None = None
        for provider, default_id in default_model_per_provider.items():
            match = next((m for m in available_models if m.provider == provider and m.id == default_id), None)
            if match:
                fallback_model = match
                break

        if not fallback_model:
            fallback_model = available_models[0]

        if should_print_messages:
            import sys

            print(f"Falling back to: {fallback_model.provider}/{fallback_model.id}", file=sys.stderr)

        return RestoreModelResult(
            model=fallback_model,
            fallback_message=(
                f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
                f"Using {fallback_model.provider}/{fallback_model.id}."
            ),
        )

    return RestoreModelResult(model=None, fallback_message=None)
