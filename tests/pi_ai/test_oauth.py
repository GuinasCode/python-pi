"""Tests for the OpenAI OAuth provider.

Guards against the module being unimportable and against basic
misconfiguration — this module previously imported names that don't exist
in pi_ai (they live in pi_ai.models), which broke `import pi_ai.oauth`
entirely with zero test coverage catching it.
"""

from __future__ import annotations

import pytest

from pi_ai.models import AuthCheck
from pi_ai.oauth import (
    OpenAICodexAuth,
    OpenAICodexAuthInteraction,
    OpenAICodexAuthProvider,
    openai_oauth_provider,
)


def test_openai_codex_auth_builds_pkce_challenge() -> None:
    auth = OpenAICodexAuth()
    assert auth.code_verifier
    assert len(auth.code_verifier) >= 32


def test_build_auth_url_state_is_independent_of_verifier() -> None:
    interaction = OpenAICodexAuthInteraction()
    url = interaction.build_auth_url()
    assert "state=" in url
    assert "code_challenge=" in url
    # The state param must not leak a prefix of the PKCE secret.
    verifier_prefix = interaction.code_verifier[:8]
    assert f"state={verifier_prefix}" not in url


def test_auth_provider_check_unconfigured_without_credential() -> None:
    provider = OpenAICodexAuthProvider()
    check = provider.check()
    assert isinstance(check, AuthCheck)
    assert check.configured is False


def test_resolve_without_interaction_returns_none() -> None:
    provider = OpenAICodexAuthProvider()
    assert provider.resolve(interaction=None) is None


def test_openai_oauth_provider_constructs_without_error() -> None:
    from pi_ai import Model

    model, models, auth = openai_oauth_provider(Model(id="gpt-test"))
    assert model.provider == "openai-oauth"
    assert models.stream is not None
    assert auth.check().configured is False


@pytest.mark.asyncio
async def test_oauth_stream_fn_errors_without_credential() -> None:
    from pi_ai import Context, Model

    model, models, _auth = openai_oauth_provider(Model(id="gpt-test"))
    stream = await models.stream(model, Context(messages=[]))
    result = await stream.result()
    assert result.stop_reason.value == "error"
    assert result.error_message
