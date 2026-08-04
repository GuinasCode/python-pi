"""OpenAI OAuth provider for Codex-style login via browser.

Supports the OpenAI OAuth flow:
1. Open browser to authorization URL
2. User logs in
3. Receive authorization code
4. Exchange code for access token

This mirrors the OAuth provider pattern in the TypeScript Pi agent.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from pi_ai import (
    AuthCheck,
    AuthInteraction,
    Credential,
    Model,
    Provider,
    ProviderAuth,
    SimpleStreamOptions,
)
from pi_ai.event_stream import create_assistant_message_event_stream
from pi_ai.models import MutableModels


class OpenAICodexAuth:
    """OAuth-based authentication for OpenAI Codex/O1-style models."""

    AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
    TOKEN_URL = "https://auth.openai.com/oauth/token"

    def __init__(self, code_challenge: str | None = None) -> None:
        self._code_verifier = secrets.token_urlsafe(32)
        self._code_challenge = code_challenge or self._generate_challenge(self._code_verifier)

    def _generate_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        import base64
        import hashlib

        sha256 = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(sha256).rstrip(b"=").decode()

    def build_auth_url(self, redirect_uri: str, scopes: list[str] | None = None) -> str:
        """Build the OAuth authorization URL for the user to visit."""
        scopes = scopes or ["read", "write", "offline_access"]
        params = {
            "client_id": "pi-agent",
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": self._code_verifier[:8],
            "code_challenge": self._code_challenge,
            "code_challenge_method": "S256",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.AUTHORIZE_URL}?{query}"

    def exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> Credential | None:
        """Exchange authorization code for access token."""
        try:
            import httpx

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
                "client_id": "pi-agent",
            }
            response = httpx.post(self.TOKEN_URL, data=data, timeout=30)
            if response.status_code != 200:
                return None
            result = response.json()
            return Credential(
                type="oauth",
                token=result.get("access_token"),
                refresh_token=result.get("refresh_token"),
                expires_at=time.time() + result.get("expires_in", 3600),
                metadata=result,
            )
        except Exception:
            return None


class OpenAICodexAuthInteraction:
    """AuthInteraction implementation that prints URL and reads code from stdin."""

    def __init__(self, redirect_uri: str = "http://localhost:8080/callback") -> None:
        self.redirect_uri = redirect_uri
        self._auth = OpenAICodexAuth()

    def open_url(self, url: str) -> None:
        """Print the URL for the user to visit in their browser."""
        print("\n[OpenAI OAuth Login Required]", file=__import__("sys").stderr)
        print(f"Please visit: {url}", file=__import__("sys").stderr)
        print("Enter the authorization code here: ", end="", flush=True, file=__import__("sys").stderr)

    def display_code(self, code: str) -> None:
        """Display a code to the user (alternative to URL)."""
        print(f"\n[OpenAI OAuth Code: {code}]", file=__import__("sys").stderr)


class OpenAICodexAuthProvider:
    """ProviderAuth implementation for OpenAI Codex OAuth flow."""

    def __init__(self) -> None:
        self._credential: Credential | None = None

    def check(self) -> AuthCheck:
        """Check if credentials are configured."""
        if self._credential is None:
            return AuthCheck(configured=False, type="oauth", label="OpenAI OAuth")

        if self._credential.expires_at is not None:
            expiring_soon = time.time() > self._credential.expires_at - 300  # 5 min buffer
            return AuthCheck(
                configured=not expiring_soon,
                type="oauth",
                label="OpenAI OAuth (expiring soon)" if expiring_soon else "OpenAI OAuth",
                error="Token expired" if expiring_soon else None,
            )
        return AuthCheck(configured=True, type="oauth", label="OpenAI OAuth")

    def resolve(self, interaction: AuthInteraction | None = None) -> Credential | None:
        """Resolve credentials, optionally using an interaction."""
        check = self.check()
        if check.configured:
            return self._credential
        if self._credential and check.error:
            self._credential = None  # Clear expired credentials

        if interaction is None:
            return None

        auth_interaction = OpenAICodexAuthInteraction()
        interaction.open_url(
            auth_interaction._auth.build_auth_url(auth_interaction.redirect_uri)
        )

        code = input().strip()

        verif = auth_interaction._auth._code_verifier[:8]
        cred = auth_interaction._auth.exchange_code(code, verif, auth_interaction.redirect_uri)
        if cred:
            self._credential = cred
        return cred

    def login(self, interaction: AuthInteraction) -> Credential | None:
        """Initiate login flow."""
        result = self.resolve(interaction)
        if result:
            self._credential = result
        return self._credential

    def logout(self) -> None:
        """Clear stored credentials."""
        self._credential = None

    def set_credential(self, credential: Credential) -> None:
        """Set credentials programmatically (for token refresh)."""
        self._credential = credential


def openai_oauth_provider(
    model: Model,
    options: SimpleStreamOptions | None = None,
) -> tuple[Model, MutableModels, ProviderAuth]:
    """
    Create a MutableModels and ProviderAuth for OpenAI OAuth.

    Returns: (model, models_collection, auth_provider)
    Then call: auth_provider.resolve() or auth_provider.login() interactively.
    """
    models = MutableModels()
    auth = OpenAICodexAuthProvider()

    provider_model = Model(
        id=model.id,
        api="openai",
        provider="openai-oauth",
        provider_id="openai",
        context_window=128000,
        max_output_tokens=16000,
    )

    provider = Provider(
        id="openai-oauth",
        name="OpenAI OAuth (Codex)",
        models=[provider_model],
        stream_fn=lambda ctx, opts, state: _oauth_stream_fn(ctx, opts, state),
        auth=auth,
    )
    models.set_provider(provider)

    return provider_model, models, auth


async def _oauth_stream_fn(
    context: Any,
    options: SimpleStreamOptions | None,
    state: dict[str, int],
) -> None:
    """Stream function using OAuth credentials."""

    from pi_ai import (
        ErrorEvent,
    )

    stream = create_assistant_message_event_stream()

    api_key = state.get("oauth_token")
    if not api_key:
        msg = "No OAuth token available. Run auth first."
        stream.push(ErrorEvent(reason="error", error=msg))
        stream.end(msg)
        return

    # ... rest similar to openai.py streaming ...
    stream.end("OAuth token placeholder - needs full implementation")