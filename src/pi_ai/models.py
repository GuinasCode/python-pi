"""Provider system for pi_ai, mirroring packages/ai/src/models.ts.

This module defines the Provider and Models interfaces and a concrete
MutableModels implementation. Provider adapters (OpenAI, Anthropic, etc.)
are registered separately.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

from pi_ai import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)

TApi = TypeVar("TApi", bound=str)

# --- Auth types ---


@dataclass
class AuthCheck:
    configured: bool = False
    type: str = ""  # "apiKey" | "oauth"
    label: str = ""
    error: str | None = None


@dataclass
class Credential:
    type: str = ""  # "apiKey" | "oauth"
    api_key: str | None = None
    token: str | None = None
    refresh_token: str | None = None
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


AuthType = Literal["apiKey", "oauth"]


@dataclass
class AuthResult:
    credential: Credential | None = None
    headers: dict[str, str] = field(default_factory=dict)
    source: str = ""


class AuthInteraction(Protocol):
    def open_url(self, url: str) -> None: ...
    def display_code(self, code: str) -> None: ...


class ProviderAuth(Protocol):
    def check(self) -> AuthCheck: ...
    def resolve(self, interaction: AuthInteraction | None = None) -> Credential | None: ...
    def login(self, interaction: AuthInteraction) -> Credential: ...
    def logout(self) -> None: ...


# --- Models store ---


class ProviderModelsStore(Protocol):
    def get(self) -> list[Model]: ...
    def set(self, models: list[Model]) -> None: ...


class InMemoryProviderModelsStore:
    def __init__(self) -> None:
        self._models: list[Model] = []

    def get(self) -> list[Model]:
        return list(self._models)

    def set(self, models: list[Model]) -> None:
        self._models = list(models)


class ModelsStore(Protocol):
    def get_provider_store(self, provider_id: str) -> ProviderModelsStore: ...


class InMemoryModelsStore:
    def __init__(self) -> None:
        self._stores: dict[str, InMemoryProviderModelsStore] = {}

    def get_provider_store(self, provider_id: str) -> ProviderModelsStore:
        if provider_id not in self._stores:
            self._stores[provider_id] = InMemoryProviderModelsStore()
        return self._stores[provider_id]


# --- Credential store ---


class CredentialStore(Protocol):
    def get(self, provider_id: str) -> Credential | None: ...
    def set(self, provider_id: str, credential: Credential) -> None: ...
    def delete(self, provider_id: str) -> None: ...


class InMemoryCredentialStore:
    def __init__(self) -> None:
        self._creds: dict[str, Credential] = {}

    def get(self, provider_id: str) -> Credential | None:
        return self._creds.get(provider_id)

    def set(self, provider_id: str, credential: Credential) -> None:
        self._creds[provider_id] = credential

    def delete(self, provider_id: str) -> None:
        self._creds.pop(provider_id, None)


# --- Errors ---


class ModelsError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# --- Stream function type ---


StreamFn = Callable[[Model, Context, SimpleStreamOptions | None], Any]


# --- Provider ---


class Provider(Generic[TApi]):
    """A concrete runtime provider with auth, model listing, and stream behavior."""

    def __init__(
        self,
        id: str,
        name: str,
        *,
        base_url: str | None = None,
        headers: dict[str, str | None] | None = None,
        auth: ProviderAuth | None = None,
        models: list[Model] | None = None,
        stream_fn: StreamFn | None = None,
    ) -> None:
        self._id = id
        self._name = name
        self._base_url = base_url
        self._headers = headers or {}
        self._auth = auth
        self._models = models or []
        self._stream_fn = stream_fn

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def headers(self) -> dict[str, str | None]:
        return dict(self._headers)

    def get_models(self) -> list[Model]:
        return list(self._models)

    def stream(self, model: Model, context: Context, options: SimpleStreamOptions | None = None) -> Any:
        if self._stream_fn is None:
            raise NotImplementedError(f"Provider {self._id} has no stream function")
        return self._stream_fn(model, context, options)


# --- Models collection ---


class MutableModels:
    """Runtime collection of providers with auth resolution and stream delegation."""

    def __init__(
        self,
        *,
        credentials: CredentialStore | None = None,
        models_store: ModelsStore | None = None,
    ) -> None:
        self._providers: dict[str, Provider[str]] = {}
        self._credentials = credentials or InMemoryCredentialStore()
        self._models_store = models_store or InMemoryModelsStore()

    def get_providers(self) -> list[Provider[str]]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Provider[str] | None:
        return self._providers.get(id)

    def set_provider(self, provider: Provider[str]) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        if provider_id is not None:
            p = self._providers.get(provider_id)
            return p.get_models() if p else []
        result: list[Model] = []
        for p in self._providers.values():
            with contextlib.suppress(Exception):
                result.extend(p.get_models())
        return result

    def get_model(self, provider: str, id: str) -> Model | None:
        for model in self.get_models(provider):
            if model.id == id:
                return model
        return None

    async def refresh(
        self,
        *,
        allow_network: bool = True,
        force: bool = False,
    ) -> dict[str, str]:
        """Refresh all dynamic providers. Returns a dict of provider_id -> error message."""
        errors: dict[str, str] = {}
        for pid, provider in self._providers.items():
            refresh = getattr(provider, "refresh_models", None)
            if refresh is None:
                continue
            try:
                store = self._models_store.get_provider_store(pid)
                await refresh(
                    allow_network=allow_network,
                    force=force,
                    store=store,
                )
            except Exception as exc:
                errors[pid] = str(exc)
        return errors

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None) -> Any:
        provider = self._providers.get(model.provider)
        if provider is None:
            raise ModelsError("not_found", f"Unknown provider: {model.provider}")
        return provider.stream(model, context, options)  # type: ignore[arg-type]

    async def complete(self, model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessage:
        """Run stream to completion and return the final AssistantMessage."""
        stream = self.stream(model, context, options)
        # The stream is expected to be an async iterator producing events
        # The final event contains the completed AssistantMessage
        if hasattr(stream, "__aiter__"):
            async for event in stream:
                if hasattr(event, "message") and event.message is not None:
                    return event.message  # type: ignore[no-any-return]
                if hasattr(event, "error") and event.error is not None:
                    return event.error  # type: ignore[no-any-return]
        elif hasattr(stream, "__iter__"):
            for event in stream:
                if hasattr(event, "message") and event.message is not None:
                    return event.message  # type: ignore[no-any-return]
                if hasattr(event, "error") and event.error is not None:
                    return event.error  # type: ignore[no-any-return]
        raise ModelsError("stream", "Stream ended without a final message")


def create_models(
    *,
    credentials: CredentialStore | None = None,
    models_store: ModelsStore | None = None,
) -> MutableModels:
    """Create a new MutableModels instance."""
    return MutableModels(credentials=credentials, models_store=models_store)


# --- Auth context ---


@dataclass
class AuthContext:
    credentials: CredentialStore = field(default_factory=InMemoryCredentialStore)


def default_auth_context() -> AuthContext:
    return AuthContext()
