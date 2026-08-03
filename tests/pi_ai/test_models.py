"""Tests for pi_ai models module - Provider and Models collection."""

from __future__ import annotations

import pytest

from pi_ai import Model, ModelCost
from pi_ai.models import (
    Credential,
    InMemoryCredentialStore,
    InMemoryModelsStore,
    InMemoryProviderModelsStore,
    ModelsError,
    Provider,
    create_models,
)


def _make_model(model_id: str = "test-model", provider: str = "test-provider") -> Model:
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url="https://api.test.com/v1",
        context_window=4096,
        max_tokens=2048,
        cost=ModelCost(input=0.01, output=0.02),
    )


class TestProvider:
    def test_provider_creation(self) -> None:
        provider = Provider(id="test", name="Test Provider")
        assert provider.id == "test"
        assert provider.name == "Test Provider"
        assert provider.base_url is None
        assert provider.headers == {}
        assert provider.get_models() == []

    def test_provider_with_models(self) -> None:
        model = _make_model()
        provider = Provider(id="test", name="Test", models=[model])
        assert len(provider.get_models()) == 1
        assert provider.get_models()[0].id == "test-model"

    def test_provider_stream_without_fn_raises(self) -> None:
        provider = Provider(id="test", name="Test")
        model = _make_model()
        with pytest.raises(NotImplementedError):
            provider.stream(model, None, None)


class TestMutableModels:
    def test_set_and_get_provider(self) -> None:
        models = create_models()
        provider = Provider(id="test", name="Test")
        models.set_provider(provider)
        assert models.get_provider("test") is provider
        assert len(models.get_providers()) == 1

    def test_delete_provider(self) -> None:
        models = create_models()
        models.set_provider(Provider(id="test", name="Test"))
        models.delete_provider("test")
        assert models.get_provider("test") is None
        assert len(models.get_providers()) == 0

    def test_clear_providers(self) -> None:
        models = create_models()
        models.set_provider(Provider(id="a", name="A"))
        models.set_provider(Provider(id="b", name="B"))
        models.clear_providers()
        assert len(models.get_providers()) == 0

    def test_get_models_all(self) -> None:
        models = create_models()
        m1 = _make_model("m1", "p1")
        m2 = _make_model("m2", "p2")
        models.set_provider(Provider(id="p1", name="P1", models=[m1]))
        models.set_provider(Provider(id="p2", name="P2", models=[m2]))
        all_models = models.get_models()
        assert len(all_models) == 2

    def test_get_models_by_provider(self) -> None:
        models = create_models()
        m1 = _make_model("m1", "p1")
        m2 = _make_model("m2", "p2")
        models.set_provider(Provider(id="p1", name="P1", models=[m1]))
        models.set_provider(Provider(id="p2", name="P2", models=[m2]))
        assert len(models.get_models("p1")) == 1
        assert models.get_models("p1")[0].id == "m1"

    def test_get_model(self) -> None:
        models = create_models()
        model = _make_model("test-model", "test-provider")
        models.set_provider(Provider(id="test-provider", name="Test", models=[model]))
        found = models.get_model("test-provider", "test-model")
        assert found is not None
        assert found.id == "test-model"
        assert models.get_model("test-provider", "nonexistent") is None
        assert models.get_model("nonexistent", "test-model") is None

    def test_stream_unknown_provider_raises(self) -> None:
        models = create_models()
        model = _make_model(provider="unknown")
        with pytest.raises(ModelsError, match="Unknown provider"):
            models.stream(model, None, None)


class TestInMemoryStores:
    def test_credential_store(self) -> None:
        store = InMemoryCredentialStore()
        assert store.get("p1") is None
        cred = Credential(type="apiKey", api_key="key123")
        store.set("p1", cred)
        assert store.get("p1") is cred
        store.delete("p1")
        assert store.get("p1") is None

    def test_models_store(self) -> None:
        store = InMemoryModelsStore()
        ps = store.get_provider_store("p1")
        assert isinstance(ps, InMemoryProviderModelsStore)
        assert ps.get() == []
        model = _make_model()
        ps.set([model])
        assert len(ps.get()) == 1
