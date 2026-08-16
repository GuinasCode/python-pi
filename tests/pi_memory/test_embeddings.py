"""Tests for pi_memory.embeddings.

The real EmbeddingGemma-300M download/inference path is exercised only if
the model is already cached locally (or network access + extras are
available) — it's skipped otherwise so the default test run never hits
the network.
"""

from __future__ import annotations

import pytest

from pi_memory.embeddings import EmbeddingManager, _default_model_dir


class TestEmbeddingManagerAvailability:
    def test_fresh_instance_reports_availability_without_raising(self) -> None:
        manager = EmbeddingManager()
        # Should never raise even if onnxruntime/tokenizers are missing or
        # the model hasn't been downloaded yet.
        assert isinstance(manager.is_available(), bool)


@pytest.mark.skipif(
    not _default_model_dir().exists(),
    reason="EmbeddingGemma model not cached locally; skipping network-dependent test",
)
class TestEmbeddingManagerRealModel:
    def test_embed_returns_768_dim_vector(self) -> None:
        manager = EmbeddingManager()
        vector = manager.embed("hello world", task="query")
        assert len(vector) == 768
