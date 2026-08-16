"""Local embedding generation via EmbeddingGemma-300M (ONNX Runtime, CPU).

Uses the ``onnx-community/embeddinggemma-300m-ONNX`` export so the heavy
``torch``/``transformers`` stack is never required — only ``onnxruntime``
and ``tokenizers``. Both are optional (the ``memory`` extra); when absent,
:meth:`EmbeddingManager.is_available` returns ``False`` and callers fall
back to keyword-only search instead of failing.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    _IMPORTS_AVAILABLE = True
except ImportError:
    _IMPORTS_AVAILABLE = False

EMBEDDING_DIM = 768
_MODEL_REPO = "onnx-community/embeddinggemma-300m-ONNX"
# Available variants in the repo: (unsuffixed=fp32), fp16, quantized (dynamic
# int8, ~305MB — the closest to "q8"), q4, q4f16, no_gather_q4. There is no
# "q8"-named file, despite that being the common shorthand for int8.
_DEFAULT_QUANTIZATION = "quantized"

# EmbeddingGemma expects task-specific prefixes prepended to the raw text.
# See https://ai.google.dev/gemma/docs/embeddinggemma/model_card
_QUERY_PREFIX = "task: search result | query: "
_DOCUMENT_PREFIX = "title: none | text: "

EmbeddingTask = Literal["query", "document"]


def _default_model_dir() -> Path:
    agent_dir = os.environ.get("PI_AGENT_DIR")
    base = Path(agent_dir) if agent_dir else Path.home() / ".pi"
    return base / "models" / "embeddinggemma-300m-onnx"


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> list[float]:
    """Mean-pool token embeddings over non-padding positions, then L2-normalize."""
    mask = attention_mask[0]
    vectors = last_hidden_state[0]
    summed = [0.0] * len(vectors[0])
    count = 0
    for token_vec, keep in zip(vectors, mask, strict=True):
        if not keep:
            continue
        count += 1
        for i, value in enumerate(token_vec):
            summed[i] += float(value)
    if count == 0:
        count = 1
    pooled = [value / count for value in summed]
    norm = math.sqrt(sum(value * value for value in pooled)) or 1.0
    return [value / norm for value in pooled]


class EmbeddingManager:
    """Lazily-loaded singleton wrapping the EmbeddingGemma-300M ONNX model."""

    _instance: EmbeddingManager | None = None

    def __init__(self, model_dir: Path | None = None, *, quantization: str = _DEFAULT_QUANTIZATION) -> None:
        self._model_dir = model_dir or _default_model_dir()
        self._quantization = quantization
        self._session: Any = None
        self._tokenizer: Any = None
        self._load_failed = False
        self._on_progress: Callable[[str], None] | None = None

    @classmethod
    def get_instance(cls) -> EmbeddingManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback fired with a human-readable status line around
        the one-time model download (e.g. to surface it in a TUI). Exceptions
        raised by the callback are swallowed — a broken UI hook must never
        break embedding."""
        self._on_progress = callback

    def _notify(self, message: str) -> None:
        if self._on_progress is not None:
            with contextlib.suppress(Exception):
                self._on_progress(message)

    def is_available(self) -> bool:
        if not _IMPORTS_AVAILABLE or self._load_failed:
            return False
        try:
            self._ensure_loaded()
        except Exception:
            self._load_failed = True
            return False
        return True

    def embed(self, text: str, *, task: EmbeddingTask = "query") -> list[float]:
        """Return a 768-dim embedding for *text*.

        Raises if the model isn't available — callers should check
        :meth:`is_available` first and treat failures as "no embeddings".
        """
        self._ensure_loaded()
        prefix = _QUERY_PREFIX if task == "query" else _DOCUMENT_PREFIX
        encoding = self._tokenizer.encode(prefix + text)
        input_ids = [encoding.ids]
        attention_mask = [encoding.attention_mask]

        output_names = [o.name for o in self._session.get_outputs()]
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in {i.name for i in self._session.get_inputs()}:
            feed["token_type_ids"] = [[0] * len(encoding.ids)]

        outputs = self._session.run(output_names, feed)
        by_name = dict(zip(output_names, outputs, strict=True))

        if "sentence_embedding" in by_name:
            vector = by_name["sentence_embedding"][0]
            return [float(v) for v in vector]

        return _mean_pool(by_name["last_hidden_state"], attention_mask)

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        if not _IMPORTS_AVAILABLE:
            raise RuntimeError("onnxruntime/tokenizers not installed — install python-pi[memory]")

        self._download_if_needed()

        onnx_path = self._model_dir / "onnx" / f"model_{self._quantization}.onnx"
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(self._model_dir / "tokenizer.json"))

    def _download_if_needed(self) -> None:
        onnx_filename = f"model_{self._quantization}.onnx"
        onnx_path = self._model_dir / "onnx" / onnx_filename
        tokenizer_path = self._model_dir / "tokenizer.json"
        if onnx_path.exists() and tokenizer_path.exists():
            return

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        # huggingface_hub prints its own raw tqdm bars to stdout/stderr by
        # default, which clashes with Rich-rendered TUIs. Callers get a
        # single clean status line via set_progress_callback() instead.
        disable_progress_bars()
        self._notify(f"Downloading local memory embedding model ({self._quantization}, ~300MB, one-time)...")

        # Only pull the selected quantization's weights (+ external data file)
        # instead of every variant in the repo — the repo hosts fp32, fp16,
        # quantized (int8), q4, q4f16 and no_gather_q4 side by side (~2.5GB
        # combined), and we only need the one we're going to load.
        try:
            snapshot_download(
                repo_id=_MODEL_REPO,
                local_dir=str(self._model_dir),
                allow_patterns=[
                    f"onnx/{onnx_filename}",
                    f"onnx/{onnx_filename}_data",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
        except Exception:
            self._notify("Memory embedding model download failed — falling back to keyword-only memory search.")
            raise
        self._notify("Memory embedding model ready.")
