"""Embedding providers: local ONNX/torch + cloud (OpenAI, Voyage, Gemini) via BYOK."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------

# Model to use for the local backend. Default is the fine-tuned code search model.
# Set VECGREP_MODEL=sentence-transformers/all-MiniLM-L6-v2 (or any HF model)
# to use a different model with the torch backend.
#
# Citation for default model:
#   isuruwijesiri. (2026). all-MiniLM-L6-v2-code-search-512 [Model].
#   Hugging Face. https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512
DEFAULT_MODEL = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
MODEL_NAME = os.environ.get("VECGREP_MODEL", DEFAULT_MODEL)

# Backend to use for the local provider.
# - "onnx"  (default) — fastembed + ONNX Runtime, ~100ms startup, no PyTorch needed
# - "torch" — sentence-transformers + PyTorch, ~2-3s startup, supports any HF model
BACKEND = os.environ.get("VECGREP_BACKEND", "onnx").lower()

# Batch sizes tuned per device for the torch backend
_TORCH_BATCH_SIZE: dict[str, int] = {
    "cuda": 256,
    "mps": 256,
    "cpu": 64,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name (e.g. 'local', 'openai')."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier used for embeddings."""

    @property
    @abstractmethod
    def dims(self) -> int:
        """Embedding dimensionality."""

    @property
    def batch_size(self) -> int:
        """Number of texts to embed per API call."""
        return 64

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning a float32 array of shape (N, dims)."""

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalize rows so cosine similarity equals dot product."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Local provider (ONNX / torch)
# ---------------------------------------------------------------------------


def _detect_device() -> str:
    """Return the best available compute device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class LocalProvider(EmbeddingProvider):
    """Local embeddings using fastembed (ONNX) or sentence-transformers (torch)."""

    def __init__(self) -> None:
        self._onnx_model = None
        self._torch_model = None
        self._torch_device: str | None = None

    @property
    def name(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return MODEL_NAME

    @property
    def dims(self) -> int:
        return 384

    @property
    def batch_size(self) -> int:
        return 64

    def _get_onnx_model(self):
        if self._onnx_model is None:
            from fastembed import TextEmbedding  # type: ignore
            from fastembed.common.model_description import ModelSource, PoolingType  # type: ignore

            if MODEL_NAME == DEFAULT_MODEL:
                try:
                    TextEmbedding.add_custom_model(
                        model=DEFAULT_MODEL,
                        pooling=PoolingType.MEAN,
                        normalization=True,
                        sources=ModelSource(hf=DEFAULT_MODEL),
                        dim=384,
                        model_file="onnx/model.onnx",
                        description="Fine-tuned MiniLM for semantic code search",
                    )
                except ValueError:
                    # Model already registered in this process — safe to ignore
                    pass
            self._onnx_model = TextEmbedding(MODEL_NAME)
        return self._onnx_model

    def _get_torch_model(self):
        if self._torch_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._torch_device = _detect_device()
            self._torch_model = SentenceTransformer(MODEL_NAME, device=self._torch_device)
        return self._torch_model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        if BACKEND == "torch":
            model = self._get_torch_model()
            device = self._torch_device or "cpu"
            batch = _TORCH_BATCH_SIZE.get(device, 64)
            vecs = model.encode(
                texts,
                batch_size=batch,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            model = self._get_onnx_model()
            vecs = np.array(list(model.embed(texts)))

        return self._normalize(vecs)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIProvider(EmbeddingProvider):
    """Cloud embeddings via OpenAI text-embedding-3-small (1536 dims)."""

    _MODEL = "text-embedding-3-small"
    _DIMS = 1536

    def __init__(self) -> None:
        key = os.environ.get("VECGREP_OPENAI_KEY")
        if not key:
            raise RuntimeError(
                "OpenAI provider requires VECGREP_OPENAI_KEY environment variable. "
                "Set it to your OpenAI API key and retry."
            )
        self._key = key
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._MODEL

    @property
    def dims(self) -> int:
        return self._DIMS

    @property
    def batch_size(self) -> int:
        return 2048  # OpenAI supports up to 2048 inputs per request

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required for OpenAI embeddings. "
                    "Install it with: pip install 'vecgrep[openai]'"
                ) from exc
            self._client = OpenAI(api_key=self._key)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        client = self._get_client()
        response = client.embeddings.create(model=self._MODEL, input=texts)
        vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
        return self._normalize(vecs)


# ---------------------------------------------------------------------------
# Voyage AI provider
# ---------------------------------------------------------------------------


class VoyageProvider(EmbeddingProvider):
    """Cloud embeddings via Voyage AI voyage-code-2 (1024 dims)."""

    _MODEL = "voyage-code-2"
    _DIMS = 1024

    def __init__(self) -> None:
        key = os.environ.get("VECGREP_VOYAGE_KEY")
        if not key:
            raise RuntimeError(
                "Voyage provider requires VECGREP_VOYAGE_KEY environment variable. "
                "Set it to your Voyage AI API key and retry."
            )
        self._key = key
        self._client = None

    @property
    def name(self) -> str:
        return "voyage"

    @property
    def model(self) -> str:
        return self._MODEL

    @property
    def dims(self) -> int:
        return self._DIMS

    @property
    def batch_size(self) -> int:
        return 128  # Voyage AI recommended batch size

    def _get_client(self):
        if self._client is None:
            try:
                import voyageai  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "voyageai package is required for Voyage embeddings. "
                    "Install it with: pip install 'vecgrep[voyage]'"
                ) from exc
            self._client = voyageai.Client(api_key=self._key)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        client = self._get_client()
        result = client.embed(texts, model=self._MODEL, input_type="query")
        vecs = np.array(result.embeddings, dtype=np.float32)
        return self._normalize(vecs)


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


class GeminiProvider(EmbeddingProvider):
    """Cloud embeddings via Google Gemini gemini-embedding-001 (3072 dims).

    Uses the google-genai SDK (install with: pip install 'vecgrep[gemini]').
    """

    _MODEL = "gemini-embedding-001"
    _DIMS = 3072

    def __init__(self) -> None:
        key = os.environ.get("VECGREP_GEMINI_KEY")
        if not key:
            raise RuntimeError(
                "Gemini provider requires VECGREP_GEMINI_KEY environment variable. "
                "Set it to your Google Gemini API key and retry."
            )
        self._key = key
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._MODEL

    @property
    def dims(self) -> int:
        return self._DIMS

    @property
    def batch_size(self) -> int:
        return 100  # Gemini batch limit

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai package is required for Gemini embeddings. "
                    "Install it with: pip install 'vecgrep[gemini]'"
                ) from exc
            self._client = genai.Client(api_key=self._key)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dims), dtype=np.float32)

        client = self._get_client()
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=self._MODEL,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)

        vecs = np.array(embeddings, dtype=np.float32)
        return self._normalize(vecs)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "local": LocalProvider,
    "openai": OpenAIProvider,
    "voyage": VoyageProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str) -> EmbeddingProvider:
    """Instantiate and return an embedding provider by name.

    Args:
        name: One of 'local', 'openai', 'voyage', 'gemini'.

    Raises:
        ValueError: If the provider name is not recognised.
        RuntimeError: If required env var / package is missing.
    """
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider '{name}'. Valid options: {valid}")
    return cls()


# ---------------------------------------------------------------------------
# Backward-compatible public API
# ---------------------------------------------------------------------------

# Module-level singleton for the default local provider (lazy-initialised)
_local_provider: LocalProvider | None = None


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using the local provider.

    Backward-compatible wrapper — delegates to LocalProvider so existing
    call-sites and tests continue to work unchanged.

    Returns a normalised float32 array of shape (N, 384).
    """
    global _local_provider
    if _local_provider is None:
        _local_provider = LocalProvider()
    return _local_provider.embed(texts)
