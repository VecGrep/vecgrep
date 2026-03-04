"""Unit tests for the BYOK embedding provider strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vecgrep.embedder import (
    PROVIDER_REGISTRY,
    GeminiProvider,
    LocalProvider,
    OpenAIProvider,
    VoyageProvider,
    embed,
    get_provider,
)

# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_providers_registered(self):
        assert set(PROVIDER_REGISTRY.keys()) == {"local", "openai", "voyage", "gemini"}

    def test_get_provider_local(self):
        p = get_provider("local")
        assert p.name == "local"

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_get_provider_error_lists_valid_options(self):
        with pytest.raises(ValueError, match="gemini"):
            get_provider("bad")


# ---------------------------------------------------------------------------
# LocalProvider
# ---------------------------------------------------------------------------


class TestLocalProvider:
    def test_properties(self):
        p = LocalProvider()
        assert p.name == "local"
        assert p.dims == 384
        assert p.batch_size == 64
        assert "MiniLM" in p.model or p.model  # model string set

    def test_embed_shape(self):
        p = LocalProvider()
        vecs = p.embed(["hello world", "def foo(): pass"])
        assert vecs.shape == (2, 384)
        assert vecs.dtype == np.float32

    def test_embed_empty(self):
        p = LocalProvider()
        vecs = p.embed([])
        assert vecs.shape == (0, 384)
        assert vecs.dtype == np.float32

    def test_embed_normalized(self):
        p = LocalProvider()
        vecs = p.embed(["alpha", "beta", "gamma"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_normalize_helper_zero_vector(self):
        p = LocalProvider()
        zero = np.zeros((1, 4), dtype=np.float32)
        result = p._normalize(zero)
        # Should not raise; zero vector stays zero after safe division
        assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# Backward-compatible embed() free function
# ---------------------------------------------------------------------------


class TestEmbedFreeFunction:
    def test_returns_correct_shape(self):
        vecs = embed(["hello"])
        assert vecs.shape == (1, 384)

    def test_returns_float32(self):
        assert embed(["x"]).dtype == np.float32

    def test_empty(self):
        assert embed([]).shape == (0, 384)


# ---------------------------------------------------------------------------
# Cloud providers — missing key raises RuntimeError
# ---------------------------------------------------------------------------


class TestCloudProvidersNoKey:
    def test_openai_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("VECGREP_OPENAI_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VECGREP_OPENAI_KEY"):
            OpenAIProvider()

    def test_voyage_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("VECGREP_VOYAGE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VECGREP_VOYAGE_KEY"):
            VoyageProvider()

    def test_gemini_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("VECGREP_GEMINI_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VECGREP_GEMINI_KEY"):
            GeminiProvider()


# ---------------------------------------------------------------------------
# Cloud providers — missing package raises RuntimeError
# ---------------------------------------------------------------------------


class TestCloudProvidersNoPackage:
    def test_openai_missing_package(self, monkeypatch):
        monkeypatch.setenv("VECGREP_OPENAI_KEY", "test-key")
        provider = OpenAIProvider()
        with patch("builtins.__import__", side_effect=ImportError("No module named 'openai'")):
            provider._client = None  # reset lazy cache
            with pytest.raises((RuntimeError, ImportError)):
                provider._get_client()

    def test_voyage_missing_package(self, monkeypatch):
        monkeypatch.setenv("VECGREP_VOYAGE_KEY", "test-key")
        provider = VoyageProvider()
        with patch("builtins.__import__", side_effect=ImportError("No module named 'voyageai'")):
            provider._client = None
            with pytest.raises((RuntimeError, ImportError)):
                provider._get_client()

    def test_gemini_missing_package(self, monkeypatch):
        monkeypatch.setenv("VECGREP_GEMINI_KEY", "test-key")
        provider = GeminiProvider()
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            provider._client = None
            with pytest.raises((RuntimeError, ImportError)):
                provider._get_client()


# ---------------------------------------------------------------------------
# Cloud providers — mocked API calls
# ---------------------------------------------------------------------------


class TestOpenAIProviderMocked:
    def test_embed_returns_normalized_array(self, monkeypatch):
        monkeypatch.setenv("VECGREP_OPENAI_KEY", "test-key")
        provider = OpenAIProvider()

        # Build fake response
        fake_embedding = [0.1] * 1536
        fake_data = [MagicMock(embedding=fake_embedding)]
        fake_response = MagicMock(data=fake_data)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = fake_response
        provider._client = mock_client

        vecs = provider.embed(["hello"])
        assert vecs.shape == (1, 1536)
        assert vecs.dtype == np.float32
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty(self, monkeypatch):
        monkeypatch.setenv("VECGREP_OPENAI_KEY", "test-key")
        provider = OpenAIProvider()
        vecs = provider.embed([])
        assert vecs.shape == (0, 1536)

    def test_properties(self, monkeypatch):
        monkeypatch.setenv("VECGREP_OPENAI_KEY", "test-key")
        p = OpenAIProvider()
        assert p.name == "openai"
        assert p.dims == 1536
        assert p.batch_size == 2048
        assert "text-embedding" in p.model


class TestVoyageProviderMocked:
    def test_embed_returns_normalized_array(self, monkeypatch):
        monkeypatch.setenv("VECGREP_VOYAGE_KEY", "test-key")
        provider = VoyageProvider()

        fake_result = MagicMock(embeddings=[[0.2] * 1024])
        mock_client = MagicMock()
        mock_client.embed.return_value = fake_result
        provider._client = mock_client

        vecs = provider.embed(["hello"])
        assert vecs.shape == (1, 1024)
        assert vecs.dtype == np.float32
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty(self, monkeypatch):
        monkeypatch.setenv("VECGREP_VOYAGE_KEY", "test-key")
        provider = VoyageProvider()
        vecs = provider.embed([])
        assert vecs.shape == (0, 1024)

    def test_properties(self, monkeypatch):
        monkeypatch.setenv("VECGREP_VOYAGE_KEY", "test-key")
        p = VoyageProvider()
        assert p.name == "voyage"
        assert p.dims == 1024
        assert p.batch_size == 128
        assert "voyage-code" in p.model


class TestGeminiProviderMocked:
    def test_embed_returns_normalized_array(self, monkeypatch):
        monkeypatch.setenv("VECGREP_GEMINI_KEY", "test-key")
        provider = GeminiProvider()

        # Build fake response: client.models.embed_content returns object with
        # .embeddings list of objects having .values attribute
        fake_embedding = MagicMock()
        fake_embedding.values = [0.3] * 3072
        fake_response = MagicMock()
        fake_response.embeddings = [fake_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = fake_response
        provider._client = mock_client

        vecs = provider.embed(["hello"])
        assert vecs.shape == (1, 3072)
        assert vecs.dtype == np.float32
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty(self, monkeypatch):
        monkeypatch.setenv("VECGREP_GEMINI_KEY", "test-key")
        provider = GeminiProvider()
        vecs = provider.embed([])
        assert vecs.shape == (0, 3072)

    def test_properties(self, monkeypatch):
        monkeypatch.setenv("VECGREP_GEMINI_KEY", "test-key")
        p = GeminiProvider()
        assert p.name == "gemini"
        assert p.dims == 3072
        assert p.batch_size == 100
        assert "gemini-embedding" in p.model
