"""Shared fixtures for VecGrep tests."""

from __future__ import annotations

import numpy as np
import pytest

# Import graph module here so its import-time tree-sitter probe runs before
# test_chunker_ast.py replaces sys.modules["tree_sitter_languages"] with a mock.
import vecgrep.graph  # noqa: F401
from vecgrep.store import VectorStore


@pytest.fixture()
def store(tmp_path):
    """A fresh VectorStore backed by a temp directory."""
    s = VectorStore(tmp_path / "test_store")
    yield s


def make_unit_vecs(n: int, dim: int = 384, seed: int = 42) -> np.ndarray:
    """Return n random unit-normalised float32 vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_rows(
    n: int,
    file_path: str = "a.py",
    file_hash: str = "filehash",
) -> list[dict]:
    """Return n minimal chunk-row dicts."""
    return [
        {
            "file_path": file_path,
            "start_line": i + 1,
            "end_line": i + 1,
            "content": f"content_{i}",
            "file_hash": file_hash,
            "chunk_hash": f"chunkhash_{i}",
            "mtime": 123456789.0,
            "size": 1024,
        }
        for i in range(n)
    ]
