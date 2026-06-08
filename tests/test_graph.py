"""Tests for GraphStore extraction and queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from vecgrep.graph import GraphStore, _file_id, _make_id


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def test_make_id_basic() -> None:
    assert _make_id("Foo", "bar") == "foo_bar"


def test_make_id_strips_specials() -> None:
    assert _make_id("foo-bar!baz") == "foo_bar_baz"


def test_make_id_dedup_underscores() -> None:
    result = _make_id("foo__bar")
    assert "__" not in result


def test_file_id_with_parent() -> None:
    rel = Path("src/store.py")
    assert _file_id(rel) == "src_store"


def test_file_id_top_level() -> None:
    rel = Path("server.py")
    assert _file_id(rel) == "server"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def py_project(tmp_path: Path) -> Path:
    """A tiny Python project with two files."""
    (tmp_path / "models.py").write_text(
        """\
class User:
    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello {self.name}"
""",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        """\
from models import User

class UserService:
    def create(self, name: str) -> User:
        return User(name)
""",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def built_store(tmp_path: Path, py_project: Path) -> GraphStore:
    """A GraphStore that has been built from the py_project fixture."""
    gs = GraphStore(tmp_path / "graph_index")
    files = list(py_project.glob("*.py"))
    gs.build(files, py_project)
    return gs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def test_build_returns_stats(tmp_path: Path, py_project: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    files = list(py_project.glob("*.py"))
    stats = gs.build(files, py_project)
    assert stats["files"] == 2
    assert stats["nodes"] > 0
    assert stats["edges"] > 0


def test_build_persists_graph(tmp_path: Path, py_project: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    files = list(py_project.glob("*.py"))
    gs.build(files, py_project)
    assert (tmp_path / "idx" / "graph.json").exists()


def test_build_idempotent(tmp_path: Path, py_project: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    files = list(py_project.glob("*.py"))
    stats_a = gs.build(files, py_project)
    # Force reload from disk on second build by clearing cached graph
    gs2 = GraphStore(tmp_path / "idx")
    stats_b = gs2.build(files, py_project)
    assert stats_a["nodes"] == stats_b["nodes"]


def test_build_empty_files(tmp_path: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    stats = gs.build([], tmp_path)
    assert stats["nodes"] == 0
    assert stats["edges"] == 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_before_build(tmp_path: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    s = gs.status()
    assert s["exists"] is False


def test_status_after_build(built_store: GraphStore) -> None:
    s = built_store.status()
    assert s["exists"] is True
    assert s["nodes"] > 0
    assert s["last_built"] != "never"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_finds_class(built_store: GraphStore) -> None:
    results = built_store.search("User")
    labels = [r["label"] for r in results]
    assert any("User" in l for l in labels)


def test_search_returns_score(built_store: GraphStore) -> None:
    results = built_store.search("User")
    assert all(0.0 <= r["score"] <= 1.0 for r in results)


def test_search_empty_query(built_store: GraphStore) -> None:
    assert built_store.search("") == []


def test_search_no_match(built_store: GraphStore) -> None:
    results = built_store.search("xyzzy_nonexistent_token_9999")
    assert results == []


def test_search_limit(built_store: GraphStore) -> None:
    results = built_store.search("User", limit=1)
    assert len(results) <= 1


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------


def test_neighbors_returns_node(built_store: GraphStore) -> None:
    result = built_store.neighbors("User")
    assert "node" in result
    assert result["node"]["label"] == "User"


def test_neighbors_missing_node(built_store: GraphStore) -> None:
    result = built_store.neighbors("definitely_not_a_real_node_id_xyz")
    assert "error" in result


def test_neighbors_contains_methods(built_store: GraphStore) -> None:
    result = built_store.neighbors("User", depth=1)
    # User class should contain greet and __init__
    contained = [c["label"] for c in result.get("contains", [])]
    assert any("greet" in l or "__init__" in l for l in contained)


# ---------------------------------------------------------------------------
# chunk_graph_scores
# ---------------------------------------------------------------------------


def test_chunk_graph_scores_length(built_store: GraphStore) -> None:
    chunks = [
        {"file_path": "models.py", "start_line": 1, "end_line": 6},
        {"file_path": "service.py", "start_line": 3, "end_line": 7},
    ]
    scores = built_store.chunk_graph_scores(chunks, "User")
    assert len(scores) == len(chunks)


def test_chunk_graph_scores_range(built_store: GraphStore) -> None:
    chunks = [{"file_path": "models.py", "start_line": 1, "end_line": 10}]
    scores = built_store.chunk_graph_scores(chunks, "User")
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_chunk_graph_scores_empty_query(built_store: GraphStore) -> None:
    chunks = [{"file_path": "models.py", "start_line": 1, "end_line": 10}]
    scores = built_store.chunk_graph_scores(chunks, "")
    assert scores == [0.0]


# ---------------------------------------------------------------------------
# Reload from disk
# ---------------------------------------------------------------------------


def test_reload_from_disk(tmp_path: Path, py_project: Path) -> None:
    """GraphStore loads correctly from a previously persisted graph.json."""
    idx_dir = tmp_path / "idx"
    gs1 = GraphStore(idx_dir)
    files = list(py_project.glob("*.py"))
    gs1.build(files, py_project)

    # Fresh instance — reads from disk
    gs2 = GraphStore(idx_dir)
    results = gs2.search("User")
    assert any("User" in r["label"] for r in results)
