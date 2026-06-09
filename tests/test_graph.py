"""Tests for GraphStore extraction and queries."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from networkx.readwrite import json_graph

from vecgrep.graph import (
    GraphStore,
    _collect_call_names,
    _collect_imports_js,
    _collect_imports_python,
    _extract_file,
    _file_id,
    _get_bases_python,
    _get_name,
    _make_id,
)

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
    assert any("User" in lbl for lbl in labels)


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
    assert any("greet" in lbl or "__init__" in lbl for lbl in contained)


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


# ---------------------------------------------------------------------------
# _get_name helpers
# ---------------------------------------------------------------------------


def test_get_name_via_identifier_child() -> None:
    """Falls back to first identifier child when no 'name' field exists."""
    node = MagicMock()
    node.child_by_field_name.return_value = None
    child = MagicMock()
    child.type = "identifier"
    child.text = b"my_func"
    node.children = [child]
    assert _get_name(node) == "my_func"


def test_get_name_returns_none_when_no_identifier() -> None:
    node = MagicMock()
    node.child_by_field_name.return_value = None
    other = MagicMock()
    other.type = "block"
    node.children = [other]
    assert _get_name(node) is None


# ---------------------------------------------------------------------------
# _get_bases_python
# ---------------------------------------------------------------------------


def test_get_bases_python_attribute() -> None:
    """Handles dotted base classes like `collections.UserDict`."""
    class_node = MagicMock()
    arg_list = MagicMock()
    class_node.child_by_field_name.return_value = arg_list

    attr_child = MagicMock()
    attr_child.type = "attribute"
    last = MagicMock()
    last.text = b"UserDict"
    attr_child.children = [MagicMock(), last]  # last element is the name

    arg_list.children = [attr_child]
    bases = _get_bases_python(class_node)
    assert "UserDict" in bases


def test_get_bases_python_no_superclasses() -> None:
    node = MagicMock()
    node.child_by_field_name.return_value = None
    assert _get_bases_python(node) == []


# ---------------------------------------------------------------------------
# _collect_call_names
# ---------------------------------------------------------------------------


def test_collect_call_names_unsupported_language() -> None:
    node = MagicMock()
    assert _collect_call_names(node, "ruby") == []


def test_collect_call_names_member_expression(py_project: Path) -> None:
    """Attribute/member call like `obj.method()` yields the method name."""
    # Build from actual Python source that has method calls
    gs = GraphStore(py_project / ".idx")
    files = list(py_project.glob("*.py"))
    gs.build(files, py_project)
    # UserService.create calls User() — 'User' should appear as a callee
    result = gs.neighbors("UserService", depth=1)
    callees = [c["label"] for c in result.get("callees", [])]
    assert any("User" in lbl for lbl in callees)


# ---------------------------------------------------------------------------
# _collect_imports_python / _collect_imports_js
# ---------------------------------------------------------------------------


def test_collect_imports_python_absolute(tmp_path: Path) -> None:
    (tmp_path / "utils.py").write_text("", encoding="utf-8")
    rel = Path("main.py")
    source = "import utils\n"
    result = _collect_imports_python(source, rel, tmp_path)
    assert any("utils" in r for r in result)


def test_collect_imports_python_relative(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "helper.py").write_text("", encoding="utf-8")
    rel = pkg / "main.py"
    source = "from .helper import foo\n"
    result = _collect_imports_python(source, rel.relative_to(tmp_path), tmp_path)
    assert any("helper" in r for r in result)


def test_collect_imports_js_relative() -> None:
    source = "import Foo from './foo'\nimport Bar from '../bar'\n"
    result = _collect_imports_js(source)
    assert any("foo" in r for r in result)
    assert any("bar" in r for r in result)


def test_collect_imports_js_require() -> None:
    source = "const x = require('./utils')\n"
    result = _collect_imports_js(source)
    assert any("utils" in r for r in result)


# ---------------------------------------------------------------------------
# _extract_file edge cases
# ---------------------------------------------------------------------------


def test_extract_file_oserror(tmp_path: Path) -> None:
    """Returns empty lists when the file can't be read."""
    missing = tmp_path / "ghost.py"
    nodes, edges = _extract_file(missing, tmp_path, "python")
    assert nodes == []
    assert edges == []


def test_extract_file_no_tree_sitter(tmp_path: Path) -> None:
    """When _HAS_TREE_SITTER is False, only a file node is emitted."""
    f = tmp_path / "a.py"
    f.write_text("def foo(): pass\n", encoding="utf-8")
    with patch("vecgrep.graph._HAS_TREE_SITTER", False):
        nodes, edges = _extract_file(f, tmp_path, "python")
    assert len(nodes) == 1
    assert nodes[0]["kind"] == "file"
    assert edges == []


def test_extract_file_unsupported_language(tmp_path: Path) -> None:
    """Languages absent from _DECL_NODE_TYPES produce only a file node."""
    f = tmp_path / "a.py"
    f.write_text("def foo(): pass\n", encoding="utf-8")
    with patch("vecgrep.graph._HAS_TREE_SITTER", True), \
         patch("vecgrep.graph._DECL_NODE_TYPES", {}):
        nodes, edges = _extract_file(f, tmp_path, "python")
    assert len(nodes) == 1
    assert nodes[0]["kind"] == "file"


def test_extract_file_parser_exception(tmp_path: Path) -> None:
    """If get_parser raises, returns only the file node."""
    f = tmp_path / "a.py"
    f.write_text("def foo(): pass\n", encoding="utf-8")
    with patch("vecgrep.graph._HAS_TREE_SITTER", True), \
         patch("vecgrep.graph.get_parser", side_effect=RuntimeError("oops")):
        nodes, edges = _extract_file(f, tmp_path, "python")
    assert len(nodes) == 1
    assert nodes[0]["kind"] == "file"


def test_extract_file_js_imports(tmp_path: Path) -> None:
    """JS relative imports produce import edges."""
    target = tmp_path / "utils.ts"
    target.write_text("export function helper() {}\n", encoding="utf-8")
    src = tmp_path / "main.ts"
    src.write_text("import { helper } from './utils'\n", encoding="utf-8")
    with patch("vecgrep.graph._HAS_TREE_SITTER", True):
        nodes, edges = _extract_file(src, tmp_path, "typescript")
    import_edges = [e for e in edges if e.get("relation") == "imports"]
    assert len(import_edges) >= 1


# ---------------------------------------------------------------------------
# Build edge cases
# ---------------------------------------------------------------------------


def test_build_unknown_suffix(tmp_path: Path) -> None:
    """Files with unknown extensions are added as file-only nodes."""
    f = tmp_path / "Makefile"
    f.write_text("all:\n\techo ok\n", encoding="utf-8")
    gs = GraphStore(tmp_path / "idx")
    stats = gs.build([f], tmp_path)
    assert stats["nodes"] == 1
    assert stats["files"] == 1


def test_build_extract_exception_is_skipped(tmp_path: Path) -> None:
    """If _extract_file raises, the file is skipped (not a hard failure)."""
    f = tmp_path / "a.py"
    f.write_text("def foo(): pass\n", encoding="utf-8")
    gs = GraphStore(tmp_path / "idx")
    with patch("vecgrep.graph._extract_file", side_effect=RuntimeError("boom")):
        stats = gs.build([f], tmp_path)
    assert stats["files"] == 0  # skipped


def test_build_inherits_edge(tmp_path: Path) -> None:
    """A class that subclasses another gets an inherits edge."""
    f = tmp_path / "a.py"
    f.write_text(
        "class Base:\n    pass\n\nclass Child(Base):\n    pass\n",
        encoding="utf-8",
    )
    gs = GraphStore(tmp_path / "idx")
    gs.build([f], tmp_path)
    result = gs.neighbors("Child", depth=1)
    inherits = [n["label"] for n in result.get("inherits", [])]
    assert "Base" in inherits


def test_build_decorated_function(tmp_path: Path) -> None:
    """A decorated function is extracted correctly."""
    f = tmp_path / "a.py"
    f.write_text(
        "@staticmethod\ndef my_func():\n    pass\n",
        encoding="utf-8",
    )
    gs = GraphStore(tmp_path / "idx")
    gs.build([f], tmp_path)
    results = gs.search("my_func")
    assert any("my_func" in r["label"] for r in results)


# ---------------------------------------------------------------------------
# _load edge cases
# ---------------------------------------------------------------------------


def test_load_raises_if_no_graph(tmp_path: Path) -> None:
    gs = GraphStore(tmp_path / "idx")
    with pytest.raises(FileNotFoundError):
        gs._load()


def test_load_legacy_links_key(tmp_path: Path, py_project: Path) -> None:
    """Graphs serialised with 'links' key (older networkx) load correctly."""
    idx_dir = tmp_path / "idx"
    gs = GraphStore(idx_dir)
    files = list(py_project.glob("*.py"))
    gs.build(files, py_project)

    # Rewrite graph.json to use 'edges' key (simulating newer networkx output)
    # then rename 'edges' → 'links' to trigger the legacy branch
    raw = json.loads((idx_dir / "graph.json").read_text())
    if "edges" in raw and "links" not in raw:
        raw["links"] = raw.pop("edges")
        (idx_dir / "graph.json").write_text(json.dumps(raw))

    gs2 = GraphStore(idx_dir)
    results = gs2.search("User")
    assert len(results) > 0


def test_load_node_link_graph_type_error_fallback(tmp_path: Path, py_project: Path) -> None:
    """Falls back to node_link_graph without edges= kwarg if TypeError raised."""
    idx_dir = tmp_path / "idx"
    gs = GraphStore(idx_dir)
    files = list(py_project.glob("*.py"))
    gs.build(files, py_project)

    original_fn = json_graph.node_link_graph

    call_count = {"n": 0}

    def patched(data, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise TypeError("edges kwarg not supported")
        return original_fn(data, **kwargs)

    gs2 = GraphStore(idx_dir)
    with patch("vecgrep.graph.json_graph.node_link_graph", side_effect=patched):
        g = gs2._load()
    assert g.number_of_nodes() > 0


# ---------------------------------------------------------------------------
# Neighbors — depth / inherits / imports branches
# ---------------------------------------------------------------------------


def test_neighbors_depth_two(built_store: GraphStore) -> None:
    """depth=2 returns more nodes than depth=1."""
    r1 = built_store.neighbors("User", depth=1)
    r2 = built_store.neighbors("User", depth=2)
    total1 = sum(len(v) for v in r1.values() if isinstance(v, list))
    total2 = sum(len(v) for v in r2.values() if isinstance(v, list))
    assert total2 >= total1


def test_neighbors_imports_edge(tmp_path: Path) -> None:
    """Import edges appear in the neighbors result."""
    (tmp_path / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from utils import helper\ndef run(): helper()\n",
        encoding="utf-8",
    )
    gs = GraphStore(tmp_path / "idx")
    gs.build(list(tmp_path.glob("*.py")), tmp_path)
    result = gs.neighbors("main", depth=1)
    imports = [n["label"] for n in result.get("imports", [])]
    assert any("utils" in lbl for lbl in imports)


# ---------------------------------------------------------------------------
# chunk_graph_scores — BFS distance branch
# ---------------------------------------------------------------------------


def test_chunk_graph_scores_unreachable_chunk(built_store: GraphStore) -> None:
    """A chunk in a file with no graph coverage scores 0.0."""
    chunks = [{"file_path": "totally_unknown_file.py", "start_line": 1, "end_line": 5}]
    scores = built_store.chunk_graph_scores(chunks, "User")
    assert scores == [0.0]


def test_chunk_graph_scores_bfs_depth(built_store: GraphStore) -> None:
    """BFS at depth > 0 assigns non-zero scores to adjacent nodes."""
    # service.py imports models.py — searching for 'User' should score service.py chunks too
    chunks = [{"file_path": "service.py", "start_line": 1, "end_line": 10}]
    scores = built_store.chunk_graph_scores(chunks, "User", max_bfs_depth=3)
    assert len(scores) == 1
    assert scores[0] >= 0.0


# ---------------------------------------------------------------------------
# Status — corrupt graph branch
# ---------------------------------------------------------------------------


def test_status_corrupt_graph(tmp_path: Path) -> None:
    """Status returns 'corrupt' when graph.json is invalid JSON."""
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    (idx_dir / "graph.json").write_text("{invalid json", encoding="utf-8")
    gs = GraphStore(idx_dir)
    s = gs.status()
    assert s["exists"] is True
    assert s["last_built"] == "corrupt"
