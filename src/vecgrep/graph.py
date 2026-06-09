"""Knowledge-graph store: AST-based structural extraction and graph queries."""

from __future__ import annotations

import datetime
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

try:
    from tree_sitter_languages import get_parser  # type: ignore

    # Verify the parser is real at import time.
    # Guards against two failure modes:
    # 1. tree-sitter version mismatch (get_parser raises TypeError at runtime)
    # 2. Mock injection by test_chunker_ast.py (root_node.type is not a str)
    _probe = get_parser("python")
    _probe_tree = _probe.parse(b"x = 1")
    _HAS_TREE_SITTER = isinstance(_probe_tree.root_node.type, str)
    del _probe, _probe_tree
except Exception:
    _HAS_TREE_SITTER = False

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_FILENAME = "graph.json"

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "c_sharp",
}

# AST node types that represent named declarations, per language
_DECL_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "async_function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
    },
    "rust": {
        "function_item": "function",
        "impl_item": "impl",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "constructor_declaration": "constructor",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
    },
    "ruby": {
        "method": "method",
        "class": "class",
        "module": "module",
    },
    "swift": {
        "function_declaration": "function",
        "class_declaration": "class",
        "struct_declaration": "struct",
        "protocol_declaration": "protocol",
    },
    "kotlin": {
        "function_declaration": "function",
        "class_declaration": "class",
    },
    "c_sharp": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "constructor_declaration": "constructor",
    },
}

# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def _make_id(*parts: str) -> str:
    """Build a stable, lowercase node ID from name parts."""
    combined = "_".join(p.strip("_.") for p in parts if p)
    combined = unicodedata.normalize("NFKC", combined)
    cleaned = re.sub(r"[^\w]+", "_", combined, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").casefold()


def _file_id(rel_path: Path) -> str:
    """Stable file-level node ID: '{parent}_{stem}' relative to project root."""
    parent = rel_path.parent.name
    stem = rel_path.stem
    if parent and parent not in (".", ""):
        return _make_id(parent, stem)
    return _make_id(stem)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _get_name(node: Any) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode(errors="ignore")
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode(errors="ignore")
    return None


def _get_bases_python(class_node: Any) -> list[str]:
    bases: list[str] = []
    arg_list = class_node.child_by_field_name("superclasses")
    if arg_list is None:
        return bases
    for child in arg_list.children:
        if child.type == "identifier":
            bases.append(child.text.decode(errors="ignore"))
        elif child.type == "attribute":
            bases.append(child.children[-1].text.decode(errors="ignore"))
    return bases


def _collect_call_names(node: Any, language: str) -> list[str]:
    """Walk an AST subtree and collect called function/method names."""
    _CALL_SPEC: dict[str, tuple[str, str]] = {
        "python": ("call", "function"),
        "javascript": ("call_expression", "function"),
        "typescript": ("call_expression", "function"),
        "tsx": ("call_expression", "function"),
        "go": ("call_expression", "function"),
        "rust": ("call_expression", "function"),
        "java": ("method_invocation", "name"),
        "c": ("call_expression", "function"),
        "cpp": ("call_expression", "function"),
    }
    spec = _CALL_SPEC.get(language)
    if spec is None:
        return []
    call_type, fn_field = spec
    names: list[str] = []

    def _walk(n: Any) -> None:
        if n.type == call_type:
            fn = n.child_by_field_name(fn_field)
            if fn is not None:
                if fn.type in ("attribute", "member_expression", "field_expression"):
                    ident = fn.children[-1]
                    if ident.type == "identifier":
                        names.append(ident.text.decode(errors="ignore"))
                elif fn.type == "identifier":
                    names.append(fn.text.decode(errors="ignore"))
        for child in n.children:
            _walk(child)

    _walk(node)
    return names


# ---------------------------------------------------------------------------
# Import extraction (regex — no AST needed)
# ---------------------------------------------------------------------------


def _collect_imports_python(source: str, rel_path: Path, root: Path) -> list[str]:
    imported: list[str] = []
    # Relative: from .sibling import x
    for m in re.finditer(r"^from\s+(\.+)([\w.]*)\s+import", source, re.MULTILINE):
        dots, module_path = len(m.group(1)), m.group(2)
        base = rel_path.parent
        for _ in range(dots - 1):
            base = base.parent
        if module_path:
            imported.append(str(base / module_path.replace(".", "/")))
    # Absolute: import x.y or from x.y import z
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", source, re.MULTILINE):
        mod = m.group(1).replace(".", "/")
        for suffix in ("", ".py", "/__init__.py"):
            candidate = root / (mod + suffix)
            if candidate.exists():
                rel = str(candidate.relative_to(root))
                imported.append(rel.removesuffix(".py").removesuffix("/__init__"))
                break
    return list(set(imported))


def _collect_imports_js(source: str) -> list[str]:
    paths: list[str] = []
    for pat in (
        re.compile(r"""(?:import|export)[^'"]*['"](\.[^'"]+)['"]"""),
        re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)"""),
    ):
        for m in pat.finditer(source):
            paths.append(m.group(1))
    return list(set(paths))


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------


def _extract_file(
    file_path: Path,
    root: Path,
    language: str,
) -> tuple[list[dict], list[dict]]:
    """Extract nodes and edges from a single source file via tree-sitter.

    Returns (nodes, edges). If tree-sitter is unavailable or parse fails, only
    a file-level node is emitted (same graceful fallback as the chunker).
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return nodes, edges

    try:
        rel_path = file_path.relative_to(root)
    except ValueError:
        rel_path = file_path

    file_node_id = _file_id(rel_path)
    rel_str = str(rel_path)
    line_count = source.count("\n") + 1

    nodes.append({
        "id": file_node_id,
        "label": rel_path.name,
        "kind": "file",
        "source_file": rel_str,
        "start_line": 1,
        "end_line": line_count,
    })

    if not _HAS_TREE_SITTER:
        return nodes, edges

    decl_types = _DECL_NODE_TYPES.get(language)
    if not decl_types:
        return nodes, edges

    try:
        parser = get_parser(language)
    except Exception:
        _log.debug("graph: get_parser(%s) failed, skipping AST for %s", language, file_path)
        return nodes, edges

    tree = parser.parse(source.encode())

    # Traverse AST, tracking the nearest enclosing declaration node_id
    # so that method nodes get a `contains` edge from their class, not the file.
    def _collect_decls(node: Any, parent_id: str) -> None:
        kind = decl_types.get(node.type)
        if kind:
            if node.type == "decorated_definition" and language == "python":
                for child in node.children:
                    if child.type in decl_types:
                        inner_kind = decl_types[child.type]
                        name = _get_name(child)
                        if name:
                            node_id = _make_id(parent_id, name)
                            start_line = node.start_point[0] + 1
                            end_line = node.end_point[0] + 1
                            nodes.append({
                                "id": node_id, "label": name, "kind": inner_kind,
                                "source_file": rel_str,
                                "start_line": start_line, "end_line": end_line,
                            })
                            edges.append({
                                "source": parent_id, "target": node_id, "relation": "contains",
                            })
                            if inner_kind == "class":
                                for base in _get_bases_python(child):
                                    edges.append({
                                        "source": node_id, "target": _make_id(base),
                                        "relation": "inherits",
                                        "_unresolved_target_label": base,
                                    })
                            for called in _collect_call_names(child, language):
                                edges.append({
                                    "source": node_id, "target": _make_id(called),
                                    "relation": "calls",
                                    "_unresolved_target_label": called,
                                })
                            for grandchild in node.children:
                                _collect_decls(grandchild, node_id)
                        break
                return

            name = _get_name(node)
            if name:
                node_id = _make_id(parent_id, name)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                nodes.append({
                    "id": node_id, "label": name, "kind": kind,
                    "source_file": rel_str,
                    "start_line": start_line, "end_line": end_line,
                })
                edges.append({"source": parent_id, "target": node_id, "relation": "contains"})

                if kind == "class" and language == "python":
                    for base in _get_bases_python(node):
                        edges.append({"source": node_id, "target": _make_id(base),
                                      "relation": "inherits", "_unresolved_target_label": base})

                for called in _collect_call_names(node, language):
                    edges.append({"source": node_id, "target": _make_id(called),
                                  "relation": "calls", "_unresolved_target_label": called})

                # Recurse with this node as the new parent (finds nested/methods)
                for child in node.children:
                    _collect_decls(child, node_id)
                return

        for child in node.children:
            _collect_decls(child, parent_id)

    _collect_decls(tree.root_node, file_node_id)

    # Import edges (regex — independent of tree-sitter)
    if language == "python":
        for imp_path in _collect_imports_python(source, rel_path, root):
            edges.append({
                "source": file_node_id,
                "target": _file_id(Path(imp_path)),
                "relation": "imports",
            })
    elif language in ("javascript", "typescript", "tsx"):
        for imp_path in _collect_imports_js(source):
            imp_abs = (file_path.parent / imp_path).resolve()
            for suffix in ("", ".ts", ".tsx", ".js", ".jsx"):
                candidate = Path(str(imp_abs) + suffix) if suffix else imp_abs
                if candidate.is_file():
                    try:
                        edges.append({
                            "source": file_node_id,
                            "target": _file_id(candidate.relative_to(root)),
                            "relation": "imports",
                        })
                    except ValueError:
                        pass
                    break

    return nodes, edges


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------


class GraphStore:
    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._graph_path = index_dir / _GRAPH_FILENAME
        self._G: nx.DiGraph | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, files: list[Path], root: Path) -> dict[str, int]:
        """Extract nodes+edges from all files and persist the graph.

        Returns {"nodes": n, "edges": e, "files": f}.
        """
        all_nodes: list[dict] = []
        all_edges: list[dict] = []
        files_processed = 0

        for fp in files:
            suffix = fp.suffix.lower()
            language = _LANGUAGE_MAP.get(suffix)
            if not language:
                try:
                    rel = fp.relative_to(root)
                except ValueError:
                    rel = fp
                all_nodes.append({
                    "id": _file_id(rel),
                    "label": fp.name,
                    "kind": "file",
                    "source_file": str(rel),
                    "start_line": 1,
                    "end_line": 1,
                })
                files_processed += 1
                continue

            try:
                nodes, edges = _extract_file(fp, root, language)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                files_processed += 1
            except Exception:
                _log.warning("graph: skipped %s", fp, exc_info=True)

        G: nx.DiGraph = nx.DiGraph()

        seen_ids: set[str] = set()
        for n in all_nodes:
            if n["id"] not in seen_ids:
                G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
                seen_ids.add(n["id"])

        # Reverse index: label → [node_ids] for resolving call/inherits targets
        label_to_ids: dict[str, list[str]] = {}
        for node_id, data in G.nodes(data=True):
            label = data.get("label", "")
            if label:
                label_to_ids.setdefault(label, []).append(node_id)

        for e in all_edges:
            src, tgt, relation = e["source"], e["target"], e["relation"]
            if src not in G:
                continue
            if "_unresolved_target_label" in e:
                label = e["_unresolved_target_label"]
                candidates = label_to_ids.get(label, [])
                if not candidates:
                    continue
                src_file = G.nodes[src].get("source_file", "")
                same_file = [c for c in candidates if G.nodes[c].get("source_file") == src_file]
                tgt = same_file[0] if same_file else candidates[0]
            if tgt not in G or src == tgt:
                continue
            G.add_edge(src, tgt, relation=relation)

        G.graph["last_built"] = datetime.datetime.now(datetime.UTC).isoformat()
        G.graph["root"] = str(root)

        self._G = G
        self._persist()

        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "files": files_processed,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        assert self._G is not None
        data = json_graph.node_link_data(self._G, edges="edges")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> nx.DiGraph:
        if self._G is not None:
            return self._G
        if not self._graph_path.exists():
            raise FileNotFoundError("Graph not built. Run index_graph first.")
        raw = json.loads(self._graph_path.read_text(encoding="utf-8"))
        if "links" not in raw and "edges" in raw:
            raw = dict(raw, links=raw["edges"])
        try:
            G = json_graph.node_link_graph(raw, directed=True, edges="links")
        except TypeError:
            G = json_graph.node_link_graph(raw, directed=True)
        self._G = G
        return G

    # ------------------------------------------------------------------
    # Query: keyword search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Keyword search over node labels and source file paths."""
        G = self._load()
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []

        results: list[tuple[float, dict]] = []
        for node_id, data in G.nodes(data=True):
            label = data.get("label", "")
            all_tokens = set(re.findall(r"\w+", label.lower()))
            all_tokens |= set(re.findall(r"\w+", data.get("source_file", "").lower()))

            overlap = query_tokens & all_tokens
            if not overlap:
                continue

            score = len(overlap) / len(query_tokens)
            if label.lower() in query.lower() or query.lower() in label.lower():
                score = min(1.0, score + 0.4)

            results.append((score, {
                "id": node_id,
                "label": label,
                "kind": data.get("kind", ""),
                "source_file": data.get("source_file", ""),
                "start_line": data.get("start_line", 0),
                "end_line": data.get("end_line", 0),
                "score": round(score, 3),
                "degree": G.degree(node_id),
            }))

        results.sort(key=lambda x: (-x[0], -G.degree(x[1]["id"])))
        return [r for _, r in results[:limit]]

    # ------------------------------------------------------------------
    # Query: neighbors
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str, depth: int = 1) -> dict:
        """Return categorised neighbors of node_id up to *depth* hops."""
        G = self._load()

        if node_id not in G:
            # Prefer exact label match, then substring
            exact = [
                n for n, d in G.nodes(data=True)
                if d.get("label", "").lower() == node_id.lower()
            ]
            partial = [
                n for n, d in G.nodes(data=True)
                if node_id.lower() in d.get("label", "").lower()
            ]
            candidates = exact or partial
            if not candidates:
                return {"error": f"Node '{node_id}' not found in graph"}
            node_id = candidates[0]

        node_data = {**G.nodes[node_id], "id": node_id}

        def _info(nid: str, relation: str) -> dict:
            return {**G.nodes[nid], "id": nid, "relation": relation}

        callers, callees, imports_, contains, contained_by, inherits = [], [], [], [], [], []

        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for _, tgt, data in G.out_edges(nid, data=True):
                    rel = data.get("relation", "")
                    if tgt not in visited:
                        next_frontier.add(tgt)
                        if rel == "calls":
                            callees.append(_info(tgt, rel))
                        elif rel == "imports":
                            imports_.append(_info(tgt, rel))
                        elif rel == "contains":
                            contains.append(_info(tgt, rel))
                        elif rel == "inherits":
                            inherits.append(_info(tgt, rel))
                for src, _, data in G.in_edges(nid, data=True):
                    rel = data.get("relation", "")
                    if src not in visited:
                        next_frontier.add(src)
                        if rel == "calls":
                            callers.append(_info(src, rel))
                        elif rel == "contains":
                            contained_by.append(_info(src, rel))
            visited |= next_frontier
            frontier = next_frontier

        return {
            "node": node_data,
            "callers": callers,
            "callees": callees,
            "imports": imports_,
            "contains": contains,
            "contained_by": contained_by,
            "inherits": inherits,
        }

    # ------------------------------------------------------------------
    # Query: chunk graph scores (for hybrid search)
    # ------------------------------------------------------------------

    def chunk_graph_scores(
        self,
        chunks: list[dict],
        query: str,
        max_bfs_depth: int = 3,
    ) -> list[float]:
        """Compute 0–1 graph-proximity scores for a list of chunks.

        1. Keyword-search the graph for "seed" nodes matching the query.
        2. BFS from seeds up to max_bfs_depth hops.
        3. For each chunk, find the graph node covering its (file, lines).
        4. Score = 1 / (1 + bfs_distance), 0 if unreachable.
        """
        G = self._load()

        seed_results = self.search(query, limit=10)
        if not seed_results:
            return [0.0] * len(chunks)

        # BFS from all seeds simultaneously
        dist: dict[str, int] = {r["id"]: 0 for r in seed_results}
        frontier = set(dist)
        for depth in range(1, max_bfs_depth + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                for nb in list(G.successors(nid)) + list(G.predecessors(nid)):
                    if nb not in dist:
                        dist[nb] = depth
                        next_frontier.add(nb)
            frontier = next_frontier

        scores: list[float] = []
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            start = chunk.get("start_line", 0)
            end = chunk.get("end_line", 0)
            best = 0.0
            for node_id, data in G.nodes(data=True):
                sf = data.get("source_file", "")
                if sf and not fp.endswith(sf):
                    continue
                n_start = data.get("start_line", 0)
                n_end = data.get("end_line", 0)
                if n_end < start or n_start > end:
                    continue
                if node_id in dist:
                    best = max(best, 1.0 / (1.0 + dist[node_id]))
            scores.append(best)

        return scores

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self._graph_path.exists()

    def status(self) -> dict:
        if not self.exists():
            return {"exists": False, "nodes": 0, "edges": 0, "last_built": "never"}
        try:
            G = self._load()
            return {
                "exists": True,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "last_built": G.graph.get("last_built", "unknown"),
            }
        except Exception:
            return {"exists": True, "nodes": 0, "edges": 0, "last_built": "corrupt"}
