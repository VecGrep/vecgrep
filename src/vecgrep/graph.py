"""Knowledge-graph store: AST-based structural extraction and graph queries."""

from __future__ import annotations

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

    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_FILENAME = "graph.json"

# Maps file extension → tree-sitter language name (mirrors chunker.LANGUAGE_MAP)
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

# Node types in tree-sitter AST that represent named declarations
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
    },
}

# Per-language name-field child type for getting the identifier of a declaration
_NAME_FIELD = "name"  # tree-sitter convention: .child_by_field_name("name")

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
# AST extraction helpers
# ---------------------------------------------------------------------------


def _get_name(node: Any) -> str | None:
    """Extract the identifier name from a declaration AST node."""
    name_node = node.child_by_field_name(_NAME_FIELD)
    if name_node:
        return name_node.text.decode(errors="ignore")
    # Fallback: first named child of type "identifier"
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode(errors="ignore")
    return None


def _get_bases_python(class_node: Any) -> list[str]:
    """Extract base class names from a Python class_definition node."""
    bases: list[str] = []
    arg_list = class_node.child_by_field_name("superclasses")
    if arg_list is None:
        return bases
    for child in arg_list.children:
        if child.type == "identifier":
            bases.append(child.text.decode(errors="ignore"))
        elif child.type == "attribute":
            # e.g. module.BaseClass
            attr_name = child.children[-1].text.decode(errors="ignore")
            bases.append(attr_name)
    return bases


def _collect_call_names(node: Any, language: str) -> list[str]:
    """Walk an AST subtree and collect called function/method names."""
    names: list[str] = []
    if language == "python":
        call_type, fn_field = "call", "function"
    elif language in ("javascript", "typescript", "tsx"):
        call_type, fn_field = "call_expression", "function"
    elif language == "go":
        call_type, fn_field = "call_expression", "function"
    elif language == "rust":
        call_type, fn_field = "call_expression", "function"
    elif language == "java":
        call_type, fn_field = "method_invocation", "name"
    elif language in ("c", "cpp"):
        call_type, fn_field = "call_expression", "function"
    else:
        return names

    def _walk(n: Any) -> None:
        if n.type == call_type:
            fn = n.child_by_field_name(fn_field)
            if fn is not None:
                # Unwrap attribute access: foo.bar → "bar" and "foo"
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


def _collect_imports_python(source: str, rel_path: Path, root: Path) -> list[str]:
    """Return relative file paths that this Python file imports from the project.

    Only resolves intra-project imports (relative or matching a known module path).
    """
    imported: list[str] = []
    # Relative imports: from . import x, from .sibling import y
    rel_pattern = re.compile(r"^from\s+(\.+)([\w.]*)\s+import", re.MULTILINE)
    for m in rel_pattern.finditer(source):
        dots = len(m.group(1))
        module_path = m.group(2)
        # Resolve relative to current file's directory
        base = rel_path.parent
        for _ in range(dots - 1):
            base = base.parent
        if module_path:
            candidate = base / Path(module_path.replace(".", "/"))
            for suffix in (".py", "/__init__.py"):
                resolved = root / (str(candidate) + suffix.replace("/__init__.py", "/") + "/__init__.py" if suffix == "/__init__.py" else str(candidate) + suffix)
                # simpler: just store the module path as-is for edge target resolution
            imported.append(str(base / module_path.replace(".", "/")))
        else:
            imported.append(str(base))

    # Absolute imports: import x.y.z or from x.y import z
    abs_pattern = re.compile(r"^(?:import|from)\s+([\w.]+)", re.MULTILINE)
    for m in abs_pattern.finditer(source):
        mod = m.group(1).replace(".", "/")
        # Only include if the module path exists within the project
        for suffix in ("", ".py", "/__init__.py"):
            candidate = root / (mod + suffix)
            if candidate.exists():
                rel = str(candidate.relative_to(root))
                imported.append(rel.removesuffix(".py").removesuffix("/__init__"))
                break
    return list(set(imported))


def _collect_imports_js(source: str) -> list[str]:
    """Extract import/require paths from JS/TS source (relative paths only)."""
    paths: list[str] = []
    # import ... from './path' or "../path"
    import_pat = re.compile(r"""(?:import|export)[^'"]*['"](\.[^'"]+)['"]""")
    # require('./path')
    require_pat = re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)""")
    for pat in (import_pat, require_pat):
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
    """Extract nodes and edges from one source file.

    Returns (nodes, edges) where each is a list of dicts.
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

    # File-level node (always added)
    nodes.append({
        "id": file_node_id,
        "label": rel_path.name,
        "kind": "file",
        "source_file": rel_str,
        "start_line": 1,
        "end_line": source.count("\n") + 1,
    })

    if not _HAS_TREE_SITTER or language not in _DECL_NODE_TYPES:
        return nodes, edges

    decl_types = _DECL_NODE_TYPES[language]

    try:
        parser = get_parser(language)
    except Exception:
        return nodes, edges

    tree = parser.parse(source.encode())
    lines = source.splitlines()

    # Collect all declaration nodes in a first pass
    decl_nodes: list[tuple[Any, str, str]] = []  # (ast_node, kind, name)

    def _collect_decls(node: Any) -> None:
        kind = decl_types.get(node.type)
        if kind:
            # For decorated_definition (Python), look inside for the real decl
            if node.type == "decorated_definition" and language == "python":
                for child in node.children:
                    if child.type in decl_types:
                        inner_kind = decl_types[child.type]
                        name = _get_name(child)
                        if name:
                            decl_nodes.append((node, inner_kind, name))
                        return
            name = _get_name(node)
            if name:
                decl_nodes.append((node, kind, name))
            return
        for child in node.children:
            _collect_decls(child)

    _collect_decls(tree.root_node)

    # Build nodes and contains edges
    for ast_node, kind, name in decl_nodes:
        node_id = _make_id(file_node_id, name)
        start_line = ast_node.start_point[0] + 1
        end_line = ast_node.end_point[0] + 1

        nodes.append({
            "id": node_id,
            "label": name,
            "kind": kind,
            "source_file": rel_str,
            "start_line": start_line,
            "end_line": end_line,
        })
        edges.append({
            "source": file_node_id,
            "target": node_id,
            "relation": "contains",
        })

        # Inheritance edges (Python classes)
        if kind == "class" and language == "python":
            for base in _get_bases_python(ast_node):
                edges.append({
                    "source": node_id,
                    "target": _make_id(base),  # resolved in build() second pass
                    "relation": "inherits",
                    "_unresolved_target_label": base,
                })

        # Call edges: collect called names inside this declaration
        for called_name in _collect_call_names(ast_node, language):
            edges.append({
                "source": node_id,
                "target": _make_id(called_name),  # resolved in build() second pass
                "relation": "calls",
                "_unresolved_target_label": called_name,
            })

    # Import edges
    if language == "python":
        for imp_path in _collect_imports_python(source, rel_path, root):
            # Convert to file_id format
            imp_rel = Path(imp_path)
            target_id = _file_id(imp_rel)
            edges.append({
                "source": file_node_id,
                "target": target_id,
                "relation": "imports",
            })
    elif language in ("javascript", "typescript", "tsx"):
        for imp_path in _collect_imports_js(source):
            # Resolve relative to this file's directory
            imp_abs = (file_path.parent / imp_path).resolve()
            for suffix in ("", ".ts", ".tsx", ".js", ".jsx"):
                candidate = Path(str(imp_abs) + suffix) if suffix else imp_abs
                if candidate.is_file():
                    try:
                        imp_rel = candidate.relative_to(root)
                        target_id = _file_id(imp_rel)
                        edges.append({
                            "source": file_node_id,
                            "target": target_id,
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
                # For non-code files (md, yaml, etc.), add a file node only
                try:
                    rel = fp.relative_to(root)
                except ValueError:
                    rel = fp
                fid = _file_id(rel)
                all_nodes.append({
                    "id": fid,
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
                _log.warning("graph: skipped %s (extraction error)", fp, exc_info=True)

        # Build the graph
        G: nx.DiGraph = nx.DiGraph()

        # Add all nodes first so we have a complete ID set for edge resolution
        seen_node_ids: set[str] = set()
        for n in all_nodes:
            if n["id"] not in seen_node_ids:
                G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
                seen_node_ids.add(n["id"])

        # Build a label→id reverse index for resolving unresolved edges
        label_to_ids: dict[str, list[str]] = {}
        for node_id, data in G.nodes(data=True):
            label = data.get("label", "")
            if label:
                label_to_ids.setdefault(label, []).append(node_id)

        # Add edges — resolve unresolved targets
        edge_count = 0
        for e in all_edges:
            src = e["source"]
            tgt = e["target"]
            relation = e["relation"]

            if src not in G:
                continue

            # Resolve unresolved targets (calls/inherits use label-based IDs)
            if "_unresolved_target_label" in e:
                label = e["_unresolved_target_label"]
                candidates = label_to_ids.get(label, [])
                if not candidates:
                    continue  # skip dangling edges (stdlib/external)
                # Prefer same-file target; otherwise pick first
                src_file = G.nodes[src].get("source_file", "")
                same_file = [c for c in candidates if G.nodes[c].get("source_file", "") == src_file]
                tgt = same_file[0] if same_file else candidates[0]

            if tgt not in G:
                continue
            if src == tgt:
                continue

            G.add_edge(src, tgt, relation=relation)
            edge_count += 1

        # Store last_built timestamp via a graph-level attribute
        import datetime
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
            raise FileNotFoundError(f"Graph not built. Run index_graph first.")
        raw = json.loads(self._graph_path.read_text(encoding="utf-8"))
        # networkx compatibility: accept both "edges" and "links" keys
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
        """Keyword search over node labels. Returns nodes ranked by match quality."""
        G = self._load()
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return []

        results: list[tuple[float, dict]] = []
        for node_id, data in G.nodes(data=True):
            label = data.get("label", "")
            label_tokens = set(re.findall(r"\w+", label.lower()))
            # Also tokenize source_file path
            file_tokens = set(re.findall(r"\w+", data.get("source_file", "").lower()))
            all_tokens = label_tokens | file_tokens

            overlap = query_tokens & all_tokens
            if not overlap:
                continue

            # Score: fraction of query tokens matched, boosted by exact label match
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
        """Return the subgraph around node_id up to *depth* hops.

        Returns a dict with the target node and categorised neighbor lists.
        """
        G = self._load()

        # Try exact match first, then prefix/substring
        if node_id not in G:
            candidates = [n for n in G.nodes() if node_id.lower() in n.lower()]
            if not candidates:
                candidates = [
                    n for n, d in G.nodes(data=True)
                    if node_id.lower() in d.get("label", "").lower()
                ]
            if not candidates:
                return {"error": f"Node '{node_id}' not found in graph"}
            node_id = candidates[0]

        node_data = dict(G.nodes[node_id])
        node_data["id"] = node_id

        def _node_info(nid: str, relation: str) -> dict:
            d = dict(G.nodes[nid])
            d["id"] = nid
            d["relation"] = relation
            return d

        callers: list[dict] = []
        callees: list[dict] = []
        imports_: list[dict] = []
        contains: list[dict] = []
        contained_by: list[dict] = []
        inherits: list[dict] = []

        # BFS up to `depth` hops
        visited = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                for _, tgt, data in G.out_edges(nid, data=True):
                    relation = data.get("relation", "")
                    if tgt not in visited:
                        next_frontier.add(tgt)
                        if relation == "calls":
                            callees.append(_node_info(tgt, relation))
                        elif relation == "imports":
                            imports_.append(_node_info(tgt, relation))
                        elif relation == "contains":
                            contains.append(_node_info(tgt, relation))
                        elif relation == "inherits":
                            inherits.append(_node_info(tgt, relation))
                for src, _, data in G.in_edges(nid, data=True):
                    relation = data.get("relation", "")
                    if src not in visited:
                        next_frontier.add(src)
                        if relation == "calls":
                            callers.append(_node_info(src, relation))
                        elif relation == "contains":
                            contained_by.append(_node_info(src, relation))
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
    # Query: chunk graph score (for hybrid search)
    # ------------------------------------------------------------------

    def chunk_graph_scores(
        self,
        chunks: list[dict],
        query: str,
        max_bfs_depth: int = 3,
    ) -> list[float]:
        """Compute a 0–1 graph-proximity score for each chunk.

        Strategy:
        1. Keyword-search the graph for nodes matching the query ("seed" nodes).
        2. BFS from each seed node.
        3. For each chunk, find the graph node that best covers its (file, line) range.
        4. Score = max over seeds: 1 / (1 + bfs_distance). 0 if unreachable within depth.
        """
        G = self._load()

        # Step 1: find seed nodes from query
        seed_results = self.search(query, limit=10)
        if not seed_results:
            return [0.0] * len(chunks)

        seeds = [r["id"] for r in seed_results]

        # Step 2: BFS from all seeds simultaneously
        dist_from_seeds: dict[str, int] = {s: 0 for s in seeds}
        frontier = set(seeds)
        for depth in range(1, max_bfs_depth + 1):
            next_frontier: set[str] = set()
            for nid in frontier:
                for neighbor in list(G.successors(nid)) + list(G.predecessors(nid)):
                    if neighbor not in dist_from_seeds:
                        dist_from_seeds[neighbor] = depth
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # Step 3: map each chunk to its best graph node
        scores: list[float] = []
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            start = chunk.get("start_line", 0)
            end = chunk.get("end_line", 0)

            best_score = 0.0
            for node_id, data in G.nodes(data=True):
                if data.get("source_file") and not fp.endswith(data["source_file"]):
                    continue
                n_start = data.get("start_line", 0)
                n_end = data.get("end_line", 0)
                # Check overlap
                if n_end < start or n_start > end:
                    continue
                if node_id in dist_from_seeds:
                    node_score = 1.0 / (1.0 + dist_from_seeds[node_id])
                    best_score = max(best_score, node_score)

            scores.append(best_score)

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
