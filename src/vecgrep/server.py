
"""FastMCP server exposing VecGrep tools."""

from __future__ import annotations

import atexit
import fnmatch
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from vecgrep.chunker import chunk_file
from vecgrep.embedder import EmbeddingProvider, _detect_device, embed, get_provider
from vecgrep.store import VectorStore

_log = logging.getLogger(__name__)


def _stop_all_observers() -> None:
    """Stop all background watchdog threads on process exit."""
    for observer in list(_OBSERVER_REGISTRY.values()):
        try:
            observer.stop()
            observer.join(timeout=2)
        except Exception:
            pass
    _OBSERVER_REGISTRY.clear()

# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("vecgrep")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECGREP_HOME = Path.home() / ".vecgrep"
_WATCH_STATE_FILE = VECGREP_HOME / "watched.json"

ALWAYS_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "eggs",
    ".eggs",
    "htmlcov",
}

ALWAYS_SKIP_PATTERNS = [
    "*.min.js",
    "*.bundle.js",
    "*.lock",
    "*.pyc",
    "*.class",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.DS_Store",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.whl",
    "*.egg",
]

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go",
    ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".rb", ".swift", ".kt", ".cs", ".md", ".txt", ".yaml",
    ".yml", ".toml", ".json", ".sh", ".bash", ".zsh",
    ".fish", ".html", ".css", ".scss", ".less", ".sql",
    ".graphql", ".proto", ".tf", ".hcl", ".dockerfile",
    ".vue", ".svelte",
}

MAX_FILE_BYTES = 512 * 1024  # 512 KB — skip very large files
EMBED_BATCH = 64

# ---------------------------------------------------------------------------
# Per-path indexing locks
# ---------------------------------------------------------------------------

_LOCK_REGISTRY: dict[str, threading.Lock] = {}
_LOCK_REGISTRY_LOCK = threading.Lock()
_MERKLE_LOCK = threading.Lock()


def _get_index_lock(path: str) -> threading.Lock:
    """Return (and create if needed) a per-path threading.Lock."""
    with _LOCK_REGISTRY_LOCK:
        if path not in _LOCK_REGISTRY:
            _LOCK_REGISTRY[path] = threading.Lock()
        return _LOCK_REGISTRY[path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_hash(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _get_store(path: str, dims: int = 384) -> VectorStore:
    index_dir = VECGREP_HOME / _project_hash(path)
    return VectorStore(index_dir, dims=dims)


def _sha256_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _load_gitignore(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    patterns: list[str] = []
    if gitignore.exists():
        for line in gitignore.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _resolve_provider(path: str, requested: str | None, force: bool) -> EmbeddingProvider:
    """Determine which provider to use and enforce the per-project lock.

    Rules:
    - If no stored provider, use *requested* (or 'local' as default).
    - If stored provider matches *requested* (or *requested* is None), use stored.
    - If stored differs from *requested* and force=False → raise RuntimeError.
    - If stored differs and force=True → allow switch (caller must recreate table).
    """
    with _get_store(path) as store:
        meta = store.get_provider_meta()
        stored_provider = meta["provider"]

    # Treat 'unknown' stored provider as 'local' (pre-BYOK index)
    if stored_provider == "unknown":
        stored_provider = "local"

    effective = requested or stored_provider

    if requested and stored_provider != "unknown" and stored_provider != requested and not force:
        raise RuntimeError(
            f"Index was built with provider '{stored_provider}'. "
            f"Switching to '{requested}' requires force=True to rebuild the index."
        )

    return get_provider(effective)


# ---------------------------------------------------------------------------
# Merkle Tree Change Detection
# ---------------------------------------------------------------------------


def _project_dir(root: str) -> Path:
    d = VECGREP_HOME / _project_hash(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_merkle_tree(root: Path, gitignore: list[str]) -> dict[str, str]:
    tree: dict[str, str] = {}

    def _hash_dir(dirpath: Path) -> str:
        child_hashes: list[str] = []
        try:
            entries = sorted(dirpath.iterdir())
        except PermissionError:
            return ""

        for entry in entries:
            try:
                rel = str(entry.relative_to(root))
            except ValueError:
                rel = str(entry)
            try:
                is_dir = entry.is_dir()
                is_file = entry.is_file()
            except OSError:
                continue
            if is_dir:
                if entry.name in ALWAYS_SKIP_DIRS or entry.name.startswith("."):
                    continue
                if _is_ignored_by_gitignore(rel, gitignore):
                    continue
                dh = _hash_dir(entry)
                if dh:
                    child_hashes.append(dh)
            elif is_file:
                if _is_ignored_by_gitignore(rel, gitignore) or _should_skip_file(entry):
                    continue
                try:
                    st = entry.stat()
                    h = hashlib.sha256(f"{st.st_mtime}:{st.st_size}".encode()).hexdigest()
                except OSError:
                    continue
                tree[str(entry)] = h
                child_hashes.append(h)

        if child_hashes:
            dir_hash = hashlib.sha256("".join(child_hashes).encode()).hexdigest()
        else:
            dir_hash = ""
        if dir_hash:
            tree[str(dirpath)] = dir_hash
        return dir_hash

    _hash_dir(root)
    return tree


def _save_merkle_tree(root: str, tree: dict[str, str]) -> None:
    path = _project_dir(root) / "merkle.json"
    path.write_text(json.dumps(tree))


def _load_merkle_tree(root: str) -> dict[str, str]:
    path = _project_dir(root) / "merkle.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _find_changed_files(old_tree: dict[str, str], new_tree: dict[str, str]) -> list[Path]:
    changed: list[Path] = []
    for path_str, new_hash in new_tree.items():
        p = Path(path_str)
        if p.is_file() and old_tree.get(path_str) != new_hash:
            changed.append(p)
    return changed


def _merkle_sync(path_str: str) -> str:
    root = Path(path_str).resolve()
    gitignore = _load_gitignore(root)

    old_tree = _load_merkle_tree(str(root))
    new_tree = _build_merkle_tree(root, gitignore)

    root_str = str(root)
    if old_tree.get(root_str) == new_tree.get(root_str):
        _save_merkle_tree(root_str, new_tree)
        _ensure_watcher(root, gitignore)
        return "No changes detected"

    changed = _find_changed_files(old_tree, new_tree)
    if not changed:
        _save_merkle_tree(root_str, new_tree)
        _ensure_watcher(root, gitignore)
        return "No file changes detected"

    # Auto-read stored provider; do not pass provider param so it defaults to stored
    result = _do_index(path_str, force=False, watch=True, provider=None)
    return result


# ---------------------------------------------------------------------------
# Watch State Persistence
# ---------------------------------------------------------------------------


def _save_watched_paths() -> None:
    paths = list(_OBSERVER_REGISTRY.keys())
    _WATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _WATCH_STATE_FILE.write_text(json.dumps(paths))


def _load_watched_paths() -> list[str]:
    if _WATCH_STATE_FILE.exists():
        try:
            return json.loads(_WATCH_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


# ---------------------------------------------------------------------------
# Background File Watcher
# ---------------------------------------------------------------------------

_OBSERVER_REGISTRY: dict[str, Observer] = {}
atexit.register(_stop_all_observers)


class LiveSyncHandler(FileSystemEventHandler):
    def __init__(self, root_path: str, gitignore_patterns: list[str]):
        self.root_path = root_path
        self.gitignore_patterns = gitignore_patterns
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._debounce_delay = 2.0  # seconds

    def _process_file(self, file_path_str: str) -> None:
        file_path = Path(file_path_str)
        if not file_path.exists() or not file_path.is_file():
            return

        try:
            rel = str(file_path.relative_to(self.root_path))
        except ValueError:
            rel = str(file_path)

        if _is_ignored_by_gitignore(rel, self.gitignore_patterns) or _should_skip_file(file_path):
            return

        # Read stored provider — skip live sync for non-local providers (cost risk)
        with _get_store(self.root_path) as store:
            provider_name = store.get_provider_meta()["provider"]
        if provider_name not in ("local", "unknown"):
            _log.warning(
                "LiveSync skipped for %s: provider '%s' is not local",
                file_path_str, provider_name,
            )
            return

        emb_provider = get_provider("local")

        # Perform targeted index for this single file
        lock = _get_index_lock(self.root_path)
        if not lock.acquire(blocking=False):
            return  # Wait for full index to finish or let a future event pick it up

        try:
            with _get_store(self.root_path) as store:
                # Fast track: check if actually changed by stat
                # (watchdog can be trigger-happy with save sequences)
                stats = store.get_file_stats()
                fp_str = str(file_path)

                try:
                    stat_res = file_path.stat()
                    current_mtime = stat_res.st_mtime
                    current_size = stat_res.st_size
                except OSError:
                    return

                existing = stats.get(fp_str)
                if (
                    existing
                    and existing["mtime"] == current_mtime
                    and existing["size"] == current_size
                ):
                    return

                # It changed, process it
                file_hash = _sha256_file(file_path)
                chunks = chunk_file(fp_str)

                if not chunks:
                    store.delete_file_chunks(fp_str)
                    return

                rows = [
                    {
                        "file_path": fp_str,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "content": c.content,
                        "file_hash": file_hash,
                        "chunk_hash": _sha256_str(c.content),
                        "mtime": current_mtime,
                        "size": current_size,
                    }
                    for c in chunks
                ]
                texts = [c.content for c in chunks]

                batch_size = emb_provider.batch_size
                batch_vecs: list[np.ndarray] = []
                for i in range(0, len(texts), batch_size):
                    batch_vecs.append(emb_provider.embed(texts[i : i + batch_size]))
                vecs = np.concatenate(batch_vecs)

                store.replace_file_chunks(fp_str, rows, vecs)
                store.touch_last_indexed()

                # Update Merkle tree incrementally (global lock prevents concurrent
                # load-modify-save races across different file events)
                with _MERKLE_LOCK:
                    tree = _load_merkle_tree(self.root_path)
                    tree[fp_str] = hashlib.sha256(
                        f"{current_mtime}:{current_size}".encode()
                    ).hexdigest()
                    _save_merkle_tree(self.root_path, tree)
        except Exception:
            _log.exception("LiveSync error processing %s", file_path_str)
        finally:
            lock.release()

    def _schedule_processing(self, file_path: str) -> None:
        if file_path in self._debounce_timers:
            self._debounce_timers[file_path].cancel()

        def _run_and_cleanup() -> None:
            self._process_file(file_path)
            self._debounce_timers.pop(file_path, None)

        timer = threading.Timer(self._debounce_delay, _run_and_cleanup)
        self._debounce_timers[file_path] = timer
        timer.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_processing(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_processing(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            root = self.root_path
            fp = event.src_path
            if fp in self._debounce_timers:
                self._debounce_timers[fp].cancel()

            def _run_and_cleanup() -> None:
                self._delete_file(root, fp)
                self._debounce_timers.pop(fp, None)

            timer = threading.Timer(self._debounce_delay, _run_and_cleanup)
            self._debounce_timers[fp] = timer
            timer.start()

    def _delete_file(self, root: str, file_path: str) -> None:
        lock = _get_index_lock(root)
        if lock.acquire(blocking=False):
            try:
                with _get_store(root) as store:
                    store.delete_file_chunks(file_path)
            finally:
                lock.release()

def _ensure_watcher(root: Path, gitignore: list[str]) -> None:
    root_str = str(root)
    with _LOCK_REGISTRY_LOCK:
        if root_str not in _OBSERVER_REGISTRY:
            observer = Observer()
            handler = LiveSyncHandler(root_str, gitignore)
            observer.schedule(handler, root_str, recursive=True)
            observer.start()
            _OBSERVER_REGISTRY[root_str] = observer
    _save_watched_paths()


def _is_ignored_by_gitignore(rel_path: str, patterns: list[str]) -> bool:
    parts = Path(rel_path).parts
    for pattern in patterns:
        # Match against the full relative path
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # Match against each path component
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _should_skip_file(file_path: Path) -> bool:
    name = file_path.name
    for pattern in ALWAYS_SKIP_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return True
    except OSError:
        return True
    return False


def _walk_files(root: Path, gitignore_patterns: list[str]) -> list[Path]:
    """Collect all indexable files under root."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and not _is_ignored_by_gitignore(
                str(Path(dirpath).relative_to(root) / d), gitignore_patterns
            )
        ]
        for fname in filenames:
            fp = Path(dirpath) / fname
            try:
                rel = str(fp.relative_to(root))
            except ValueError:
                rel = str(fp)
            if _is_ignored_by_gitignore(rel, gitignore_patterns):
                continue
            if _should_skip_file(fp):
                continue
            files.append(fp)
    return files


def _do_index(
    path: str,
    force: bool = False,
    watch: bool = False,
    provider: str | None = None,
) -> str:
    root = Path(path).resolve()
    if not root.exists():
        return f"Error: path does not exist: {path}"

    # Resolve provider before acquiring lock (may raise)
    try:
        emb_provider = _resolve_provider(str(root), provider, force)
    except (RuntimeError, ValueError) as exc:
        return f"Error: {exc}"

    # Guard: cloud providers must not be used with watch=True
    if watch and emb_provider.name != "local":
        return (
            f"Error: watch=True is not supported with provider '{emb_provider.name}'. "
            "Live sync with cloud embedding providers would incur unbounded API costs. "
            "Use watch=True only with the local provider."
        )

    lock = _get_index_lock(str(root))
    if not lock.acquire(blocking=False):
        return f"Error: indexing of {path} is already in progress"

    try:
        gitignore = _load_gitignore(root)
        all_files = _walk_files(root, gitignore)

        with _get_store(str(root), dims=emb_provider.dims) as store:
            # If force + provider changed dims, drop and recreate the table
            stored_meta = store.get_provider_meta()
            stored_dims = stored_meta["dims"]
            if force and stored_dims != emb_provider.dims:
                store.drop_and_recreate_chunks(emb_provider.dims)

            existing_stats = {} if force else store.get_file_stats()

            # Orphan cleanup: remove chunks for files no longer on disk
            all_file_strs = {str(fp) for fp in all_files}
            orphan_paths = set(existing_stats) - all_file_strs
            for p in orphan_paths:
                store.delete_file_chunks(p)

            files_changed = 0
            files_skipped = 0
            files_skipped_chunking = 0
            total_new_chunks = 0

            for fp in all_files:
                fp_str = str(fp)

                try:
                    stat_res = fp.stat()
                    current_mtime = stat_res.st_mtime
                    current_size = stat_res.st_size
                except OSError:
                    files_skipped += 1
                    continue

                if not force:
                    existing = existing_stats.get(fp_str)
                    if (
                        existing
                        and existing["mtime"] == current_mtime
                        and existing["size"] == current_size
                    ):
                        files_skipped += 1
                        continue

                # File changed or is new, calculate hash for metadata
                file_hash = _sha256_file(fp)
                is_existing = fp_str in existing_stats

                chunks = chunk_file(fp_str)
                if not chunks:
                    files_skipped_chunking += 1
                    # If it existed before but now has no chunks, clear it
                    if is_existing:
                        store.delete_file_chunks(fp_str)
                    continue

                rows = [
                    {
                        "file_path": fp_str,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "content": c.content,
                        "file_hash": file_hash,
                        "chunk_hash": _sha256_str(c.content),
                        "mtime": current_mtime,
                        "size": current_size,
                    }
                    for c in chunks
                ]
                texts = [c.content for c in chunks]

                batch_size = emb_provider.batch_size
                batch_vecs: list[np.ndarray] = []
                for i in range(0, len(texts), batch_size):
                    batch_vecs.append(emb_provider.embed(texts[i : i + batch_size]))
                vecs = np.concatenate(batch_vecs)

                if is_existing:
                    store.replace_file_chunks(fp_str, rows, vecs)
                else:
                    store.add_chunks(rows, vecs)

                files_changed += 1
                total_new_chunks += len(chunks)

            store.touch_last_indexed()
            store.set_provider_meta(emb_provider.name, emb_provider.model, emb_provider.dims)

            # Build ANN index when files changed (enables sub-linear search on large repos)
            if files_changed > 0:
                store.build_index()

        # Save Merkle tree for future change detection
        merkle = _build_merkle_tree(root, gitignore)
        _save_merkle_tree(str(root), merkle)

        # Only start the background watcher when explicitly requested
        if watch:
            _ensure_watcher(root, gitignore)

        return (
            f"Indexed {files_changed} file(s), {total_new_chunks} chunk(s) added, "
            f"{len(orphan_paths)} orphan(s) removed "
            f"({files_skipped} file(s) skipped, unchanged)"
        )
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def index_codebase(
    path: str,
    force: bool = False,
    watch: bool = False,
    provider: str | None = None,
) -> str:
    """
    Index a codebase directory for semantic search.

    Walks the directory, extracts semantic code chunks using AST analysis,
    embeds them locally with sentence-transformers, and stores in a vector index.
    Subsequent calls skip unchanged files (incremental updates).

    Args:
        path: Absolute path to the codebase root directory.
        force: If True, re-index all files even if unchanged.
        watch: If True, start a background watcher for live sync on file changes.
               Not supported with cloud providers (openai, voyage, gemini).
        provider: Embedding provider to use. One of: 'local' (default), 'openai',
                  'voyage', 'gemini'. Cloud providers require the corresponding
                  env var (VECGREP_OPENAI_KEY, VECGREP_VOYAGE_KEY, VECGREP_GEMINI_KEY)
                  and optional dependency (pip install 'vecgrep[openai]' etc.).
                  Once set, switching providers requires force=True to rebuild the index.

    Returns:
        Summary: files indexed, chunks added, files skipped.
    """
    try:
        return _do_index(path, force=force, watch=watch, provider=provider)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def search_code(query: str, path: str, top_k: int = 8) -> str:
    """
    Semantically search an indexed codebase for code relevant to a query.

    Embeds the query and performs cosine similarity search against indexed
    code chunks, returning the most semantically relevant snippets with
    file paths and line numbers.

    If the codebase is not yet indexed, it will be indexed automatically first.

    Args:
        query: Natural language description of what you're looking for.
               E.g. "how does authentication work", "database connection setup"
        path: Absolute path to the codebase root directory.
        top_k: Number of results to return (default 8, max 20).

    Returns:
        Formatted list of matching code chunks with file:line references and
        similarity scores.
    """
    try:
        if len(query) > 500:
            return "Error: query is too long (max 500 characters)"
        if not query.strip():
            return "Error: query must not be empty"

        top_k = max(1, min(top_k, 20))
        root = Path(path).resolve()

        # Check if index has data
        with _get_store(str(root)) as store:
            needs_index = store.status()["total_chunks"] == 0

        index_result: str | None = None
        if needs_index:
            index_result = _do_index(str(root), force=False)

        # Read stored provider to embed the query with the same model
        with _get_store(str(root)) as store:
            if store.status()["total_chunks"] == 0:
                msg = f"No indexable files found in {path}."
                if index_result:
                    msg += f"\n(Index attempt: {index_result})"
                return msg

            stored_provider = store.get_provider_meta()["provider"]
            try:
                emb_provider: EmbeddingProvider = get_provider(
                    stored_provider if stored_provider not in ("unknown",) else "local"
                )
            except (RuntimeError, ValueError):
                emb_provider = get_provider("local")

            query_vec = emb_provider.embed([query])[0]
            results = store.search(query_vec, top_k=top_k)

        if not results:
            return "No results found. Try re-indexing with index_codebase()."

        lines = [f"Top {len(results)} results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            try:
                rel = str(Path(r["file_path"]).relative_to(root))
            except ValueError:
                rel = r["file_path"]
            lines.append(
                f"[{i}] {rel}:{r['start_line']}-{r['end_line']} (score: {r['score']:.2f})"
            )
            lines.append(r["content"])
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_index_status(path: str) -> str:
    """
    Get the status of the vector index for a codebase.

    Args:
        path: Absolute path to the codebase root directory.

    Returns:
        Index statistics: file count, chunk count, last indexed time, disk usage.
    """
    try:
        root = Path(path).resolve()
        with _get_store(str(root)) as store:
            s = store.status()

        size_mb = s["index_size_bytes"] / (1024 * 1024)
        device = _detect_device()
        device_label = {"cuda": "CUDA (GPU)", "mps": "Metal (Apple Silicon)", "cpu": "CPU"}.get(
            device, device
        )
        return (
            f"Index status for: {root}\n"
            f"  Files indexed:  {s['total_files']}\n"
            f"  Total chunks:   {s['total_chunks']}\n"
            f"  Last indexed:   {s['last_indexed']}\n"
            f"  Index size:     {size_mb:.1f} MB\n"
            f"  Provider:       {s['provider']}\n"
            f"  Model:          {s['model']}\n"
            f"  Dimensions:     {s['dims']}\n"
            f"  Compute device: {device_label}"
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def stop_watching(path: str) -> str:
    """Stop watching a codebase for file changes."""
    root = Path(path).resolve()
    root_str = str(root)
    with _LOCK_REGISTRY_LOCK:
        observer = _OBSERVER_REGISTRY.pop(root_str, None)
    if observer:
        observer.stop()
        observer.join(timeout=2)
    _save_watched_paths()
    return f"Stopped watching: {root_str}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _restore_watchers_background() -> None:
    """Restore watchers and run incremental sync in background."""
    watched = _load_watched_paths()
    pruned = []
    for path_str in watched:
        root = Path(path_str)
        if root.exists():
            pruned.append(path_str)
            try:
                _merkle_sync(path_str)
            except Exception:
                _log.exception("Failed to sync %s on startup", path_str)
    if len(pruned) != len(watched):
        _WATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _WATCH_STATE_FILE.write_text(json.dumps(pruned))


def main() -> None:
    t = threading.Thread(target=_restore_watchers_background, daemon=True)
    t.start()
    mcp.run()


if __name__ == "__main__":
    main()
