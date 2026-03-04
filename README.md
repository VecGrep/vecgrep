# VecGrep

[![CI](https://github.com/VecGrep/VecGrep/actions/workflows/ci.yml/badge.svg)](https://github.com/VecGrep/VecGrep/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/VecGrep/VecGrep/branch/main/graph/badge.svg)](https://codecov.io/gh/VecGrep/VecGrep)
[![Discussions](https://img.shields.io/github/discussions/VecGrep/VecGrep)](https://github.com/VecGrep/VecGrep/discussions)

Cursor-style semantic code search as an MCP plugin for Claude Code.

Instead of grepping 50 files and sending 30,000 tokens to Claude, VecGrep returns the top 8 semantically relevant code chunks (~1,600 tokens). That's a **~95% token reduction** for codebase queries.

## How it works

1. **Chunk** — Parses source files with tree-sitter to extract semantic units (functions, classes, methods)
2. **Embed** — Encodes each chunk using the configured embedding provider:
   - **Local** (default) — [`all-MiniLM-L6-v2-code-search-512`](https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512) via fastembed ONNX (~100ms startup, no API key) or PyTorch, with auto device detection (Apple Silicon, CUDA, CPU)
   - **Cloud (BYOK)** — OpenAI, Voyage AI, or Google Gemini via your own API key (higher-quality embeddings, optional)
3. **Store** — Saves embeddings + metadata in LanceDB under `~/.vecgrep/<project_hash>/`; vector dimensions adapt automatically to the chosen provider
4. **Search** — ANN index (IVF-PQ) for fast approximate search on large codebases

Incremental re-indexing via mtime/size checks skips unchanged files.

## Architecture

![VecGrep architecture diagram](.github/res/diagram.jpeg)

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

> **Note:** Python 3.12 is required — `tree-sitter-languages` does not yet have wheels for Python 3.13+.

```bash
pip install vecgrep                        # standard pip
uv tool install --python 3.12 vecgrep     # uv tool (recommended)
```

## Claude Code integration

Run once — works for every project:

```bash
claude mcp add --scope user vecgrep -- vecgrep
```

This installs VecGrep as a persistent binary and registers it in your user config (`~/.claude.json`) so it's available globally across all projects. Starts instantly — no download delay on Claude Code launch.

## Usage with Claude

You don't trigger VecGrep manually - Claude decides when to call the tools based on what you ask.

| What you say to Claude | Tool invoked |
|---|---|
| "Index my project at /Users/me/myapp" | `index_codebase` |
| "How does authentication work in this codebase?" | `search_code` |
| "Find where database connections are set up" | `search_code` |
| "How many files are indexed?" | `get_index_status` |

**Typical first-time flow:**

```
You:    "Search for how payments are handled in /Users/me/myapp"
Claude: [calls index_codebase automatically since no index exists]
Claude: [calls search_code with your query]
Claude: "Here's how payments work — in src/payments.py:42..."
```

After the first index, subsequent searches skip unchanged files automatically — no re-indexing needed unless your code changes.

## Tools

### `index_codebase(path, force=False, watch=False, provider=None)`

Index a project directory. Skips unchanged files on subsequent calls.

```
index_codebase("/path/to/myproject")
# → "Indexed 142 file(s), 1847 chunk(s) added (0 file(s) skipped, unchanged)"

# Use OpenAI embeddings instead of local
index_codebase("/path/to/myproject", provider="openai")
```

**Provider lock**: once a project is indexed with a provider, re-indexing with a different provider requires `force=True` (this rebuilds the vector table with the new embedding dimensions).

**Note:** `watch=True` is only supported with the `local` provider — live sync with cloud providers would incur unbounded API costs.

### `search_code(query, path, top_k=8)`

Semantic search. Auto-indexes if no index exists.

```
search_code("how does user authentication work", "/path/to/myproject")
```

Returns formatted snippets with file paths, line numbers, and similarity scores:

```
[1] src/auth.py:45-72 (score: 0.87)
def authenticate_user(token: str) -> User:
    ...

[2] src/middleware.py:12-28 (score: 0.81)
...
```

### `get_index_status(path)`

Check index statistics, including the embedding provider used.

```
Index status for: /path/to/myproject
  Files indexed:  142
  Total chunks:   1847
  Last indexed:   2026-02-22T07:20:31+00:00
  Index size:     28.4 MB
  Provider:       local
  Model:          isuruwijesiri/all-MiniLM-L6-v2-code-search-512
  Dimensions:     384
```

## Configuration

VecGrep can be tuned via environment variables:

### Local provider

| Variable | Default | Description |
|---|---|---|
| `VECGREP_BACKEND` | `onnx` | Local backend: `onnx` (fastembed, fast startup) or `torch` (sentence-transformers, any HF model) |
| `VECGREP_MODEL` | `isuruwijesiri/all-MiniLM-L6-v2-code-search-512` | HuggingFace model ID (local provider only) |

**Backend comparison:**

| Backend | Startup | PyTorch required | Custom HF models |
|---|---|---|---|
| `onnx` (default) | ~100ms | No | ONNX-exported models only |
| `torch` | ~2–3s | Yes | Any HuggingFace model |

### Cloud providers (BYOK — Bring Your Own Key)

VecGrep supports three cloud embedding providers. Each requires an API key environment variable and the corresponding optional dependency.

| Provider | Env var | Model | Dims | Install extra |
|---|---|---|---|---|
| `openai` | `VECGREP_OPENAI_KEY` | `text-embedding-3-small` | 1536 | `vecgrep[openai]` |
| `voyage` | `VECGREP_VOYAGE_KEY` | `voyage-code-3` | 1024 | `vecgrep[voyage]` |
| `gemini` | `VECGREP_GEMINI_KEY` | `gemini-embedding-exp-03-07` | 3072 | `vecgrep[gemini]` |

**Install cloud extras:**

```bash
# Single provider
uv tool install --python 3.12 'vecgrep[openai]'
pip install 'vecgrep[openai]'

# All cloud providers at once
pip install 'vecgrep[cloud]'
```

**Use a cloud provider:**

```bash
# Set your API key
export VECGREP_OPENAI_KEY=sk-...

# Index with OpenAI embeddings
index_codebase("/path/to/myproject", provider="openai")

# Or tell Claude to use it:
# "Index my project at /path/to/myproject using openai embeddings"
```

**Switch providers** (requires force re-index to rebuild the vector table):

```
index_codebase("/path/to/myproject", provider="voyage", force=True)
```

**Local backend examples:**

```bash
# Use a different model with the torch backend
VECGREP_BACKEND=torch VECGREP_MODEL=sentence-transformers/all-MiniLM-L6-v2 vecgrep

# Use a custom ONNX model
VECGREP_MODEL=my-org/my-onnx-model vecgrep
```

## Supported languages

Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, Ruby, Swift, Kotlin, C#

All other text files fall back to sliding-window line chunks.

## Index location

`~/.vecgrep/<sha256-of-project-path>/index.db`

Each project gets its own isolated index. Delete the directory to wipe the index.

## Acknowledgements

The embedding model used by VecGrep is [`all-MiniLM-L6-v2-code-search-512`](https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512), a model fine-tuned specifically for semantic code search by [@isuruwijesiri](https://huggingface.co/isuruwijesiri).

```bibtex
@misc{all_MiniLM_L6_v2_code_search_512,
  author    = {isuruwijesiri},
  title     = {all-MiniLM-L6-v2-code-search-512},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/isuruwijesiri/all-MiniLM-L6-v2-code-search-512}
}
```

## Community

| | |
|---|---|
| ❓ **Questions** | [Start a Q&A discussion](https://github.com/VecGrep/VecGrep/discussions/new?category=q-a) |
| 💡 **Ideas** | [Share an idea](https://github.com/VecGrep/VecGrep/discussions/new?category=ideas) |
| 🚀 **Show & Tell** | [Share how you use VecGrep](https://github.com/VecGrep/VecGrep/discussions/new?category=show-and-tell) |
| 🐛 **Bugs** | [Open an issue](https://github.com/VecGrep/VecGrep/issues/new) |
