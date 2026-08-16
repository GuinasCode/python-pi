# Pi Agent Harness (Python Port)

A Python port of the [Pi coding agent](https://pi.dev) — a minimal agent harness for terminal-based coding assistance.

## Packages

| Package | Description |
|---------|-------------|
| `pi_ai` | Unified multi-provider LLM API types and provider system |
| `pi_agent_core` | Agent runtime with tool calling and state management |
| `pi_protocol` | Binary protocol: CBOR codec, framing, schema validation |
| `pi_tui` | Terminal UI utilities: fuzzy matching, word navigation, colors |
| `pi_client` | Client for connecting to Pi servers |
| `pi_server` | Server with session management and protocol handling |
| `pi_storage_sqlite` | SQLite session storage backend |
| `pi_coding_agent` | CLI coding agent with built-in tools |
| `pi_memory` | Persistent memory across sessions: SQLite + FTS5 + sqlite-vec hybrid search |

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Interactive mode (TUI not yet implemented)
pi

# Print mode (non-interactive)
pi -p "What is 2+2?"

# JSON event stream mode
pi --mode json "Explain recursion"

# List available models
pi --list-models

# Help
pi --help
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -q

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Security audit
uv run pip-audit
```

## Persistent memory

`pi` keeps a small, curated set of facts (decisions, style preferences, project notes)
that persist *across* sessions — separate from conversation history/compaction. It's on
by default (`memory.enabled` in settings) and stored in a per-user SQLite database at
`~/.pi/memory.db`, created automatically on first use. This file is never checked into
the repo and is excluded via `.gitignore` — it holds your own data, not the project's.

Two tiers, degrading gracefully:

- **Keyword-only** (default, no extra dependencies): full-text search via SQLite FTS5.
- **Hybrid semantic + keyword** (needs the `memory` extra): adds local embeddings via
  EmbeddingGemma-300M (ONNX Runtime, CPU, no GPU/torch required) and vector search via
  `sqlite-vec`, fused with the FTS5 results. Enable it with:

  ```bash
  uv sync --extra memory
  # or: pip install -e ".[memory]"
  ```

  The model weights (~300MB, quantized) download automatically to `~/.pi/models/` the
  first time memory is used with the extra installed — one-time, shown as a status line
  in the interactive UI. No manual download step is needed.

The agent recalls the top matching memories automatically at the start of every turn and
can write new ones proactively via the `remember`/`recall` tools. Relevant settings
(`memory.enabled`, `memory.dbPath`, `memory.topK`, `memory.autoCapture`) live in
`~/.pi/settings.json` alongside the rest of the config.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full conversion plan and dependency mapping.

## License

MIT
