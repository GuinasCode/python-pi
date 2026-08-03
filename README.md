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

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full conversion plan and dependency mapping.

## License

MIT
