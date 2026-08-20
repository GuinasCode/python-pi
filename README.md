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
| `pi_evals` | Behavioral, model-backed evals for Pi workflows (`pytest-evals`-based) |

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

## Evals

`pi_evals` runs behavioral, model-backed checks for Pi workflows — an isolated
`AgentSession` adapted to [`pytest-evals`](https://pypi.org/project/pytest-evals/), used to
measure end-to-end behavior and compare prompts/tools/models/other harness configurations.
It's a port of `packages/evals` (the TypeScript project's `vitest-evals`-based suite); see
[ARCHITECTURE.md](ARCHITECTURE.md) for what's ported vs. still pending.

Unlike the rest of the test suite, evals call a real model and cost real money — install
the extra and set a default provider/model to run them:

```bash
uv sync --extra eval

# Runs the eval phase then the analysis phase in one command:
uv run pi-evals --provider openai --model gpt-5.1

# Equivalent via env vars (useful in CI):
PI_PROVIDER=openai PI_MODEL=gpt-5.1 uv run pi-evals

# Scope to one file or a -k expression — forwarded to both phases:
uv run pi-evals --provider openai --model gpt-5.1 tests/pi_evals/test_smoke.py
uv run pi-evals --provider openai --model gpt-5.1 -k "smoke"
```

A plain `uv run pytest` run — the normal test suite, CI included — never triggers evals:
`pytest-evals` only collects `@pytest.mark.eval`-marked tests when invoked with
`--run-eval`, which is exactly what `pi-evals` (or the manually-triggered
[`evals.yml`](.github/workflows/evals.yml) GitHub Actions workflow) does.

Writing an eval:

```python
import asyncio
import pytest
from pi_evals import create_pi_coding_agent_harness

harness = create_pi_coding_agent_harness(no_tools=True)

@pytest.mark.eval(name="my_eval")
def test_answers_a_question(eval_bag):
    result = asyncio.run(harness.run("What is the capital of France?"))
    eval_bag.output = result.output
    assert "Paris" in result.output

@pytest.mark.eval_analysis(name="my_eval")
def test_my_eval_analysis(eval_results):
    assert len(eval_results) > 0
```

See `pi_evals.judges` for LLM-as-judge scoring with observation-only thresholds, and
`pi_evals.harness_table` for comparative baseline/candidate eval sets with pass-rate lift.
Run artifacts (`.eval/runs.jsonl` + per-run session snapshots) are written via
`pi_evals.artifacts.EvalArtifactWriter` — gitignored, same as pytest-evals' own
`test-out/` result dumps.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full conversion plan and dependency mapping.

## License

MIT
