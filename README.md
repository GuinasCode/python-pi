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
| `pi_coding_agent` | CLI coding agent with built-in tools, and the extension system (`pi_coding_agent.extensions`) |
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

## Extensions

`pi` loads custom tools (and more) from plain Python files under `.pi/extensions/`
(project-local) or `~/.pi/extensions/` (global) — a port of
`packages/coding-agent/src/core/extensions/`. See
[`docs/extensions.md`](src/pi_coding_agent/docs/extensions.md) for the full authoring
guide and [`examples/extensions/hello.py`](src/pi_coding_agent/examples/extensions/hello.py)
for a runnable example; `pi` points the model at both of these itself when asked to write
an extension.

```python
# .pi/extensions/hello.py
from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent


async def _hello(_tool_call_id, args, _context, _on_update):
    return AgentToolResult(content=[TextContent(text=f"Hello, {args.get('name', '')}!")])


def extension(pi):
    pi.register_tool(
        AgentTool(
            name="hello",
            description="Greets someone by name.",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            execute=_hello,
        )
    )
```

The `pi` object also supports:

- `pi.on(event_name, handler)` — subscribe to `tool_call`/`tool_result` (mutate arguments
  in place, block execution, override results) and `agent_start`/`agent_end`/`turn_start`/
  `turn_end`/`session_start` lifecycle notifications.
- `pi.register_command(name, handler, description=)` — a `/name ...` slash command in
  interactive mode.
- `pi.register_flag(name, type=, default=, description=)` / `pi.get_flag(name)` — declare
  and read a value programmatically (not yet wired to real CLI argv parsing).
- `pi.register_provider(provider)` / `pi.unregister_provider(name)` — add or remove a
  model provider, effective immediately.

A broken extension (bad import, a raised exception, a handler that errors) is recorded as
a load/runtime error and skipped — never crashes the session. `/extensions` in interactive
mode lists what loaded and what didn't. Extension entry points must be synchronous
(`def extension(pi)`, not `async def`) — do async setup lazily inside a tool's `execute()`.

Not implemented: keyboard shortcuts (`register_shortcut`) and rendering/UI hooks (custom
message/markdown/entry renderers, dialogs, widgets, autocomplete, keybindings) — these need
a keybinding dispatcher and a widget/dialog framework this port's `pi_tui` doesn't have yet,
tracked as a separate, larger prerequisite in [ARCHITECTURE.md](ARCHITECTURE.md).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full conversion plan and dependency mapping.

## License

MIT
