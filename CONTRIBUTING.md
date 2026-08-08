# Contributing to Pi (Python Port)

This guide covers contributing to the Python port of Pi.

## Philosophy

Pi's core is minimal. If a feature does not belong in the core, it should be an extension.

## Development Setup

```bash
# Clone and install
git clone https://github.com/GuinasCode/python-pi
cd python-pi
uv sync --extra dev

# Run checks before committing
uv run ruff check src/ tests/
uv run pytest tests/ -q
uv run mypy   # checks [tool.mypy] packages in pyproject.toml; see note below
```

`mypy` currently only enforces strict typing on the packages listed under
`[tool.mypy] packages` in `pyproject.toml` (pi_protocol, pi_tui, pi_client,
pi_server, pi_storage_sqlite, pi_agent_core). `pi_ai` and `pi_coding_agent`
are not yet clean under `--strict` — if you're working in those packages,
run `uv run mypy src/pi_ai` (or the relevant path) to check your changes,
and add the package to the list once it passes.

## Code Quality

- Python 3.11+ required
- Type hints everywhere, no bare `Any`
- `from __future__ import annotations` at top of each file
- snake_case for functions and variables
- Line length 120 max (enforced by ruff)
- `dataclass` for simple data containers
- `pydantic v2` for runtime schema validation

## Testing

- Tests use `pytest` and live in `tests/` mirroring `src/` structure
- Each module must have corresponding tests
- Run targeted tests: `uv run pytest tests/pi_protocol/ -v`
- Run full suite: `uv run pytest tests/ -q`

## Dependency Security

- All dependencies pinned to exact versions in `pyproject.toml`
- `uv.lock` is the dependency ground truth
- Run `uv run pip-audit` before releases
- Prefer well-maintained, widely-used packages

## Git Conventions

```
type(scope): concise description

Types: feat, fix, docs, refactor, chore
Scopes: ai, agent, tui, protocol, coding-agent, client, server, storage
```

## License

MIT
