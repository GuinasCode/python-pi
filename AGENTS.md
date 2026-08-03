# Development Rules

## Conversational Style

- Keep answers short and concise
- No emojis in commits, issues, PR comments, or code
- Technical prose only, be direct

## Code Quality

- Type hints everywhere, no bare `Any` without justification
- `from __future__ import annotations` at top of each file
- snake_case for functions and variables
- Line length 120 max (enforced by ruff)
- Use `dataclass` for simple data containers
- Use `pydantic v2` for runtime schema validation

## Commands

- After code changes: `uv run ruff check src/ tests/`
- Run tests: `uv run pytest tests/ -q`
- Type check: `uv run mypy src/`
- Security audit: `uv run pip-audit`

## Git

- Stage explicit paths (`git add <path1> <path2>`); never `git add -A`
- Message format: `{feat,fix,docs}[(scope)]: <message>`
- Never force push
