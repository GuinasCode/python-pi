# Python Port Architecture

This document captures the initial architecture decision record for converting `python-pi` from the current TypeScript monorepo to a pure Python project.

**Status note:** the plan below is the original ADR and is largely implemented. `pi_evals`
now exists (`src/pi_evals/`, `eval` extra in `pyproject.toml`) as a port onto
[`pytest-evals`](https://pypi.org/project/pytest-evals/) rather than a Python equivalent
of `vitest-evals` (none exists) — see the README's Evals section for usage. Ported:
`pi_harness.py` (the `AgentSession` adapter), `judges.py` (LLM-as-judge scoring),
`harness_table.py` (comparative baseline/candidate sets), `artifacts.py` (`.eval/`
run/session snapshots), the `pi-evals` CLI, and `smoke.eval.ts` → `test_smoke.py`. Not
ported: `extensions.eval.ts` — it needs an extensions system (`.pi/extensions`) this
Python port doesn't have yet; that's a separate, larger prerequisite, not an evals gap.

## Scope Reality Check

The repository currently contains 1,049 TypeScript/TSX/declaration files and roughly 237k lines across production, tests, examples, scripts, and package configs. The conversion is therefore a full product rewrite, not a mechanical migration.

Initial inventory:

| Area | TS files | Lines |
|---|---:|---:|
| `packages/ai` | 306 | 55,424 |
| `packages/agent` | 63 | 19,667 |
| `packages/tui` | 74 | 29,271 |
| `packages/coding-agent` | 500 | 118,330 |
| `packages/protocol` | 12 | 1,902 |
| `packages/client` | 18 | 2,463 |
| `packages/server` | 38 | 5,682 |
| `packages/storage` | 13 | 2,092 |
| `packages/evals` | 14 | 1,774 |
| `.pi/extensions` | 4 | 692 |
| `scripts` | 6 | 1,133 |

The public README names four primary packages, but the repository also includes protocol, client, server, storage, evals, scripts, examples, and `.pi` extensions that contain production or operational logic. A true zero-TypeScript final state must handle those too.

## Recommended Python Layout

Use a single Python distribution with `src/` subpackages first, then split into separately published distributions only if packaging/release needs require it.

```text
src/
  pi_ai/
  pi_agent_core/
  pi_tui/
  pi_coding_agent/
  pi_protocol/
  pi_client/
  pi_server/
  pi_storage_sqlite/
  pi_evals/
tests/
  pi_ai/
  pi_agent_core/
  pi_tui/
  pi_coding_agent/
  pi_protocol/
  pi_client/
  pi_server/
  pi_storage_sqlite/
  pi_evals/
docs/
scripts/
pyproject.toml
uv.lock
```

Rationale:

- Keeps imports simple while preserving package boundaries.
- Avoids the complexity of a Python multi-wheel workspace during the rewrite.
- Allows later extraction into separate packages if needed.
- Maps cleanly to the existing lockstep release model.

## Tooling Decisions

| TypeScript/Node role | Python replacement | Rationale |
|---|---|---|
| npm workspaces/package-lock | `uv` + `uv.lock` | Fast deterministic locking; exact versions; modern Python equivalent to npm lockfile workflow. |
| Vitest/node:test | `pytest` | Standard Python test runner; supports async tests and rich fixtures. |
| Biome | `ruff` | Fast lint + format; widely used. |
| TypeScript typecheck | `mypy` or `pyright` | Static type checking for typed Python. `pyright` gives stronger editor parity; `mypy` is conventional in Python CI. |
| TypeBox/zod-style runtime schemas | `pydantic v2` | Runtime validation, JSON schema emission, typed models. |
| CLI entrypoint | `typer` or `click` | `typer` for typed command APIs; `click` if lower-level CLI control is needed. |
| TUI differential renderer | `textual` + `rich` | Closest maintained Python stack for terminal UI components/rendering. Some low-level renderer behavior may need custom code. |
| OpenAI JS SDK | `openai` Python SDK | Official provider SDK. |
| Anthropic JS SDK | `anthropic` Python SDK | Official provider SDK. |
| Google GenAI JS SDK | `google-genai` Python SDK | Official provider SDK. |
| AWS Bedrock JS SDK | `boto3`/`botocore` | Canonical AWS Python SDK. |
| Mistral JS SDK | `mistralai` Python SDK | Official Python SDK. |
| YAML | `PyYAML` or `ruamel.yaml` | `ruamel.yaml` if round-trip comments matter; `PyYAML` for simple parsing. |
| glob/minimatch/ignore | `pathspec`, `wcmatch`, stdlib `glob/pathlib` | Gitignore-compatible matching and glob behavior. |
| diff | `difflib` or `unidiff` | Use stdlib first; add dependency only if patch parsing parity requires it. |
| semver | `packaging.version` or `semver` | Prefer `packaging` for Python-native version handling; use `semver` for strict npm semver parity. |
| undici/fetch | `httpx` | Async HTTP, streaming, proxies, timeouts. |
| partial-json | custom incremental parser or maintained parser | Required for streaming structured output parity; evaluate before pinning. |
| proper-lockfile | `filelock` or `portalocker` | Cross-platform lock files. |
| CBOR/protocol codecs | `cbor2` | Common Python CBOR implementation. |
| SQLite storage | stdlib `sqlite3` initially | Avoid dependency unless advanced features require APSW. |

Every dependency must be pinned exactly in `uv.lock` after verifying stable latest version and running `pip-audit`.

## Conversion Order

Dependency graph from current package imports:

```text
pi_tui: no internal package deps
pi_protocol: no internal package deps
pi_ai: self/provider subpath imports only
pi_agent_core: pi_ai
pi_client: pi_protocol
pi_storage_sqlite: pi_agent_core, pi_ai
pi_coding_agent: pi_agent_core, pi_ai, pi_client, pi_protocol, pi_tui
pi_server: pi_ai, pi_coding_agent, pi_protocol
pi_evals: pi_ai, pi_coding_agent
```

Recommended order:

1. `pi_protocol` - schemas, framing, CBOR codec. Small and foundational.
2. `pi_tui` - independent but large; convert core rendering primitives early.
3. `pi_ai` - provider API, model catalog, streaming abstractions.
4. `pi_agent_core` - agent loop, messages, tools, compaction, sessions.
5. `pi_client` - RPC/client transport on top of protocol.
6. `pi_storage_sqlite` - session persistence/search backend.
7. `pi_coding_agent` - CLI, interactive mode, extensions, themes, export, commands.
8. `pi_server` - server/RPC integration.
9. `pi_evals`, scripts, examples, docs, release tooling.

The user's proposed order starts with `pi-ai`; that is acceptable, but `pi_protocol` and `pi_tui` are independent and should be ported before `pi_coding_agent` regardless.

## Public Behavior to Preserve

From repository docs:

- `pi` interactive coding agent CLI.
- `pi -p "query"` / print mode.
- JSON event stream mode.
- RPC/SDK integration.
- Multi-provider LLM support.
- AGENTS.md and SYSTEM.md context loading.
- Skills, prompt templates, extensions, themes, packages.
- Session tree navigation, export/share behavior.
- Context compaction.
- Security model: no built-in sandbox by default; relies on user/process permissions or external sandbox/containerization.
- Supply-chain hardening with exact dependency versions and audit checks.

## Test Strategy

For each converted package:

1. Port the TS tests adjacent to the Python implementation into `tests/<package>/`.
2. Preserve test names semantically, not necessarily literally.
3. Avoid live provider calls in unit tests; use faux/fake providers and recorded fixtures.
4. Run targeted pytest for the package before advancing.
5. Run global checks before any commit:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
uv run pip-audit
```

## Cleanup Criteria

The repository is not considered converted until all are true:

```bash
find . -path ./.git -prune -o \( -name '*.ts' -o -name '*.tsx' -o -name '*.d.ts' \) -print
find . -path ./.git -prune -o \( -name 'package.json' -o -name 'package-lock.json' -o -name 'tsconfig*.json' -o -name 'biome.json' -o -name 'vitest*.ts' -o -name '.npmrc' \) -print
```

Both commands should produce no output, unless an explicitly approved `legacy/` archive is retained.

## Known High-Risk Areas

- `packages/coding-agent`: very large surface area: interactive TUI, extension system, examples, update/release logic, prompt/context plumbing.
- `packages/ai`: provider streaming and tool-call compatibility across many providers; hard to preserve without exhaustive tests.
- `packages/tui`: Python `textual` will not be a line-for-line replacement for the custom renderer; observable UI behavior requires screenshot/terminal fixture tests.
- Generated model data: conversion must decide whether to keep generated JSON data or rewrite model generation scripts in Python.
- Extension ecosystem: TypeScript extensions cannot run in a pure Python runtime unless fully redesigned. The no-shim rule means existing TS extensions must be converted or dropped with explicit approval; they cannot be silently bridged through Node.

## Initial Conclusion

A complete conversion is feasible but is a large rewrite measured in many focused implementation passes. The safe path is incremental package conversion with tests and compatibility fixtures, not one-shot automated translation.
