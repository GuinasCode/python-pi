# Python Port Architecture

This document captures the initial architecture decision record for converting `python-pi` from the current TypeScript monorepo to a pure Python project.

**Status note:** the plan below is the original ADR and is largely implemented. `pi_evals`
now exists (`src/pi_evals/`, `eval` extra in `pyproject.toml`) as a port onto
[`pytest-evals`](https://pypi.org/project/pytest-evals/) rather than a Python equivalent
of `vitest-evals` (none exists) — see the README's Evals section for usage. Ported:
`pi_harness.py` (the `AgentSession` adapter), `judges.py` (LLM-as-judge scoring),
`harness_table.py` (comparative baseline/candidate sets), `artifacts.py` (`.eval/`
run/session snapshots), the `pi-evals` CLI, `smoke.eval.ts` → `test_smoke.py`, and
`extensions.eval.ts` → `test_extensions.py` (see below — the extension system this
needed has since been ported).

**Extension system** (`src/pi_coding_agent/extensions/`, port of
`packages/coding-agent/src/core/extensions/`): ported in phases, tracked here since the
original is a ~4,000-line, 40+-event plugin SDK — too large to port in one pass and not
all of it has a Python-side prerequisite yet.

Ported: discovery/dynamic loading of `.pi/extensions/*.py` (no jiti/virtual-module
machinery needed — Python extensions just `import pi_ai`/`pi_coding_agent` directly);
`ExtensionRunner` wired into `AgentSession` (tool registration, `reload()`, error
reporting); the "Pi documentation" system-prompt section pointing at a bundled authoring
guide + example; the event bus (`tool_call`/`tool_result`/agent+turn+session lifecycle —
the highest-leverage subset of the original's 30+ event types, not all of them);
`register_command`/`register_flag`+`get_flag`; `register_provider`/`unregister_provider`;
`register_shortcut`/`register_theme` (Textual-app only — see below); Phase G's rendering
hooks — `register_markdown_transformer`, `register_message_renderer`,
`register_entry_renderer` — wired into `InteractiveSession._flush_text_block`/`_handle_event`
directly, so unlike the Textual-only items above these work in *both* front-ends (the classic
REPL and the Textual app alike), since they transform/replace what gets printed rather than
being chrome specific to one UI.
See the README's Extensions section for usage.

The Textual app (`--ui-mode fullscreen`/`--alt`, `pi_coding_agent/tui_app.py`) is built up in
phases (T0-T6, then G/H) on top of Textual 8.2.8 rather than a from-scratch port of the TS
`packages/tui`. So far: T0 (app shell — transcript/input/footer, reusing
`InteractiveSession` via an `OutputSink`), T1 (multi-line prompt editor), T2 (modal dialogs —
`ConfirmDialog`, permission-mode confirmation), T3 (keybinding dispatcher — `PiApp.on_key`
matches a pressed key against every `pi.register_shortcut()` registration and fires the first
match; a no-op in the classic REPL, which has no dispatcher to call it from), T4 (slash-command
autocomplete — an `OptionList` popup driven by plain prefix matching over built-in and
extension-registered command names, since Textual's `Suggester` only attaches to `Input`, not
the `TextArea` the prompt editor is built on; Tab accepts, Escape dismisses, Up/Down move the
highlight), T5 (color themes — `pi.register_theme()` registers a `textual.theme.Theme` onto
the app; switchable today via Textual's built-in command palette, Ctrl+P), T6 (live-updating
footer — `InteractiveSession._on_status_change` fires after every streamed event, so the
footer reflects "thinking...", "running: `<tool>`", and "ready" transitions as they happen
during a turn instead of only right before/after submission).

This closes out the T0-T6 foundation phases as originally scoped. Deliberately not added
speculatively during T6: a generic custom header/extra-widget slot — nothing concrete needs
one yet (no extension API exposes it), and adding one now would be exactly the "half-wiring
registration APIs with nothing real behind them" this section warns against below; it lands
in Phase H alongside `ExtensionUIContext.setHeader`/`setWidget` instead, once there's a real
caller.

Phase H (`ExtensionUIContext`) is in progress. Ported so far: `select()`/`confirm()`/
`input()`/`notify()` — `ExtensionContext.ui`, defaulting to `NoopExtensionUIContext`
(`extension_ui.py`; the classic REPL keeps this default, so these four report
"cancelled"/"declined" there rather than blocking or hand-building a REPL-side prompt UI for
a surface the Textual app already covers), swapped for `TextualExtensionUIContext`
(`tui_app.py`) in the Textual app — `select`/`confirm`/`input` push `SelectDialog`/
`ConfirmDialog`/`InputDialog` (`dialogs.py`) and await the result; `notify` uses Textual's own
toast mechanism directly. Also ported: `set_header`/`set_footer`/`set_title`/`set_widget` —
three reserved, hidden-until-set `Static` slots in `PiApp.compose()` (`#ext-header` docked
top; `#ext-footer` alongside — not replacing — the built-in status footer; `#ext-widget` just
above the prompt editor) plus `app.title`, all no-ops in the classic REPL the same way the
prompts above are. Also ported: `get_theme`/`set_theme` (thin wrapper over `app.theme` on top
of T5's registration); `add_autocomplete_provider` (extra suggestions merged into T4's popup
alongside the built-in slash-command matches); `get_tools_expanded`/`set_tools_expanded`
(whether a tool-call transcript entry's full result preview prints or just its summary line)
— unlike every other Phase H method, this one is consulted directly by
`InteractiveSession._handle_event`, so it's real, working state in the classic REPL too, not
just the Textual app.

Not ported, and blocked on a real prerequisite rather than just unscheduled:
- `pasteToEditor`/`setEditorText`/`getEditorText`/`setEditorComponent` and the interactive
  TUI's own extension management components
  (`extension-input`/`extension-editor`/`extension-selector`) — the management screens in
  particular are substantial UI in their own right, not a small wiring exercise like Phase H's
  items above, and are being scoped separately rather than rushed in alongside them. The
  classic REPL stays untouched for all of these — genuinely Textual-app UI chrome, unlike
  Phase G's rendering hooks (and Phase H's `tools_expanded`, which apply to both front-ends).

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
