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
just the Textual app. Also ported: `get_editor_text`/`set_editor_text`/`paste_to_editor` —
direct reads/writes of the prompt editor's text (`paste_to_editor` inserts at the cursor
rather than replacing); the REPL's Noop reports `""` for `get_editor_text` and drops the
setters, since its `input()` prompt is line-by-line and ephemeral, not a persistent buffer.

This closes out every `ExtensionUIContext` method that was just a mechanism decision away.
Not ported, and blocked on real design work rather than just a missing mechanism:
- `setEditorComponent` (swapping the prompt editor for an entirely custom widget) — what a
  custom editor component's contract should even be isn't decided.
- The interactive TUI's own extension management components
  (`extension-input`/`extension-editor`/`extension-selector`) — substantial UI in their own
  right (what they show, how extensions get browsed/toggled/edited), not a small wiring
  exercise like everything above; scoped separately rather than rushed in alongside it.

The classic REPL stays untouched for both of these — genuinely Textual-app UI chrome, unlike
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

## `pi_runtime` — explicit-state agent runtime (research-first-runtime plan)

Started on the `research-first-runtime` branch, following `PLAN.md`/`PROMPT.md` (not
committed to the repo — see the branch history). Fase 1 added `src/pi_runtime/`: `Budget`,
`Goal`, `Plan`/`PlanStep`, `AgentState`, `VerificationResult` (`state.py`) plus
`Planner`/`Executor`/`Verifier`/`Replanner`/`AgentRuntime` (`loop.py`), implementing the
`goal → plan → act → observe → verify → replan | finish` loop as a thin, testable layer
*around* the existing `AgentSession` (`pi_coding_agent/agent_session.py`) — every actual
model/tool call still goes through `AgentSession.prompt()`, unchanged. Fase 1's planner is
deliberately a single-step passthrough (real multi-step decomposition is Fase 4's Research
Engine); its verifier checks `stop_reason`/non-empty text, not content quality. Not yet
wired into the CLI/TUI — later phases (Fase 17) do that once the runtime layer is stable.
See `src/pi_runtime/__init__.py` for the contract summary and `tests/pi_runtime/` for the
Fase 1 acceptance-criteria tests.

Fase 2 added `src/pi_runtime/context.py` (`ContextItem`, `ContextEngine`): a ranked working
set (priority/relevance/freshness score) built from `AgentState` + the live conversation,
replacing "context window as truncation" with selective compaction — goal/constraints/
decisions/unresolved-questions/evidence are always protected from being dropped; only
low-ranked conversational filler is trimmed under budget pressure. Its real consumer is
`pi_runtime.loop.Executor`, which renders a working-set note before each
`AgentSession.prompt()` call and queues it via the existing `queue_steer_message` hook (Fase
1) when there's anything to surface — a state with nothing accumulated yet renders nothing,
so Fase 1's own single-turn behavior is unchanged. `AgentSession` gained a small public
`get_messages()` accessor for this (previously only reachable via the private `_messages`).

Fase 3 added `src/pi_runtime/tools.py`: `ToolSpec` (name/description/capabilities/side
effects/risk/idempotency/timeout/cost hint/environment requirements/confirmation — plan.md
3.4; `input_schema` is deliberately left to each tool's existing JSON schema in
`agent_session.get_builtin_tools()` rather than duplicated), `ToolRegistry`, and
`PolicyEngine` (validation -> risk-based ALLOW/ASK/DENY, ASK failing closed to DENY with no
confirm callback). `default_registry()` classifies every existing builtin tool
(read/grep/ls=none, webfetch=low, browser=medium, write/edit=medium+mutating,
bash=high+mutating) — extending, not replacing, `permission_mode.MUTATING_TOOL_NAMES`'s
existing notion of "dangerous", which stays exactly as-is for the CLI. Real consumer:
`pi_runtime.loop.Executor`, given a `PolicyEngine`, validates every tool active on the
session before a step runs at all — an unregistered tool, missing environment, or denied
tool makes "tool não registrada não executa" literally true, propagating as
`PolicyViolation` into `AgentRuntime`'s existing failure handling. Opt-in (`Executor()` with
no `policy_engine` skips the check entirely, unchanged Fase 1/2 behavior).

Fase 4 added `src/pi_runtime/research.py`: `Evidence`/`Claim`/`ResearchTask`/`ResearchResult`
(plan.md 8) plus `ResearchEngine` and `ResearchVerifier`. Scoped deliberately small — there is
no real web-search API configured anywhere in this repo, so a query planner/search provider
that turns a question into URLs would have to fabricate results to demonstrate anything
(forbidden by Regra 1.3, "não use mocks como produto"). What's real here: `ToolExtractProvider`
wraps the existing, already-tested `fetch_url` tool (not a second HTTP client) to turn a
caller-supplied URL into provenance-preserving `Evidence`; `ResearchEngine.research()` reports
coverage honestly, including recognizing "no evidence gathered" instead of inventing certainty
(plan.md section 6); `ResearchVerifier` rejects any claim marked supported with no evidence
refs, reusing `pi_runtime.state.VerificationResult` rather than a second verification concept.
Claim synthesis from evidence (needs a real LLM call) and a real search provider are explicit
TODOs (Regra 1.5), not faked.

Fase 5 added `src/pi_runtime/browser.py`: `BrowserManager`/`BrowserSessionInfo`/
`NavigationResult`, wrapping the existing, already-tested `browser_fetch_url` tool with
session bookkeeping (create/close/timeout), navigation history, `PolicyEngine` validation
(opt-in, Fase 3), and page -> `Evidence` conversion (Fase 4). `browser_fetch_url` is a
one-shot open->goto->extract->close call, not a persistent Playwright page kept alive across
calls — real click/type/submit interaction needs exactly that, which nothing in this codebase
has yet; building it is a real, separate lift, registered here as a TODO (Regra 1.5) rather
than faked on top of the one-shot path. Session identity in this phase is therefore
bookkeeping (id/timeout/history), not a genuinely persistent browser process. Failures (a
missing Playwright install, a bad URL, an exception Playwright itself raises) never propagate
out of `navigate()` — they become `NavigationResult.error`, satisfying "falhas não derrubam
todo o agente" and "browser deve ser opcional".

Fase 6 added `src/pi_runtime/delegation.py`: `DelegationRequest`/`DelegationOutcome`/
`DelegationManager`/`aggregate_results`. Does not reimplement subagent process spawning —
`pi_coding_agent.subagent` already has a real, tested one (`SubagentRegistry`/
`spawn_subagent`, bounded-concurrency parallel execution, `/agents` list/stop/steer, built
earlier in this same effort). `DelegationRequest` gives plan.md section 7's exact contract
(objective/constraints/curated_context/budget/allowed_tools/success_criteria) a structured,
auditable shape via `render_task()` instead of an ad hoc string assembled per call site;
`DelegationManager.delegate_parallel()` fans out over the real `spawn_subagent` (each call
starts its own OS process immediately, so gathering them is genuinely concurrent) with one
delegation's exception never preventing the others from completing (`asyncio.gather` isn't
even needed for isolation here — each `delegate()` call already catches its own exception
into `DelegationOutcome.error`). Token/dollar cost tracking is a registered TODO (Regra
1.5) — only wall-clock elapsed time is tracked today, since a child process's own usage
doesn't flow back to the parent yet.

Fase 7 added `src/pi_runtime/memory.py`: does not replace `pi_memory` — it's already a real,
tested system (SQLite+FTS5+sqlite-vec hybrid search with automatic lexical fallback, secret
detection, semantic + Soul-specific dedupe). Adds `CognitiveMemoryType`
(Working/Episodic/Semantic/Procedural/User/Project, plan.md 11) mapped onto `pi_memory`'s
existing `MemoryType` values — `PROCEDURAL -> SOUL` is the most meaningful mapping, since
Soul is already documented as "stable, high-priority, low-churn principles", exactly what
procedural memory means in cognitive-architecture terms. `WORKING` has no storage mapping at
all (`write_with_policy` raises `WorkingMemoryNotPersistable`) — working memory is
session-scoped and already lives on `AgentState.working_memory` (Fase 1); persisting it into
`pi_memory` would blur "session scratch space" with "curated facts across sessions."
`write_with_policy()` enforces "memória não deve ser registrada indiscriminadamente" via a
confidence gate (write-time only — `confidence` isn't a stored `MemoryRecord` field, adding
one would need a schema migration bigger than this slice needs) plus dedupe-before-write
reusing `find_similar` unchanged; secret detection still runs, unchanged. `retrieve_ranked()`
wraps `search()` (unchanged, still degrades to lexical without embeddings) with an explicit
freshness score so a stale match doesn't dominate purely on text relevance.

Fase 8 added `src/pi_runtime/learning.py`: `TrajectoryAnalyzer` (deterministic — structural
analysis of an `AgentState`, Fase 1, no LLM call), `generate_memory_candidates()` (reuses Fase
7's cognitive-type vocabulary), and a versioned `SkillRegistry`/`SkillCandidate`. Skill storage
here is intentionally a small in-memory versioned store, not the full Skills System (loader,
selector, progressive disclosure, on-disk persistence) — that's Fase 9's job, building on this
contract. `apply()` only ever appends a new version; `rollback()` appends a copy of an older
version as the new current one rather than deleting history — "nunca alterar skill
automaticamente sem diff/avaliação/rollback/provenance" is structural, not conventional:
`SkillCandidate` always carries a diff and `source_run_id`. `check_regression()` reuses
`VerificationResult` (Fase 1) again rather than a second verification concept.

Fase 9 added `src/pi_runtime/skills.py`: `SkillSelector`/`SkillUsageTracker`. Does not replace
the existing `pi_coding_agent.resource_loader.Skill`/`load_skills` (SKILL.md discovery,
already a working form of progressive disclosure — the system prompt only ever gets
name+description, never the full body, which the model reads via the `read` tool only if it
decides to) — the actual gap was that every discovered skill's name+description was *always*
injected into the prompt regardless of relevance, with no record of what was selected or how
it performed. `SkillSelector` scores relevance with the same deterministic keyword-overlap
approach already used for Soul overlap detection (stopword-filtered, not a new
embedding-based subsystem); only the top-`k` above `min_score` are marked selected.
`SkillUsageTracker` is a plain in-memory selection/outcome log (persisting it into `pi_memory`
is a natural follow-up, not required by this phase). Real consumer: `ContextEngine` (Fase 2)
gained a `skills` parameter on `collect_items()`/`assemble_working_set()` — the "skills"
source Fase 2 explicitly named but deferred (no selection mechanism existed then) is now
real; omitting the parameter behaves exactly as before (verified — Fase 2's own tests pass
untouched).

Fase 10 added `src/pi_runtime/router.py`: `ModelRouter`/`RoutingDecision`/`TaskType`/`Tier`.
Routes a `TaskType` to a desired capability `Tier` (classification/subagent->cheap,
general/coding->medium, planning/research/verification->strong, plan.md's own examples) and
picks from whatever's actually registered in a real `MutableModels` (unchanged, not a second
model registry) — `classify_model_tier()` reads a model's own already-declared `reasoning`/
`context_window` fields rather than guessing. Fallback on a missing tier follows a fixed table
(deterministic, never timing-dependent); `estimate_cost()` reuses `Model.cost`'s existing $-per-
1M-token rates (the same convention `openai_provider`'s `cost_input=2.50` default already
uses); a candidate whose cost would exceed a given `Budget`'s remaining `max_cost` is skipped
in the same fallback order. No matching model at any tier (or none within budget) returns an
explicit `unavailable_reason`, never a bare `None`. Credential pooling and provider-failure
retry are not reinvented — `pi_ai.models`'s `CredentialStore` and `nvidia_models.py`'s
`nvidia/auto` fallback chain already cover those; this module's job is purely deciding which
tier a task needs before either of them runs.

Fase 11 added `src/pi_runtime/environments.py`: `ExecutionBackend` (cwd, command execution,
file read/write, timeout, artifact access — plan.md 15) with `LocalExecutionBackend`
(wraps the existing, already-tested `execute_bash`/`read_file`/`write_file` tools, unchanged),
`DockerExecutionBackend`/`SshExecutionBackend` (real `docker exec`/`ssh` subprocess calls —
genuine execution when the binary is on PATH, an explicit `CommandResult` reporting
unavailability when it isn't, same pattern `browser_fetch_url` already uses for a missing
Playwright install — never simulated). `normalize_path()` uses `posixpath` deliberately, not
`pathlib.Path` (OS-dependent), so a path behaves identically for a local backend on any host
OS or a remote POSIX shell backend, without resolving against the local filesystem.
`SandboxExecutionBackend` is an explicit `NotImplementedError` (Regra 1.5) rather than a fake
sandbox — a real OS-level sandbox is a substantial, platform-specific undertaking with nothing
in this repo to build on yet, and faking a security boundary would be actively misleading.

Fase 12 added `src/pi_runtime/sessions.py`: `RuntimeSessionStore`, `state_to_dict()`/
`state_from_dict()`. Does not replace `SessionManager` (JSONL session store, already handles
create/open/list/`fork_session`, reused unchanged) — the actual gap: `AgentState` (Fase 1) was
never persisted into a session at all, only chat messages were. `state_to_dict()`/
`state_from_dict()` round-trip every field including nested `Goal`/`Plan`/`PlanStep`/`Budget`/
`VerificationResult` and enums through real JSON (`json.loads(json.dumps(...))` is asserted in
the tests, matching what `SessionManager.append_entry` actually does), storing each snapshot
as a `SessionEntry(kind="runtime_state")`. `resume()` is the acceptance criterion made literal:
open the session plus its latest saved state in one call, no manual reconstruction. `fork()`
delegates straight to `SessionManager.fork_session` (already copies every entry, runtime-state
ones included, since it's kind-agnostic); `branch()` truncates a fork at a given `seq` for
inspecting "what if we'd stopped at step N"; `replay()` returns the full ordered lineage of
every snapshot, not just the latest.

Fase 13 added `src/pi_runtime/mcp.py`: `MCPAdapter`/`MCPClient`/`MCPToolDescriptor`. Plan.md's
explicit rule — "MCP deve ser adapter sobre o Tool Registry. Não crie uma segunda semântica de
tool" — is literal here: `register_server()` turns every tool an `MCPClient` reports into a
real `ToolSpec` in the same `ToolRegistry` builtin tools use (Fase 3, unchanged); `call()`
routes through the same `PolicyEngine` before execution, no parallel tool-calling path. No
`mcp` Python SDK is installed or declared as a dependency anywhere in this repo, and no MCP
server is configured to connect to — `McpSdkClient` raises an explicit `MCPUnavailable` the
moment the SDK is missing (verified by a real test confirming `mcp_sdk_available()` is
genuinely `False` in this environment, not simulated), rather than adding the dependency
speculatively or faking a working connection (Regra 1.3/1.5). `InMemoryMcpClient` is
explicitly a test double (Regra 1.3 permits mocks in tests) that validates the adapter's real
registry/policy wiring without needing a live server. Every MCP tool defaults to `Risk.MEDIUM`
+ confirmation-required (same bar as `write`/`edit`/`browser` in `default_registry()`) since
there's no protocol-level signal to classify an external, unreviewed MCP tool more precisely.

Fase 14 added `src/pi_runtime/scheduler.py`: `Scheduler`/`Job`/`JobStore`/`Schedule`.
plan.md's explicit rule — "Scheduler executa o mesmo runtime normal. Não criar um 'segundo
agent'" — is literal: `run_job()` drives a real `pi_runtime.loop.AgentRuntime` (Fase 1)
against a real `AgentSession`, the same execution path any other run uses. Persistence reuses
`SessionManager` (unchanged) rather than a new database — every job is a
`SessionEntry(kind="job")` in one dedicated session, saved append-only (same pattern as Fase
12's `RuntimeSessionStore`: `save()` never overwrites, `all_jobs()` resolves to the latest
version per `job_id`), proven by reloading jobs through a second `JobStore` instance over the
same `SessionManager`. `Schedule` supports one-shot (`at`) and fixed-interval recurring
(`interval_seconds`) — not a real cron parser, since a correct one is a solved problem
(`croniter` and similar) not currently a dependency of this repo, and faking one would violate
Regra 1.5; fixed-interval recurrence honestly covers the same "recurring job" acceptance
criterion. A failed run retries by rescheduling (`SCHEDULED`, `next_run_at=now`) up to
`max_retries` rather than looping in-process, so a crashed scheduler process never loses a
pending retry — it's already persisted. `cancel()` only affects jobs still pending
(`SCHEDULED`/`RUNNING`); a job already finished can't be cancelled again.

Fase 15 added `src/pi_runtime/telemetry.py`: `Trace`/`Span`/`CostRecord`/`TelemetryRecorder`.
Real consumer: `TelemetryRecorder.attach()` hooks into `AgentSession.on_event()` (the existing,
already-tested event bus — `_emit()`/`on_event()`, unchanged, not a second event system) to
record real `tool_call` spans from the same `tool_call_start`/`tool_call_end` events
`interactive_mode._handle_event` already renders; `pi_runtime.loop.AgentRuntime.run()` gained
an optional `telemetry` parameter (every earlier-phase caller passes nothing and is unaffected
— verified, all prior tests pass untouched) that records one top-level `agent_run` span with
the run's `stop_reason`. Every span shares one `Trace`'s `trace_id`; `record_cost()` reuses
`pi_runtime.router.estimate_cost` (Fase 10, unchanged) rather than a second cost model.
`to_json()`/`to_dict()` answer plan.md's literal acceptance-criterion questions from one Trace
— quanto custou, quanto tempo levou, quais tools usou, onde falhou, por que terminou — proven
end to end in `TestReconstructTheRunFromTelemetry`.

Fase 16 added `src/pi_runtime/evals.py`: mechanical behavioral metrics across the 5 suites
plan.md section 20 names (Agent/Research/Memory/Delegation/Skills). Does not replace
`pi_evals` (already a real harness — `pi_harness.py` wraps `AgentSession`, `judges.py` does
LLM-as-judge scoring, `harness_table.py` does baseline/candidate comparisons — for evaluating
chat-style output quality). The gap: plan.md's own explicit rule against measuring only
"output contains X" needed metrics computed directly from `pi_runtime`'s own structured data
— `AgentState` (Fase 1), `ResearchResult` (Fase 4), `RankedMemory` (Fase 7), `DelegationOutcome`
(Fase 6), `SkillSelection` (Fase 9) — none of which need an LLM judge, since they're
deterministic functions over data every earlier phase already produces (e.g.
`research_citation_precision` catches a claim marked supported with no `evidence_refs`, the
exact rule from Fase 4's own docstring; `skills_regression_rate` reuses Fase 8's
`check_regression` rather than a second regression concept).

## Initial Conclusion

A complete conversion is feasible but is a large rewrite measured in many focused implementation passes. The safe path is incremental package conversion with tests and compatibility fixtures, not one-shot automated translation.
