# Browser harness

A real, persistent Playwright browser session — not a browser launched
fresh per tool call. `browser_navigate` followed later by
`browser_click` acts on the same open page, with the DOM state the
navigation left behind still there.

```text
BrowserManager
   |
   +-- BrowserSession (one per browser_session_id)
   |      +-- BrowserContext (Playwright)  <- isolation boundary
   |      +-- Page[] (multiple tabs, one active)
   |      +-- ref map (from the most recent snapshot)
   |
   +-- one shared Browser process backs every session
```

## Lifecycle

```python
from pi_runtime.browser import BrowserManager

async with BrowserManager() as manager:
    session = await manager.open_session()
    await manager.navigate(session.session_id, "https://example.com")
    ...
    await manager.close_session(session.session_id)
# or let the `async with` block close everything on exit
```

- `open_session()` — create. Launches the shared Chromium on first use.
- `attach_session(session_id)` — retrieve and touch an existing session.
- `detach_session(session_id)` — remove from the registry *without*
  closing resources; the caller now owns cleanup.
- `close_session(session_id)` — close the Context and every Page in it.
- `cleanup_expired()` / `close_all()` — reap timed-out sessions, or tear
  down everything (call from a `finally`, at task end, or on a fatal
  error).

## Snapshot and refs

`browser_snapshot` never returns raw HTML — it returns a bounded
accessibility-tree text representation with a stable ref per element:

```text
[heading @e1] Fixture Home
[textbox @e2]
[combobox @e3]
  [option @e4] Red
  [option @e5] Blue
[button @e6] Submit
```

Refs resolve via Playwright's own recommended stable-locator strategy
(accessibility role + accessible name), disambiguated by occurrence
index when two elements share both. **A ref only resolves against the
snapshot that produced it** — call `browser_snapshot` again after any
navigation or DOM-mutating action before using a new ref; an old one
raises `StaleRefError` instead of silently clicking whatever now
happens to match.

## Interactions

`click`, `type_text` (real key events), `fill` (direct value set),
`press`, `select_option`, `scroll_into_view`, `wait_for` (URL substring
/ text appearing / load state — never an arbitrary sleep). Every one
returns a typed `InteractionStatus`
(`success`/`stale_ref`/`not_found`/`timeout`/`policy_denied`/`error`) —
never a bare string a caller has to pattern-match.

## Evaluate, upload, download, tabs

- `evaluate(session_id, script)` — real JS, output-bounded the same way
  `execute_code` bounds Python output: small results ride inline as a
  JSON preview; large ones get a truncated preview plus an artifact
  file pointer.
- `upload(session_id, ref, file_paths)` — sets real files on a file
  input.
- `download_via_click(session_id, ref, artifacts_dir=...)` — waits for
  a triggered download and saves it with full provenance
  (path/filename/mime/size/sha256).
- `new_page` / `list_pages` / `switch_page` / `close_page` — multiple
  tabs per session, one explicit active page.

## Policy, budget, telemetry

Every action is registered in the shared `ToolRegistry` with real risk
metadata (`browser_navigate`/`snapshot`/`scroll`/`wait` = LOW,
`click`/`type`/`fill`/`press`/`select` = MEDIUM,
`evaluate`/`upload`/`persistent_profile` = HIGH, `download` = MEDIUM)
and goes through the same `PolicyEngine` the rest of the runtime uses —
a denial becomes a typed `policy_denied` result, never a crash.

Telemetry is opt-in (`BrowserManager(telemetry_sink=...)`) and never
includes typed values: `fill`/`type_text` always record
`"[value redacted]"` regardless of what was typed, since a generic
harness has no reliable way to classify which field is a password.

## Persistent profiles

Sessions are ephemeral by default. `open_session(storage_state_path=...)`
loads cookies/storage from a file if it already exists;
`save_storage_state(session_id, path)` writes the current session's
state out — nothing is ever persisted automatically. Loading a
persistent profile goes through the higher `browser_persistent_profile`
policy check, not just `browser`.

## Backend

`BrowserManager(backend=BrowserBackend.PLAYWRIGHT_LOCAL)` (default)
launches and owns its own Chromium. `backend=BrowserBackend.CDP` with a
`cdp_url` connects to an already-running browser instead. A persistent,
on-disk Chrome user-data-dir (`browser.profile`) is **not implemented**
— see [`docs/security/browser.md`](security/browser.md) for why.

## execute_code integration (optional, off by default)

```python
from pi_runtime.execute_code.browser_bridge import build_browser_bridge_handlers
from pi_runtime.execute_code.handlers import DEFAULT_HANDLERS

handlers = {**DEFAULT_HANDLERS, **build_browser_bridge_handlers(browser_manager, session.session_id)}
result = await CodeExecutor().execute(
    "from pi_tools import browser_evaluate\nimport json\nprint(json.loads(browser_evaluate('...'))[:5])",
    rpc_handlers=handlers,
)
```

Bound to exactly one pre-opened session chosen by whoever wires the
bridge — a script can read/act on that session's current page, but can
never open a new one itself (that would let a script pick its own
backend/cdp_url, defeating the "explicit, controlled" RPC allowlist).

## Extending it

`src/pi_coding_agent/examples/extensions/browser.py` is a working
example exposing `browser_open`/`navigate`/`snapshot`/`click`/`fill`/
`close` as model-callable tools through the real extension mechanism —
copy it into `.pi/extensions/` to try it.
