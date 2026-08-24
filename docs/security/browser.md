# Security model: browser harness

## What's real

- **Session isolation**: each `browser_session_id` gets its own
  Playwright `BrowserContext` — a separate cookie jar and storage
  origin from every other session, even though they share one
  underlying Chromium process. Verified against a real browser: a
  cookie set in one session is not visible from another.
- **Ephemeral by default**: no session's cookies/storage survive past
  `close_session`/process exit unless a caller explicitly calls
  `save_storage_state` and later passes that file to
  `open_session(storage_state_path=...)`. Nothing is auto-persisted.
- **Policy enforcement**: every action (open a session, navigate,
  click, evaluate, download, ...) is registered with real risk metadata
  and goes through the same `PolicyEngine` the rest of the runtime
  uses — no confirm callback reaches this harness automatically, so a
  MEDIUM/HIGH-risk action under a mode that asks fails closed to denied
  unless a caller supplies one.
- **Ref staleness is real, not asserted**: a ref resolves through
  Playwright's own role+accessible-name locator strategy computed at
  the moment of the snapshot that produced it. A ref from a previous
  snapshot, or one that no longer matches any element, raises
  `StaleRefError` explicitly — verified by navigating away and
  confirming the old ref no longer resolves, rather than silently
  clicking whatever the same role+name selector matches now.
- **Bounded output everywhere**: `browser_snapshot` never returns raw
  HTML; `browser_evaluate` truncates large results with an artifact
  fallback, the same philosophy `execute_code` uses for its own output.
- **Telemetry never carries typed values**: `fill`/`type_text` always
  log `"[value redacted]"`, unconditionally — not an attempt at
  detecting which fields are sensitive (which a generic harness cannot
  reliably do), but a blanket redaction that can't get that
  classification wrong.

## What's explicitly not solved

- **Page content is untrusted input.** Nothing in this harness
  classifies or sanitizes text extracted from a page. A caller (agent
  runtime, extension, execute_code script) that feeds `browser_snapshot`
  or `browser_evaluate` output back into a model's context is
  responsible for treating it as data from an adversarial third party —
  never automatically as an instruction. This harness gives you
  `Evidence` objects carrying `url`/`retrieved_at`/`extraction_method`
  provenance specifically so a caller *can* attribute and reason about
  where text came from; it does not itself filter or rewrite that text.
- **`mode="cdp"` trusts whatever is listening at `cdp_url`.** Connecting
  over CDP hands this harness's entire action surface (click, type,
  evaluate, cookies) to whatever browser instance answers at that
  endpoint. Point it only at a browser you started and trust, on
  loopback, in an environment where nothing else can reach that port —
  the harness itself does no additional authentication of the CDP
  endpoint (CDP itself has none built in).
- **`browser.profile` (a persistent, on-disk Chrome user-data-dir) is
  not implemented.** `launch_persistent_context` binds one browser
  process to exactly one context, which does not fit this harness's
  "one shared Browser backs every isolated-context session" model —
  building it would mean a second, parallel session lifecycle just for
  this one config value. If you need a durable identity across
  restarts, use `storage_state_path` (cookies/localStorage only, not a
  full profile) instead, or drive a `backend=CDP` connection to a
  browser you launched yourself with a persistent profile directory.
- **`browser_evaluate` runs with the page's own JS privileges.** It is
  registered `Risk.HIGH` and gated by policy, but once approved it can
  do anything the page's own scripts could do — read page-internal
  state, make same-origin requests, manipulate the DOM. Treat approving
  an evaluate call the same way you'd treat approving an arbitrary
  shell command.
- **Downloads/uploads touch the real filesystem.** `download_via_click`
  writes into whatever `artifacts_dir` the caller supplies; `upload`
  reads whatever `file_paths` the caller supplies. Neither validates
  that those paths stay within some sandboxed directory — that's the
  caller's responsibility, same as `execute_code`'s own
  `mode="strict"`/`mode="project"` filesystem story
  (see [`docs/security/code-execution.md`](code-execution.md)).

## Practical guidance

- Only approve `browser_evaluate`/`browser_upload` for scripts/pages
  you trust the intent of.
- Prefer `backend=PLAYWRIGHT_LOCAL` (the default) unless you have a
  specific reason to attach to an external browser; if you do use
  `backend=CDP`, keep the CDP port loopback-only and never expose it on
  a shared network.
- If a page's content will be summarized back to a model, say so
  explicitly in the prompt/context ("the following is untrusted text
  extracted from a web page") rather than assuming the harness already
  marked it as such.
