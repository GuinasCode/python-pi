# execute_code

Runs a Python script as a real subprocess so the model can process tool
output programmatically — filter a large log, summarize test results,
loop/branch over structured data — instead of pulling raw data into the
conversation and reasoning over it token-by-token.

```text
full tool result
      ↓
Python (inside execute_code)
      ↓
filtering / branching / summarizing
      ↓
small result
      ↓
model's context
```

## Capabilities

- Runs arbitrary Python via `sys.executable` (same interpreter/venv as
  the host process), as a child process — not `exec()` in-process.
- The script can call back into a small, explicitly allowlisted set of
  real Pi tools via `pi_tools`:

  ```python
  from pi_tools import read_file, search_files, list_files, terminal, fetch_url

  text = read_file("logs/app.log")
  errors = [line for line in text.splitlines() if "ERROR" in line]
  print("\n".join(errors[-100:]))
  ```

  Each call is a synchronous RPC round-trip to the parent process, over
  a per-execution TCP loopback server (127.0.0.1, OS-assigned port) with
  a random auth token — never HTTP, one JSON line per request/response.
  `execute_code` and `delegate_task` are never in that allowlist: a
  script cannot recursively call itself, and cannot spawn subagents.

- Output is always bounded in memory regardless of how much the script
  prints: a small head+tail preview reaches the result, the complete
  stream is always written to an artifact file on disk with a running
  sha256, byte count, and line count. `output_mode` controls what ends
  up in the preview:

  | mode        | preview contains                                    |
  |-------------|------------------------------------------------------|
  | `head_tail` | bounded head+tail (default, always safe)              |
  | `summary`   | same guarantee as `head_tail` — intent-only distinction |
  | `full`      | the entire stream, up to a 5MB hard cap                |
  | `artifact`  | nothing inline — only the artifact path + stats        |

- Every execution's `status` is one of: `success`, `nonzero_exit`,
  `timeout`, `cancelled`, `policy_denied`, `rpc_error`, `invalid_code`,
  `resource_limit` — never collapsed into a generic error string.

- Artifacts (script, stdout, stderr, `metadata.json` with timing/exit
  code/RPC trace) are written under
  `.pi/runs/<run-id>/execute-code/<execution-id>/` when a `run_id` is
  supplied, or a scratch temp directory otherwise.

## Policy and budget

`execute_code` is a registered tool like any other (`Risk.HIGH`,
`confirmation_required`) — passing a `PolicyEngine` gates the whole
invocation before any subprocess spawns, and every individual RPC call
inside the script is re-checked against the same engine. There is no
interactive confirmation available to a running child: an ASK decision
with no confirm callback fails closed to DENY, same as everywhere else
in the runtime. A shared `Budget` is consumed by the `execute_code` call
itself and by every RPC call made from inside it.

## Modes: `strict` vs `project`

- `strict` (default): the child's cwd is a throwaway directory inside
  the execution's own artifacts dir, and its environment is reduced to
  a small explicit allowlist (`PATH`, `SYSTEMROOT`, etc.) rather than a
  full copy of the parent's environment.
- `project`: an explicit opt-in restoring the real project cwd and the
  full parent environment — use only when the script genuinely needs
  project-level access.

An explicitly-passed `cwd`/`env` always overrides the mode default.

See [`docs/security/code-execution.md`](security/code-execution.md) for
what these modes do **not** protect against.

## Extending it

The allowlist lives in `pi_runtime.execute_code.handlers.DEFAULT_HANDLERS`
— each entry wraps a real, already-tested `pi_coding_agent.tools`
function. Adding a tool means adding a handler there (and a thin wrapper
in `pi_tools`), not inventing a parallel dispatch mechanism.

`src/pi_coding_agent/examples/extensions/execute_code.py` is a working
example that exposes `execute_code` as a model-callable tool through the
real extension mechanism (`pi.register_tool`) — copy it into
`.pi/extensions/` to try it.
