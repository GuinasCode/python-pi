# Security model: `execute_code`

## What's real

- **RPC authorization**: the parent never trusts the child. Every RPC
  request is validated for a matching per-execution token before any
  tool dispatch happens; a wrong or missing token is rejected as
  `unauthorized` and never reaches a handler. A child cannot gain
  permissions just by knowing a tool's name — only tools explicitly
  registered in the handlers dict for that execution are callable at
  all (`unknown_tool` otherwise).
- **Policy enforcement**: both the `execute_code` invocation itself and
  every individual RPC call inside it go through the same
  `PolicyEngine` the rest of the runtime uses — not a parallel
  authorization path. No confirm callback reaches a running child
  process, so an ASK decision always fails closed to DENY.
- **Budget enforcement**: a shared `Budget` object is consumed by the
  `execute_code` call and by every RPC call inside it — the same
  accounting the rest of the runtime's tool calls use.
- **No recursion, no uncontrolled delegation**: `execute_code` and
  `delegate_task` are never present in the RPC allowlist. This is
  structural (they're simply not in the handlers dict passed to the
  child), not a runtime check that could be bypassed by argument
  manipulation.
- **Bounded output**: stdout/stderr are captured with O(head + tail)
  memory regardless of how much the script actually prints; the full
  stream always lands on disk as an artifact with a running hash.
- **Cleanup on cancellation/timeout**: the RPC server is closed and the
  child process is killed in every exit path (success, timeout,
  cancellation), via an outer `finally` — no orphaned child, no leaked
  socket.

## What's explicitly *not* solved — read this before trusting `mode`

There is a real bypass class here, inherent to running arbitrary Python
as a subprocess: a script can call `open()`, `pathlib.Path`, `os`,
`subprocess`, or any other stdlib facility **directly**, and none of
that goes through `PolicyEngine`, the RPC allowlist, or the Budget at
all. Those guardrails only apply to calls made *through* `pi_tools`.

`mode="strict"` and `mode="project"` reduce **default exposure**, they
are not a sandbox:

- `strict` changes the child's default cwd to a throwaway directory and
  its default environment to a small allowlist — so a careless relative
  `open("./config.json")` or an accidentally-inherited credential in
  the environment doesn't land in/leak from the real project by
  default.
- Neither mode stops a script that uses an **absolute path**, or calls
  `os.chdir()`, from reaching anywhere the OS user running Pi can
  reach. Neither mode stops `subprocess`/`os.system`/`ctypes`/etc.
  invoked directly by the script (as opposed to through `pi_tools`).
- This is **best-effort isolation**, explicitly labeled as such. Real
  protection against a script that is actively trying to escape would
  require OS-level sandboxing — a container, chroot, or Windows
  AppContainer — which this implementation does not provide.

**Do not present `mode="strict"` to a user or another system as a
security boundary against a malicious script.** It is a default-safety
measure against accidental exposure (a script the caller trusts,
running with unnecessarily broad defaults), not a defense against a
script the caller does not trust.

## Practical guidance

- Only run `execute_code` with a script you trust the intent of — the
  same trust boundary as running any other shell command the agent
  proposes.
- Use `mode="project"` deliberately, not by default — it removes the
  cwd/env exposure reduction `strict` provides.
- If you need stronger isolation than this implementation provides,
  wrap the whole Pi process (or at minimum its `execute_code` child) in
  OS-level sandboxing yourself; this codebase does not claim to do it
  for you.
