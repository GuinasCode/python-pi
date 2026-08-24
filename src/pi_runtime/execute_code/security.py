"""Slice A4 — policy/budget enforcement for the execute_code RPC bridge,
and the honest filesystem-isolation stance (spec sections 11 and 53).

## Policy/Budget

execute_code must not become a second, parallel authorization mechanism
(spec section 8: "os wrappers das tools devem passar pelo mesmo dispatch
e guardrails do agente principal"). So this module doesn't invent a new
policy concept — it reuses the real `pi_runtime.tools.PolicyEngine` and
`pi_runtime.state.Budget` objects the rest of the runtime already uses,
gating every individual RPC call the same way `pi_runtime.loop.Executor`
gates a direct tool call: validate -> risk-based decision -> ALLOW/DENY.

Two differences from a direct tool call, both deliberate:
  - No `confirm` callback reaches a running child process — there's no
    user to interactively ask mid-script. A PolicyEngine constructed
    without one already fails ASK closed to DENY (see
    PolicyEngine._decide/evaluate), so this is the existing invariant,
    not a special case invented here.
  - RPC tool names (read_file, search_files, list_files, terminal,
    fetch_url) don't match the registry's existing names (read, grep,
    ls, bash, webfetch) — `RPC_TOOL_TO_POLICY_NAME` bridges the two
    without renaming either side's already-established vocabulary.

## Filesystem isolation — honest limitation (spec section 53)

There IS a bypass class here: a script can call Python's own `open()`/
`pathlib.Path.write_text()`/etc. directly, which never goes through
PolicyEngine or the RPC allowlist at all — those only gate calls made
*through* `pi_tools`. This module does not claim to close that class,
because doing so for real needs OS-level sandboxing (a container,
chroot, or Windows AppContainer) that this implementation does not
provide. What it does instead, honestly labeled as best-effort:

  - `mode="strict"` (default): the child's cwd defaults to a fresh
    throwaway directory under this execution's own artifacts_dir
    (never the project root), and its environment is reduced to a
    minimal, explicit allowlist rather than a full copy of the parent's
    environment (see `minimal_environment`) — so secrets/tokens present
    in the parent's env don't leak into the child by default, and a
    relative-path `open("./config.json")` lands in the throwaway
    directory instead of the real project tree.
  - `mode="project"`: an explicit opt-in that restores the project cwd
    and full environment — for scripts that genuinely need project
    access. Never the default; a caller has to ask for it.
  - Neither mode prevents a script from using an *absolute* path (or
    `os.chdir`) to reach anywhere the OS user running Pi can reach.
    That remains true in both modes. Do not represent either mode as a
    security boundary against a malicious script — they reduce
    accidental/default exposure, not adversarial escape.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pi_runtime.execute_code.rpc import RpcError, RpcHandler
from pi_runtime.tools import PolicyEngine, PolicyViolation

if TYPE_CHECKING:
    from pi_runtime.state import Budget

RPC_TOOL_TO_POLICY_NAME: dict[str, str] = {
    "read_file": "read",
    "search_files": "grep",
    "list_files": "ls",
    "terminal": "bash",
    "fetch_url": "webfetch",
}

# Kept intentionally small and explicit — every entry here is something
# the interpreter or a well-behaved script plausibly needs to function
# at all, not "whatever happened to be in the parent's environment".
_MINIMAL_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "PYTHONIOENCODING")


def minimal_environment() -> dict[str, str]:
    """The `mode="strict"` default environment: a small, explicit
    allowlist rather than `dict(os.environ)` — so credentials/tokens the
    parent process happens to hold don't reach the child by default."""
    return {key: os.environ[key] for key in _MINIMAL_ENV_PASSTHROUGH if key in os.environ}


def wrap_handlers_with_policy(
    handlers: dict[str, RpcHandler],
    *,
    policy_engine: PolicyEngine | None,
    budget: Budget | None,
) -> dict[str, RpcHandler]:
    """Returns a new handlers dict where every call first clears policy
    and budget, then — only if both pass — runs the real handler. Wrapping
    happens once per execute_code invocation, at RPC-server construction
    time, not per call, but the checks inside run on every call."""
    if policy_engine is None and budget is None:
        return handlers

    def _wrap(tool_name: str, handler: RpcHandler) -> RpcHandler:
        async def _checked(tool: str, arguments: dict[str, object]) -> object:
            if policy_engine is not None:
                policy_name = RPC_TOOL_TO_POLICY_NAME.get(tool_name, tool_name)
                try:
                    policy_engine.evaluate(policy_name)
                except PolicyViolation as exc:
                    raise RpcError(str(exc), error_type="policy_denied") from exc
            if budget is not None:
                reason = budget.exceeded()
                if reason is not None:
                    raise RpcError(f"budget exceeded: {reason}", error_type="resource_limit")
                budget.record_usage(tool_calls=1)
            return await handler(tool, arguments)

        return _checked

    return {name: _wrap(name, handler) for name, handler in handlers.items()}


__all__ = ["RPC_TOOL_TO_POLICY_NAME", "minimal_environment", "wrap_handlers_with_policy"]
