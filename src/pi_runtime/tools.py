"""Tool Registry + Policy Engine — Fase 3 of the research-first-runtime plan.

Declarative tool capabilities (ToolSpec, plan.md section 3.4) plus a
policy decision flow: tool request -> validation (registered? required
environment available?) -> policy (risk-based ALLOW/ASK/DENY) -> [actual
confirmation UX stays AgentSession's existing PermissionMode/
permission_gate, interactive_mode.py — unchanged, this doesn't replace
it] -> execution -> result.

Real consumer: pi_runtime.loop.Executor, given a PolicyEngine, validates
every tool currently active on the session before handing a step to
AgentSession.prompt() — an unregistered tool, one missing required
environment, or one denied by policy makes the whole step refuse to run
(PolicyViolation propagates to AgentRuntime's existing failure handling),
so "a tool that isn't registered doesn't execute" is literally
enforced, not aspirational. Executor without a PolicyEngine behaves
exactly like Fase 1/2 (opt-in, doesn't change default behavior).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from pi_runtime.state import VerificationResult


class Risk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PolicyMode(str, Enum):
    DEFAULT = "default"  # only HIGH risk asks
    STRICT = "strict"  # MEDIUM and HIGH ask
    PERMISSIVE = "permissive"  # everything allowed (still logged)


@dataclass
class ToolSpec:
    """plan.md 3.4. `input_schema` is intentionally left to the existing
    Tool.parameters JSON schema already defined per-tool in
    pi_coding_agent.agent_session.get_builtin_tools() — duplicating that
    schema here would be exactly the kind of impulse-rewrite Regra 1.1
    warns against; ToolSpec is metadata *about* a tool (risk, side
    effects, environment needs), not a second definition of its shape."""

    name: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    risk: Risk = Risk.NONE
    idempotent: bool = True
    timeout: float | None = None
    cost_hint: float = 0.0
    environment_requirements: list[str] = field(default_factory=list)
    confirmation_required: bool = False
    output_contract: str = ""


class PolicyViolation(Exception):
    """Raised by validate()/evaluate() on an unregistered tool, missing
    environment, or a DENY decision. Left to propagate — never swallowed
    (Regra 1.4) — Executor lets it bubble into AgentRuntime's existing
    exception handling (state.status=FAILED, stop_reason=ERROR)."""


@dataclass
class PolicyAuditEntry:
    """One evaluated decision — the "side effects são rastreados"
    acceptance criterion: every tool request that reaches evaluate() adds
    one of these, whether allowed, asked-and-approved, or denied."""

    tool_name: str
    risk: Risk
    decision: PolicyDecision


class ToolRegistry:
    """Central registry — plan.md 3, Fase 3: "tool não registrada não
    executa"."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def all(self) -> list[ToolSpec]:
        return list(self._specs.values())


# Default risk classification for python-pi's existing builtin tools (see
# pi_coding_agent.agent_session.get_builtin_tools). Mirrors
# pi_coding_agent.permission_mode.MUTATING_TOOL_NAMES for the tools that
# already had a notion of "dangerous" (bash/write/edit), extended to give
# every builtin tool an explicit risk/side-effect/environment
# classification — previously none of this existed anywhere.
DEFAULT_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="read",
        description="Read a file's contents (or an image, as ImageContent).",
        capabilities=["filesystem.read"],
        risk=Risk.NONE,
        idempotent=True,
        environment_requirements=["filesystem"],
        output_contract="text content with line numbers, or an image content block",
    ),
    ToolSpec(
        name="grep",
        description="Search file contents by regex.",
        capabilities=["filesystem.read"],
        risk=Risk.NONE,
        idempotent=True,
        environment_requirements=["filesystem"],
        output_contract="matching lines with file:line prefixes",
    ),
    ToolSpec(
        name="ls",
        description="List directory contents.",
        capabilities=["filesystem.read"],
        risk=Risk.NONE,
        idempotent=True,
        environment_requirements=["filesystem"],
        output_contract="a file/directory listing",
    ),
    ToolSpec(
        name="webfetch",
        description="Fetch a URL over plain HTTP(S).",
        capabilities=["network.read"],
        risk=Risk.LOW,
        idempotent=True,
        environment_requirements=["network"],
        output_contract="extracted page text",
    ),
    ToolSpec(
        name="browser",
        description="Load a URL in a real headless browser.",
        capabilities=["network.read", "browser"],
        risk=Risk.MEDIUM,
        idempotent=True,
        environment_requirements=["network", "browser"],
        output_contract="extracted page text, optionally plus a screenshot image block",
    ),
    ToolSpec(
        name="write",
        description="Create or overwrite a file.",
        capabilities=["filesystem.write"],
        side_effects=["filesystem_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["filesystem"],
        confirmation_required=True,
        output_contract="confirmation the file was written",
    ),
    ToolSpec(
        name="edit",
        description="Replace text within an existing file.",
        capabilities=["filesystem.write"],
        side_effects=["filesystem_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["filesystem"],
        confirmation_required=True,
        output_contract="confirmation the edit was applied",
    ),
    ToolSpec(
        name="bash",
        description="Execute an arbitrary shell command.",
        capabilities=["shell.execute"],
        side_effects=["arbitrary_execution", "filesystem_mutation", "network"],
        risk=Risk.HIGH,
        idempotent=False,
        environment_requirements=["shell"],
        confirmation_required=True,
        output_contract="stdout/stderr and exit code",
    ),
    ToolSpec(
        name="execute_code",
        description="Run a Python script as a subprocess, with an allowlisted RPC bridge back to a subset of tools.",
        capabilities=["shell.execute"],
        side_effects=["arbitrary_execution"],
        risk=Risk.HIGH,
        idempotent=False,
        environment_requirements=["shell"],
        confirmation_required=True,
        output_contract="bounded stdout/stderr preview, exit code, and an artifacts directory",
    ),
    # Per-action browser risk metadata (spec section 34) — "browser" above
    # gates opening a session at all; these gate individual actions inside
    # an already-open session, the same two-tier shape execute_code uses
    # for the top-level execute_code call vs. each RPC call inside it.
    ToolSpec(
        name="browser_navigate",
        description="Navigate the active browser page to a URL.",
        capabilities=["network.read", "browser"],
        risk=Risk.LOW,
        idempotent=True,
        environment_requirements=["network", "browser"],
        output_contract="navigation result with page evidence",
    ),
    ToolSpec(
        name="browser_snapshot",
        description="Capture a bounded accessibility-tree snapshot of the active page.",
        capabilities=["browser"],
        risk=Risk.LOW,
        idempotent=True,
        environment_requirements=["browser"],
        output_contract="bounded text snapshot with element refs",
    ),
    ToolSpec(
        name="browser_click",
        description="Click an element by ref.",
        capabilities=["browser"],
        side_effects=["page_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_type",
        description="Type text into an element by ref, dispatching real key events.",
        capabilities=["browser"],
        side_effects=["page_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_fill",
        description="Set an input/textarea/contenteditable's value directly.",
        capabilities=["browser"],
        side_effects=["page_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_press",
        description="Press a key or key combination on an element.",
        capabilities=["browser"],
        side_effects=["page_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_select",
        description="Select an option in a <select> element by value or label.",
        capabilities=["browser"],
        side_effects=["page_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_scroll",
        description="Scroll an element into view.",
        capabilities=["browser"],
        risk=Risk.LOW,
        idempotent=True,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_wait",
        description="Wait for a URL, text, or load-state condition.",
        capabilities=["browser"],
        risk=Risk.LOW,
        idempotent=True,
        environment_requirements=["browser"],
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_evaluate",
        description="Run arbitrary JavaScript in the page context and return a bounded result.",
        capabilities=["browser"],
        side_effects=["arbitrary_execution", "page_mutation"],
        risk=Risk.HIGH,
        idempotent=False,
        environment_requirements=["browser"],
        confirmation_required=True,
        output_contract="bounded JSON preview, plus an artifact for large results",
    ),
    ToolSpec(
        name="browser_upload",
        description="Set files on a file input element.",
        capabilities=["browser", "filesystem.read"],
        side_effects=["page_mutation"],
        risk=Risk.HIGH,
        idempotent=False,
        environment_requirements=["browser", "filesystem"],
        confirmation_required=True,
        output_contract="interaction result",
    ),
    ToolSpec(
        name="browser_download",
        description="Trigger and save a browser download, with full artifact provenance.",
        capabilities=["browser", "filesystem.write"],
        side_effects=["filesystem_mutation"],
        risk=Risk.MEDIUM,
        idempotent=False,
        environment_requirements=["browser", "filesystem"],
        output_contract="artifact path/filename/mime/size/sha256",
    ),
    ToolSpec(
        name="browser_persistent_profile",
        description="Open a session that loads or saves cookies/storage state across process runs.",
        capabilities=["browser", "filesystem.write"],
        side_effects=["filesystem_mutation"],
        risk=Risk.HIGH,
        idempotent=False,
        environment_requirements=["browser", "filesystem"],
        confirmation_required=True,
        output_contract="a session backed by a persisted storage state file",
    ),
]


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec in DEFAULT_TOOL_SPECS:
        registry.register(spec)
    return registry


class PolicyEngine:
    """Risk-based decision flow. Confirmation UX itself is not this
    class's job — when a decision is ASK, an optional `confirm` callback
    (sync, given the ToolSpec, returns bool) decides whether it becomes
    ALLOW; with no callback, ASK fails closed to DENY (Regra 1.6:
    segurança por default — an unattended runtime with no way to ask
    must never silently treat "should ask" as "should allow")."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        mode: PolicyMode = PolicyMode.DEFAULT,
        available_environment: set[str] | None = None,
        confirm: Callable[[ToolSpec], bool] | None = None,
    ) -> None:
        self._registry = registry
        self._mode = mode
        self._available_environment = (
            available_environment
            if available_environment is not None
            else {"filesystem", "network", "shell", "browser"}
        )
        self._confirm = confirm
        self.audit_log: list[PolicyAuditEntry] = []

    def validate(self, tool_name: str) -> ToolSpec:
        """Raises PolicyViolation if the tool isn't registered, or is
        registered but its required environment isn't available. Never
        silently skips (Regra 1.4)."""
        spec = self._registry.get(tool_name)
        if spec is None:
            raise PolicyViolation(f"tool {tool_name!r} is not registered — refusing to run it")
        missing = [req for req in spec.environment_requirements if req not in self._available_environment]
        if missing:
            raise PolicyViolation(f"tool {tool_name!r} requires environment {missing}, which is not available")
        return spec

    def _decide(self, risk: Risk) -> PolicyDecision:
        if self._mode == PolicyMode.PERMISSIVE:
            return PolicyDecision.ALLOW
        if self._mode == PolicyMode.STRICT:
            return PolicyDecision.ASK if risk in (Risk.MEDIUM, Risk.HIGH) else PolicyDecision.ALLOW
        return PolicyDecision.ASK if risk == Risk.HIGH else PolicyDecision.ALLOW

    def evaluate(self, tool_name: str) -> PolicyAuditEntry:
        spec = self.validate(tool_name)
        decision = self._decide(spec.risk)
        if decision == PolicyDecision.ASK:
            decision = (
                PolicyDecision.ALLOW if (self._confirm is not None and self._confirm(spec)) else PolicyDecision.DENY
            )
        entry = PolicyAuditEntry(tool_name=tool_name, risk=spec.risk, decision=decision)
        self.audit_log.append(entry)
        if decision == PolicyDecision.DENY:
            raise PolicyViolation(
                f"tool {tool_name!r} (risk={spec.risk.value}) denied by policy mode {self._mode.value}"
            )
        return entry

    def evaluate_active_tools(self, tool_names: list[str]) -> list[PolicyAuditEntry]:
        return [self.evaluate(name) for name in tool_names]


def verify_tool_result(spec: ToolSpec, *, is_error: bool, result_text: str) -> VerificationResult:
    """Minimal, honest tool-result verification tying a ToolSpec to
    pi_runtime.state.VerificationResult (reused from Fase 1, not a second
    verification concept) — confirms the call didn't silently fail.
    Deeper, tool-specific verification (did the edit actually apply, is
    the fetched page real content) belongs to the coding/research
    verifiers in later phases; this proves the mechanism exists and is
    wired to a real ToolSpec (Fase 3 acceptance criterion 5)."""
    if is_error:
        return VerificationResult(passed=False, score=0.0, failures=[result_text or f"{spec.name} returned an error"])
    return VerificationResult(passed=True, score=1.0)


__all__ = [
    "DEFAULT_TOOL_SPECS",
    "PolicyAuditEntry",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyMode",
    "PolicyViolation",
    "Risk",
    "ToolRegistry",
    "ToolSpec",
    "default_registry",
    "verify_tool_result",
]
