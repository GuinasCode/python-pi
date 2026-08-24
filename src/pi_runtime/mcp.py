"""MCP — Fase 13 of the research-first-runtime plan.

plan.md's explicit rule: "MCP deve ser adapter sobre o Tool Registry.
Não crie uma segunda semântica de tool." — every MCP tool this module
discovers becomes a real `pi_runtime.tools.ToolSpec` in the same
`ToolRegistry` used by builtin tools (Fase 3), and every call goes
through the same `PolicyEngine` before execution — not a parallel
tool-calling path.

No `mcp` Python SDK is installed or declared as a dependency anywhere in
this repo, and there is no MCP server configured to connect to. Adding
that dependency speculatively, or faking a working MCP connection, would
violate Regra 1.5 (registre como TODO, não desvie) and Regra 1.3 (não
use mocks como produto). What's real here instead: `MCPClient` is the
protocol any real transport implements; `McpSdkClient` is a thin wrapper
around the real `mcp` SDK that raises a clear `MCPUnavailable` error
(never a fake success) when the SDK isn't installed — proven by a real
test importing this module with the SDK genuinely absent.
`InMemoryMcpClient` is a test double representing a fake in-memory MCP
server (explicitly a test fixture, not a production path — Regra 1.3
allows mocks in tests) used to validate the adapter's own registry/
policy wiring without needing a live server or the real SDK.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol

from pi_runtime.tools import PolicyEngine, PolicyViolation, Risk, ToolRegistry, ToolSpec


class MCPUnavailable(Exception):
    """Raised when an MCP operation needs the real `mcp` SDK (or a live
    server connection) and it isn't available. Never silently degrades
    to a no-op — the caller must handle this explicitly."""


@dataclass
class MCPToolDescriptor:
    """What an MCP server reports about one of its tools — the shape
    the real `mcp` SDK's `list_tools()` response has (name/description/
    inputSchema), kept minimal and untyped-for-schema (a JSON schema
    dict, same as pi_ai.Tool.parameters) rather than redefined."""

    server_name: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPClient(Protocol):
    """What any real transport (stdio, SSE, the real SDK's
    ClientSession, or a test double) must implement."""

    def list_tools(self) -> list[MCPToolDescriptor]: ...
    def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...
    def close(self) -> None: ...


def mcp_sdk_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


class McpSdkClient:
    """Thin wrapper around the real `mcp` SDK. Raises MCPUnavailable
    immediately on construction if the SDK isn't installed — this is the
    one place that decides "real connection or explicit failure", never
    "pretend it worked"."""

    def __init__(self, server_name: str, *_args: Any, **_kwargs: Any) -> None:
        if not mcp_sdk_available():
            raise MCPUnavailable(
                f"cannot connect to MCP server {server_name!r}: the 'mcp' package is not installed. "
                "Install it (e.g. `pip install mcp`) to use a real MCP server connection."
            )
        # A real implementation would establish the SDK's ClientSession
        # here. Left unimplemented beyond the availability check — no
        # server is configured anywhere in this repo to connect to and
        # exercise against, so building the rest now would be
        # untested, speculative code (Regra 1.2/1.5).
        raise NotImplementedError(
            "McpSdkClient's connection logic is a registered TODO (Regra 1.5) pending a "
            "configured MCP server to build and test against. mcp_sdk_available() and the "
            "MCPUnavailable-on-missing-SDK behavior above are real and tested."
        )


class InMemoryMcpClient:
    """Test double — an in-memory fake MCP server. Explicitly for tests
    (Regra 1.3 permits mocks there): validates MCPAdapter's registry/
    policy/execution wiring without needing a live server or the real
    SDK installed."""

    def __init__(
        self, server_name: str, tools: list[MCPToolDescriptor], *, responses: dict[str, str] | None = None
    ) -> None:
        self.server_name = server_name
        self._tools = tools
        self._responses = responses or {}
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[MCPToolDescriptor]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self._responses.get(name, "")

    def close(self) -> None:
        self.closed = True


def _tool_spec_from_descriptor(descriptor: MCPToolDescriptor) -> ToolSpec:
    """MCP tools are external, unreviewed code an operator chose to
    connect — MEDIUM risk by default (same bar as pi_runtime.tools.
    default_registry's `write`/`edit`/`browser`), not NONE. There's no
    reliable signal in a plain MCP tool descriptor to classify risk more
    precisely than that without the server itself annotating it, which
    the base MCP protocol doesn't standardize."""
    return ToolSpec(
        name=f"mcp:{descriptor.server_name}:{descriptor.name}",
        description=descriptor.description,
        capabilities=[f"mcp.{descriptor.server_name}"],
        side_effects=["external_mcp_call"],
        risk=Risk.MEDIUM,
        idempotent=False,
        confirmation_required=True,
        output_contract="text returned by the MCP server's tool call",
    )


class MCPAdapter:
    """The Tool Registry adapter plan.md asks for. register_server()
    turns every tool a connected MCPClient reports into a real ToolSpec
    in the given registry; call() routes execution through the given
    PolicyEngine exactly like any builtin tool — same ToolSpec,
    PolicyEngine, and (via pi_runtime.tools.verify_tool_result) Verifier
    contracts, no second tool semantics."""

    def __init__(self, registry: ToolRegistry, *, policy_engine: PolicyEngine | None = None) -> None:
        self._registry = registry
        self._policy_engine = policy_engine
        self._clients: dict[str, MCPClient] = {}

    def register_server(self, client: MCPClient, *, server_name: str) -> list[ToolSpec]:
        self._clients[server_name] = client
        specs = [_tool_spec_from_descriptor(d) for d in client.list_tools()]
        for spec in specs:
            self._registry.register(spec)
        return specs

    def unregister_server(self, server_name: str) -> None:
        client = self._clients.pop(server_name, None)
        if client is not None:
            client.close()

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Fase 13's core rule made concrete: policy runs before an MCP
        tool executes, same as any other tool (Fase 3's PolicyEngine,
        unchanged, not a second check)."""
        if self._policy_engine is not None:
            self._policy_engine.evaluate(tool_name)

        if not tool_name.startswith("mcp:"):
            raise PolicyViolation(f"{tool_name!r} is not an MCP tool name (expected 'mcp:<server>:<tool>')")
        _, server_name, mcp_tool_name = tool_name.split(":", 2)
        client = self._clients.get(server_name)
        if client is None:
            raise MCPUnavailable(f"no connected MCP server named {server_name!r}")
        return client.call_tool(mcp_tool_name, arguments)


__all__ = [
    "InMemoryMcpClient",
    "MCPAdapter",
    "MCPClient",
    "MCPToolDescriptor",
    "MCPUnavailable",
    "McpSdkClient",
    "mcp_sdk_available",
]
