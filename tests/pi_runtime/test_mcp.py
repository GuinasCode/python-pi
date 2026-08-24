"""Tests for pi_runtime.mcp. Covers Fase 13's rule from plan.md section
17: "MCP deve ser adapter sobre o Tool Registry" — MCP tools use
ToolSpec/PolicyEngine, not a second tool semantics — plus the honest
"no real SDK installed" path (Regra 1.5, no faking a working
connection).
"""

from __future__ import annotations

import pytest

from pi_runtime.mcp import (
    InMemoryMcpClient,
    MCPAdapter,
    McpSdkClient,
    MCPToolDescriptor,
    MCPUnavailable,
    mcp_sdk_available,
)
from pi_runtime.tools import PolicyEngine, PolicyMode, PolicyViolation, Risk, ToolRegistry


class TestSdkAvailability:
    def test_sdk_is_genuinely_not_installed_in_this_environment(self) -> None:
        """This repo declares no `mcp` dependency anywhere — confirms
        the real (not simulated) state this module's error path relies
        on."""
        assert mcp_sdk_available() is False

    def test_mcp_sdk_client_raises_unavailable_when_sdk_missing(self) -> None:
        with pytest.raises(MCPUnavailable, match="mcp"):
            McpSdkClient("some-server")


class TestRegisterServerBecomesRealToolSpecs:
    def test_every_mcp_tool_becomes_a_registered_toolspec(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "weather",
            [MCPToolDescriptor(server_name="weather", name="get_forecast", description="get a weather forecast")],
        )
        adapter = MCPAdapter(registry)

        specs = adapter.register_server(client, server_name="weather")

        assert len(specs) == 1
        assert registry.is_registered("mcp:weather:get_forecast")
        spec = registry.get("mcp:weather:get_forecast")
        assert spec is not None
        assert spec.description == "get a weather forecast"

    def test_mcp_tools_default_to_medium_risk_and_require_confirmation(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient("weather", [MCPToolDescriptor(server_name="weather", name="get_forecast")])
        MCPAdapter(registry).register_server(client, server_name="weather")

        spec = registry.get("mcp:weather:get_forecast")
        assert spec is not None
        assert spec.risk == Risk.MEDIUM
        assert spec.confirmation_required

    def test_multiple_servers_can_be_registered_without_name_collisions(self) -> None:
        registry = ToolRegistry()
        adapter = MCPAdapter(registry)
        adapter.register_server(
            InMemoryMcpClient("weather", [MCPToolDescriptor(server_name="weather", name="search")]),
            server_name="weather",
        )
        adapter.register_server(
            InMemoryMcpClient("docs", [MCPToolDescriptor(server_name="docs", name="search")]),
            server_name="docs",
        )
        assert registry.is_registered("mcp:weather:search")
        assert registry.is_registered("mcp:docs:search")


class TestCallGoesThroughPolicy:
    def test_call_without_policy_engine_still_reaches_the_server(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "weather",
            [MCPToolDescriptor(server_name="weather", name="get_forecast")],
            responses={"get_forecast": "sunny"},
        )
        adapter = MCPAdapter(registry)
        adapter.register_server(client, server_name="weather")

        result = adapter.call("mcp:weather:get_forecast", {"city": "SF"})
        assert result == "sunny"
        assert client.calls == [("get_forecast", {"city": "SF"})]

    def test_call_is_evaluated_by_policy_before_execution(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "weather",
            [MCPToolDescriptor(server_name="weather", name="get_forecast")],
            responses={"get_forecast": "sunny"},
        )
        policy = PolicyEngine(registry, mode=PolicyMode.DEFAULT)  # MEDIUM risk -> allowed, not asked
        adapter = MCPAdapter(registry, policy_engine=policy)
        adapter.register_server(client, server_name="weather")

        adapter.call("mcp:weather:get_forecast", {})
        assert len(policy.audit_log) == 1
        assert policy.audit_log[0].tool_name == "mcp:weather:get_forecast"

    def test_denied_call_never_reaches_the_server(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "weather",
            [MCPToolDescriptor(server_name="weather", name="get_forecast")],
            responses={"get_forecast": "sunny"},
        )
        # STRICT mode asks for MEDIUM risk too, with no confirm callback -> denied
        policy = PolicyEngine(registry, mode=PolicyMode.STRICT)
        adapter = MCPAdapter(registry, policy_engine=policy)
        adapter.register_server(client, server_name="weather")

        with pytest.raises(PolicyViolation):
            adapter.call("mcp:weather:get_forecast", {})
        assert client.calls == []

    def test_calling_an_unregistered_mcp_tool_is_a_policy_violation(self) -> None:
        registry = ToolRegistry()
        policy = PolicyEngine(registry, mode=PolicyMode.PERMISSIVE)
        adapter = MCPAdapter(registry, policy_engine=policy)
        with pytest.raises(PolicyViolation):
            adapter.call("mcp:weather:get_forecast", {})


class TestCallRoutingErrors:
    def test_call_with_a_non_mcp_tool_name_is_a_policy_violation(self) -> None:
        registry = ToolRegistry()
        adapter = MCPAdapter(registry)
        with pytest.raises(PolicyViolation, match="not an MCP tool name"):
            adapter.call("read", {})

    def test_call_to_an_unconnected_server_raises_unavailable(self) -> None:
        registry = ToolRegistry()
        from pi_runtime.tools import ToolSpec

        registry.register(ToolSpec(name="mcp:ghost:tool", risk=Risk.MEDIUM))
        adapter = MCPAdapter(registry)
        with pytest.raises(MCPUnavailable):
            adapter.call("mcp:ghost:tool", {})


class TestUnregisterServer:
    def test_unregistering_closes_the_client(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient("weather", [MCPToolDescriptor(server_name="weather", name="get_forecast")])
        adapter = MCPAdapter(registry)
        adapter.register_server(client, server_name="weather")

        adapter.unregister_server("weather")
        assert client.closed

    def test_unregistering_an_unknown_server_is_a_noop(self) -> None:
        registry = ToolRegistry()
        adapter = MCPAdapter(registry)
        adapter.unregister_server("never-registered")  # must not raise
