"""Tests for pi_runtime.tools (ToolRegistry + PolicyEngine). Covers Fase
3's acceptance criteria from plan.md section 7:

- tool não registrada não executa
- tool perigosa passa por policy
- ferramenta sem ambiente requerido falha de forma explícita
- side effects são rastreados
- tool result pode ser verificado
"""

from __future__ import annotations

import pytest

from pi_runtime.tools import (
    DEFAULT_TOOL_SPECS,
    PolicyDecision,
    PolicyEngine,
    PolicyMode,
    PolicyViolation,
    Risk,
    ToolRegistry,
    ToolSpec,
    default_registry,
    verify_tool_result,
)


class TestToolRegistry:
    def test_unregistered_tool_is_not_registered(self) -> None:
        registry = ToolRegistry()
        assert not registry.is_registered("bash")
        assert registry.get("bash") is None

    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(name="bash", risk=Risk.HIGH)
        registry.register(spec)
        assert registry.is_registered("bash")
        assert registry.get("bash") is spec

    def test_default_registry_covers_every_builtin_tool(self) -> None:
        registry = default_registry()
        for spec in DEFAULT_TOOL_SPECS:
            assert registry.is_registered(spec.name)


class TestUnregisteredToolDoesNotExecute:
    def test_validate_raises_for_unknown_tool(self) -> None:
        engine = PolicyEngine(ToolRegistry())
        with pytest.raises(PolicyViolation, match="not registered"):
            engine.validate("some_unknown_tool")

    def test_evaluate_raises_for_unknown_tool_before_any_decision(self) -> None:
        engine = PolicyEngine(ToolRegistry())
        with pytest.raises(PolicyViolation):
            engine.evaluate("some_unknown_tool")
        assert engine.audit_log == []  # never even reached the decision step

    def test_evaluate_active_tools_stops_at_the_first_unregistered_one(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec(name="read", risk=Risk.NONE))
        engine = PolicyEngine(registry)
        with pytest.raises(PolicyViolation):
            engine.evaluate_active_tools(["read", "not_registered"])


class TestDangerousToolGoesThroughPolicy:
    def test_high_risk_tool_denied_without_a_confirm_callback(self) -> None:
        """DEFAULT mode + HIGH risk = ASK; no confirm callback = fail
        closed to DENY (Regra 1.6: segurança por default)."""
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
        with pytest.raises(PolicyViolation, match="denied"):
            engine.evaluate("bash")

    def test_high_risk_tool_allowed_when_confirmed(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT, confirm=lambda spec: True)
        entry = engine.evaluate("bash")
        assert entry.decision == PolicyDecision.ALLOW

    def test_high_risk_tool_denied_when_confirm_returns_false(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT, confirm=lambda spec: False)
        with pytest.raises(PolicyViolation):
            engine.evaluate("bash")

    def test_low_risk_tool_allowed_without_asking(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
        entry = engine.evaluate("read")
        assert entry.decision == PolicyDecision.ALLOW

    def test_strict_mode_asks_for_medium_risk_too(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.STRICT)
        with pytest.raises(PolicyViolation):
            engine.evaluate("write")  # medium risk, no confirm callback -> denied

    def test_permissive_mode_allows_everything(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.PERMISSIVE)
        entry = engine.evaluate("bash")
        assert entry.decision == PolicyDecision.ALLOW


class TestMissingEnvironmentFailsExplicitly:
    def test_missing_environment_raises_with_a_clear_reason(self) -> None:
        engine = PolicyEngine(default_registry(), available_environment={"filesystem"})
        with pytest.raises(PolicyViolation, match="browser"):
            engine.validate("browser")

    def test_available_environment_passes_validation(self) -> None:
        engine = PolicyEngine(default_registry(), available_environment={"filesystem", "network", "browser"})
        spec = engine.validate("browser")
        assert spec.name == "browser"


class TestSideEffectsAreTracked:
    def test_default_specs_declare_side_effects_for_mutating_tools(self) -> None:
        registry = default_registry()
        assert registry.get("write").side_effects == ["filesystem_mutation"]  # type: ignore[union-attr]
        assert registry.get("bash").side_effects  # type: ignore[union-attr]
        assert registry.get("read").side_effects == []  # type: ignore[union-attr]

    def test_audit_log_records_every_evaluated_decision(self) -> None:
        engine = PolicyEngine(default_registry(), mode=PolicyMode.PERMISSIVE)
        engine.evaluate("read")
        engine.evaluate("write")
        engine.evaluate("bash")
        assert [e.tool_name for e in engine.audit_log] == ["read", "write", "bash"]
        assert [e.risk for e in engine.audit_log] == [Risk.NONE, Risk.MEDIUM, Risk.HIGH]


class TestToolResultCanBeVerified:
    def test_error_result_fails_verification(self) -> None:
        spec = ToolSpec(name="bash", risk=Risk.HIGH)
        result = verify_tool_result(spec, is_error=True, result_text="command failed with exit code 1")
        assert not result.passed
        assert result.failures

    def test_successful_result_passes_verification(self) -> None:
        spec = ToolSpec(name="read", risk=Risk.NONE)
        result = verify_tool_result(spec, is_error=False, result_text="file contents here")
        assert result.passed
