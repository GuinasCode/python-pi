"""Tests for pi_runtime.delegation. Covers Fase 6's acceptance criteria
from plan.md section 10:

- 2+ subagents podem rodar em paralelo
- falha de um subagent não necessariamente falha a tarefa
- resultados podem ser agregados
- custo individual é rastreado (elapsed time)
- contexto recebido é auditável

Unit-level tests stub pi_coding_agent.subagent.runner.spawn_subagent
(the real spawn mechanics are already covered by
tests/pi_coding_agent/test_subagent_registry.py and test_subagent_e2e.py
— these are about DelegationManager's own request-shaping/aggregation/
failure-isolation logic). One real end-to-end test at the bottom proves
the wiring against an actual subprocess (faux provider, no network).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pi_coding_agent.subagent.registry import SubagentHandle, SubagentRegistry, SubagentResult
from pi_runtime.delegation import DelegationManager, DelegationRequest, aggregate_results
from pi_runtime.state import Budget


class _FakeProc:
    def kill(self) -> None:
        return None

    async def wait(self) -> int:
        return 0


async def _fake_spawn_ok(registry: SubagentRegistry, agent: Any, task: str, **kwargs: Any) -> SubagentHandle:
    handle = registry.register(agent.name, task, _FakeProc())
    handle.mark_done(SubagentResult(output=f"did: {task}", exit_code=0, agent_name=agent.name, status="done"))
    return handle


async def _fake_spawn_raises(registry: SubagentRegistry, agent: Any, task: str, **kwargs: Any) -> SubagentHandle:
    raise RuntimeError("spawn failed")


class TestRenderTask:
    def test_bare_objective(self) -> None:
        request = DelegationRequest(objective="find the bug")
        assert request.render_task() == "find the bug"

    def test_full_request_is_auditable(self) -> None:
        request = DelegationRequest(
            objective="research X",
            constraints=["don't touch prod"],
            curated_context=["the codebase uses SQLite"],
            success_criteria=["cite at least 2 sources"],
        )
        rendered = request.render_task()
        assert "research X" in rendered
        assert "don't touch prod" in rendered
        assert "the codebase uses SQLite" in rendered
        assert "cite at least 2 sources" in rendered


class TestDelegateParallel:
    def test_two_delegations_both_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.delegation as delegation_module

        monkeypatch.setattr(delegation_module, "spawn_subagent", _fake_spawn_ok)
        manager = DelegationManager()
        requests = [DelegationRequest(objective="task a"), DelegationRequest(objective="task b")]

        outcomes = asyncio.run(manager.delegate_parallel(requests))

        assert len(outcomes) == 2
        assert all(o.succeeded for o in outcomes)
        assert outcomes[0].result is not None and "task a" in outcomes[0].result.output
        assert outcomes[1].result is not None and "task b" in outcomes[1].result.output

    def test_one_failure_does_not_prevent_the_other_from_succeeding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.delegation as delegation_module

        calls = {"n": 0}

        async def _mixed(registry: SubagentRegistry, agent: Any, task: str, **kwargs: Any) -> SubagentHandle:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return await _fake_spawn_ok(registry, agent, task, **kwargs)

        monkeypatch.setattr(delegation_module, "spawn_subagent", _mixed)
        manager = DelegationManager()
        requests = [DelegationRequest(objective="fails"), DelegationRequest(objective="succeeds")]

        outcomes = asyncio.run(manager.delegate_parallel(requests))

        assert len(outcomes) == 2
        assert not outcomes[0].succeeded
        assert outcomes[0].error is not None
        assert outcomes[1].succeeded

    def test_individual_elapsed_time_is_tracked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.delegation as delegation_module

        monkeypatch.setattr(delegation_module, "spawn_subagent", _fake_spawn_ok)
        manager = DelegationManager()

        outcome = asyncio.run(manager.delegate(DelegationRequest(objective="x")))
        assert outcome.elapsed_seconds >= 0.0


class TestFailureIsolation:
    def test_spawn_exception_becomes_an_outcome_not_a_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.delegation as delegation_module

        monkeypatch.setattr(delegation_module, "spawn_subagent", _fake_spawn_raises)
        manager = DelegationManager()

        outcome = asyncio.run(manager.delegate(DelegationRequest(objective="x")))
        assert not outcome.succeeded
        assert "spawn failed" in (outcome.error or "")


class TestAggregateResults:
    def test_aggregates_successes_and_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pi_runtime.delegation as delegation_module

        calls = {"n": 0}

        async def _mixed(registry: SubagentRegistry, agent: Any, task: str, **kwargs: Any) -> SubagentHandle:
            calls["n"] += 1
            if calls["n"] == 1:
                return await _fake_spawn_ok(registry, agent, task, **kwargs)
            raise RuntimeError("nope")

        monkeypatch.setattr(delegation_module, "spawn_subagent", _mixed)
        manager = DelegationManager()
        outcomes = asyncio.run(
            manager.delegate_parallel(
                [
                    DelegationRequest(objective="a", agent_type="researcher"),
                    DelegationRequest(objective="b", agent_type="coder"),
                ]
            )
        )

        report = aggregate_results(outcomes)
        assert "[researcher]" in report
        assert "[coder] FAILED" in report


class TestDeadlineBudgetBecomesTimeout:
    def test_budget_deadline_shortens_the_spawn_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        import pi_runtime.delegation as delegation_module

        captured: dict[str, float] = {}

        async def _capture_timeout(registry: SubagentRegistry, agent: Any, task: str, **kwargs: Any) -> SubagentHandle:
            captured["timeout"] = kwargs.get("timeout", -1)
            return await _fake_spawn_ok(registry, agent, task, **kwargs)

        monkeypatch.setattr(delegation_module, "spawn_subagent", _capture_timeout)
        manager = DelegationManager()
        request = DelegationRequest(objective="x", budget=Budget(deadline_ts=time.time() + 10))

        asyncio.run(manager.delegate(request))
        assert 0 < captured["timeout"] <= 10


class TestRealEndToEnd:
    """One real subprocess spawn (faux provider, no network) to prove the
    wiring — same isolation technique as test_subagent_e2e.py."""

    @pytest.fixture(autouse=True)
    def _strip_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NVAPI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PI_SKIP_DOTENV", "1")

    def test_real_delegation_completes(self, tmp_path: Any) -> None:
        manager = DelegationManager(cwd=str(tmp_path))
        request = DelegationRequest(objective="say hi", agent_type="researcher")

        outcome = asyncio.run(manager.delegate(request))

        assert outcome.succeeded
        assert outcome.result is not None
        assert "faux provider" in outcome.result.output
