"""Real subprocess end-to-end test for the subagent RPC wire protocol.

Unlike test_subagent_registry.py (fake process, deterministic timing),
this actually spawns a child `pi --print --mode rpc` process — proving the
JSON-Lines protocol print_mode._run_rpc_mode speaks is really what
runner._drive expects on the other end, not just that each side's unit
tests pass in isolation.

Two things are needed to force the deterministic, no-network faux
provider (never a real model call) in the child:
  1. NVAPI_KEY/OPENAI_API_KEY actually absent from its environment.
  2. PI_SKIP_DOTENV=1, so the child's own `main()` doesn't re-run
     `load_dotenv(override=False)` and refill whatever got deleted in
     step 1 from this repo's real `.env` file — which is exactly what
     happened before this fix (a real ~60s network call instead of the
     instant faux response). python-dotenv locates `.env` via call-stack
     inspection, not the process's cwd, so overriding the child's working
     directory alone does not prevent this.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_coding_agent.subagent.agent_def import AgentDef
from pi_coding_agent.subagent.registry import SubagentRegistry
from pi_coding_agent.subagent.runner import spawn_subagent

_FAUX_RESPONSE_MARKER = "faux provider"


@pytest.fixture(autouse=True)
def _strip_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVAPI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PI_SKIP_DOTENV", "1")


class TestSubagentEndToEnd:
    def test_spawn_and_wait_returns_faux_output(self, tmp_path: Path) -> None:
        async def _run() -> None:
            registry = SubagentRegistry()
            agent = AgentDef(name="scout", system_prompt="You are a scout.")
            handle = await spawn_subagent(registry, agent, "say hi", timeout=20.0, cwd=str(tmp_path))
            assert handle in registry.list_live() or handle.status != "running"
            result = await handle.wait()
            assert result.agent_name == "scout"
            assert _FAUX_RESPONSE_MARKER in result.output
            assert handle.status == "done"
            assert registry.list_live() == []

        asyncio.run(_run())

    def test_registry_shows_it_as_live_while_running(self, tmp_path: Path) -> None:
        async def _run() -> None:
            registry = SubagentRegistry()
            agent = AgentDef(name="scout", system_prompt="You are a scout.")
            handle = await spawn_subagent(registry, agent, "say hi", timeout=20.0, cwd=str(tmp_path))
            # Immediately after spawn_subagent returns, before awaiting
            # completion, the handle must already be visible.
            assert handle in registry.list_live()
            await handle.wait()

        asyncio.run(_run())

    def test_stop_terminates_a_running_child(self, tmp_path: Path) -> None:
        async def _run() -> None:
            registry = SubagentRegistry()
            agent = AgentDef(name="scout", system_prompt="You are a scout.")
            handle = await spawn_subagent(registry, agent, "say hi", timeout=20.0, cwd=str(tmp_path))
            result = await handle.stop(graceful_timeout=5.0)
            assert result.status in ("killed", "done")  # may finish before stop() lands — both are fine
            assert registry.list_live() == []

        asyncio.run(_run())
