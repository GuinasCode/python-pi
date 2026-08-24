"""Hardening — Fase 18 of the research-first-runtime plan.

Adversarial/edge-case tests across pi_runtime as a whole (plan.md
section 22): secret scanning, permission tests, sandbox tests, path
handling, command injection defense, prompt injection defense, MCP
permission tests, network/timeout failure, malformed data, session
corruption, browser/subagent failure isolation, duplicate execution.

No new src/ behavior is invented here purely to make a test pass — where
an earlier phase already provides the real protection, these tests
exercise it under adversarial input; where a phase explicitly declined
to fake something (SandboxExecutionBackend, a real cron parser, a real
MCP SDK connection), that honest gap is what's tested, not a fabricated
guarantee.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest

from pi_coding_agent.session_manager import SessionEntry, SessionManager
from pi_runtime import environments
from pi_runtime.context import ContextEngine
from pi_runtime.learning import SkillRegistry
from pi_runtime.mcp import InMemoryMcpClient, MCPAdapter, MCPToolDescriptor, MCPUnavailable
from pi_runtime.memory import CognitiveMemoryType, write_with_policy
from pi_runtime.scheduler import JobStore, Schedule, Scheduler, _job_from_dict
from pi_runtime.sessions import RuntimeSessionStore
from pi_runtime.state import AgentState, Goal, RunStatus
from pi_runtime.tools import PolicyEngine, PolicyMode, PolicyViolation, Risk, ToolRegistry, default_registry

# --- Security: secret scanning ---------------------------------------------


class _NoEmbeddings:
    def is_available(self) -> bool:
        return False


class TestSecretScanning:
    def test_secret_is_blocked_before_it_is_ever_persisted(self) -> None:
        from pi_memory.store import MemoryStore, SecretDetectedError

        store = MemoryStore(":memory:", embeddings=_NoEmbeddings())  # type: ignore[arg-type]
        raised = False
        try:
            write_with_policy(
                store,
                cognitive_type=CognitiveMemoryType.USER,
                title="my key",
                content="use sk-abcdefghijklmnopqrstuvwx1234567890 for auth",
                confidence=0.9,
            )
        except SecretDetectedError:
            raised = True
        assert raised
        assert store.search("my key", top_k=5) == []

    def test_secret_embedded_mid_sentence_is_still_caught(self) -> None:
        from pi_memory.store import MemoryStore, SecretDetectedError

        store = MemoryStore(":memory:", embeddings=_NoEmbeddings())  # type: ignore[arg-type]
        raised = False
        try:
            write_with_policy(
                store,
                cognitive_type=CognitiveMemoryType.PROJECT,
                title="deploy note",
                content="remember the private key: -----BEGIN PRIVATE KEY----- abc123",
                confidence=0.9,
            )
        except SecretDetectedError:
            raised = True
        assert raised


# --- Security: permission / policy tests ------------------------------------


class TestPermissionHardening:
    def test_high_risk_tool_fails_closed_with_no_confirm_callback(self) -> None:
        policy = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
        with pytest.raises(PolicyViolation):
            policy.evaluate("bash")

    def test_a_confirm_callback_that_raises_does_not_silently_allow(self) -> None:
        """A broken confirm callback must not accidentally fail open."""

        def _broken_confirm(spec: object) -> bool:
            raise RuntimeError("confirm UI crashed")

        policy = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT, confirm=_broken_confirm)
        with pytest.raises(RuntimeError):
            policy.evaluate("bash")
        # crucially: nothing was allowed as a side effect of the crash
        assert policy.audit_log == []

    def test_permissive_mode_still_requires_registration(self) -> None:
        """PERMISSIVE allows any *registered* risk level — it must not
        also skip the "is this even a real tool" check."""
        policy = PolicyEngine(ToolRegistry(), mode=PolicyMode.PERMISSIVE)
        with pytest.raises(PolicyViolation, match="not registered"):
            policy.evaluate("bash")


# --- Security: sandbox is honestly absent, never silently available --------


class TestSandboxHonesty:
    def test_sandbox_backend_always_refuses_never_pretends_to_isolate(self) -> None:
        from pi_runtime.environments import SandboxExecutionBackend

        with pytest.raises(NotImplementedError):
            SandboxExecutionBackend()


# --- Security: command injection defense-in-depth ---------------------------


class TestCommandInjectionDefense:
    def test_docker_and_ssh_backends_never_use_shell_true(self) -> None:
        """subprocess with shell=True string-concatenation is the classic
        injection vector — _shell_out always passes a list of discrete
        argv entries, confirmed by reading the actual call sites rather
        than trusting a docstring."""
        source = inspect.getsource(environments)
        assert "shell=True" not in source

    def test_docker_backend_command_is_one_argv_entry_not_concatenated(self) -> None:
        """The user-supplied `command` string is passed as a single argv
        element to `sh -c`, never spliced into a larger shell string on
        our side — whatever happens after that is the remote shell's own
        job (inherent to any remote-command-execution backend, not a gap
        introduced by this wrapper)."""
        backend = environments.DockerExecutionBackend(container="test")
        backend._binary_available = lambda: True  # type: ignore[method-assign]
        captured: dict[str, list[str]] = {}

        def _fake_shell_out(args: list[str], *, timeout: float) -> environments.CommandResult:
            captured["args"] = args
            return environments.CommandResult(stdout="", stderr="", exit_code=0)

        import pi_runtime.environments as env_module

        original = env_module._shell_out
        env_module._shell_out = _fake_shell_out  # type: ignore[assignment]
        try:
            backend.run("echo hi; rm -rf /")
        finally:
            env_module._shell_out = original

        # the dangerous-looking payload is exactly one argv element, not
        # split/re-interpreted by our own code
        assert captured["args"].count("echo hi; rm -rf /") == 1


# --- Security: prompt injection defense (context labeling) ------------------


class TestPromptInjectionDefense:
    def test_carried_forward_context_is_always_explicitly_labeled_as_background(self) -> None:
        """A decision/constraint re-surfaced by the Context Engine could
        otherwise be mistaken by the model for a new instruction — the
        note must always carry the defensive framing, not just the raw
        content."""
        engine = ContextEngine()
        goal = Goal(objective="x", constraints=["never delete without confirmation"])
        state = AgentState(goal=goal, decisions=["something the model decided earlier"])
        note = engine.render_working_set_note(state, [])
        assert note is not None
        assert "not a new instruction" in note
        # the label appears before the actual content, every time
        label_pos = note.index("not a new instruction")
        content_pos = note.index("never delete without confirmation")
        assert label_pos < content_pos


# --- Security: MCP permission tests -----------------------------------------


class TestMCPPermissionHardening:
    def test_denied_mcp_call_never_reaches_the_underlying_client(self) -> None:
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "danger",
            [MCPToolDescriptor(server_name="danger", name="delete_everything")],
            responses={"delete_everything": "done"},
        )
        policy = PolicyEngine(registry, mode=PolicyMode.STRICT)  # MEDIUM risk asked, no confirm -> denied
        adapter = MCPAdapter(registry, policy_engine=policy)
        adapter.register_server(client, server_name="danger")

        with pytest.raises(PolicyViolation):
            adapter.call("mcp:danger:delete_everything", {})
        assert client.calls == []

    def test_tool_name_with_extra_colons_does_not_crash_or_misroute(self) -> None:
        """MCP tool names may themselves contain colons — split(":", 2)
        must not misparse server_name in a way that reaches the wrong
        client."""
        registry = ToolRegistry()
        client = InMemoryMcpClient(
            "srv", [MCPToolDescriptor(server_name="srv", name="a:b:c")], responses={"a:b:c": "ok"}
        )
        adapter = MCPAdapter(registry)
        adapter.register_server(client, server_name="srv")
        result = adapter.call("mcp:srv:a:b:c", {})
        assert result == "ok"
        assert client.calls == [("a:b:c", {})]

    def test_call_to_a_server_never_registered_raises_unavailable_not_silently_noop(self) -> None:
        registry = ToolRegistry()
        from pi_runtime.tools import ToolSpec

        registry.register(ToolSpec(name="mcp:ghost:tool", risk=Risk.MEDIUM))
        adapter = MCPAdapter(registry)
        with pytest.raises(MCPUnavailable):
            adapter.call("mcp:ghost:tool", {})


# --- Reliability: malformed/partial data ------------------------------------


class TestMalformedDataHandling:
    def test_job_deserialization_survives_missing_optional_fields(self) -> None:
        """A job record written by an older/different code path might be
        missing fields added later — _job_from_dict must not crash on a
        minimal-but-valid record."""
        minimal = {
            "objective": "x",
            "job_id": "abc123",
            "status": "scheduled",
        }
        job = _job_from_dict(minimal)
        assert job.objective == "x"
        assert job.run_history == []
        assert job.attempt == 0

    def test_session_manager_skips_a_corrupted_jsonl_line_instead_of_crashing(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="x")
        manager.append_entry(
            info.id, SessionEntry(seq=0, parent_seq=None, kind="message", data={"role": "user", "content": "hi"})
        )

        # Corrupt the file by hand-appending an invalid JSON line.
        file_path = tmp_path / f"{info.id}.jsonl"
        with file_path.open("a", encoding="utf-8") as f:
            f.write("{not valid json\n")

        entries = manager.get_entries(info.id)
        assert len(entries) == 1  # the corrupted line was skipped, not raised


class TestSessionCorruptionRecovery:
    def test_replay_ignores_non_runtime_state_entries_mixed_into_the_session(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="mixed")
        store = RuntimeSessionStore(manager)

        manager.append_entry(
            info.id, SessionEntry(seq=0, parent_seq=None, kind="message", data={"role": "user", "content": "hi"})
        )
        store.save_state(info.id, AgentState(status=RunStatus.DONE), seq=1)

        history = store.replay(info.id)
        assert len(history) == 1  # the chat message entry didn't confuse replay


# --- Reliability: duplicate/repeated execution safety ------------------------


class TestDuplicateExecutionSafety:
    def test_ticking_twice_in_a_row_does_not_rerun_a_finished_job(self, tmp_path: Path) -> None:
        import asyncio

        from pi_ai.models import MutableModels
        from pi_ai.providers.faux import faux_assistant_message
        from pi_ai.providers.faux import faux_provider as _faux_provider
        from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions

        def _session() -> AgentSession:
            handle = _faux_provider()
            handle.set_responses([faux_assistant_message("done")])
            models = MutableModels()
            models.set_provider(handle.provider)
            model = handle.get_model()
            assert model is not None
            return AgentSession(AgentSessionOptions(models=models, model=model, cwd="/tmp", enable_subagents=False))

        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        scheduler.enqueue("say hi", Schedule(at=time.time() - 1))

        first_tick = asyncio.run(scheduler.tick(_session()))
        assert len(first_tick) == 1

        second_tick = asyncio.run(scheduler.tick(_session()))
        assert len(second_tick) == 0  # already DONE — not due again

    def test_policy_evaluate_called_twice_for_the_same_tool_logs_both(self) -> None:
        """Calling the same tool repeatedly must not corrupt or dedupe
        the audit trail — every real call is a real, separately
        auditable event."""
        policy = PolicyEngine(default_registry(), mode=PolicyMode.PERMISSIVE)
        policy.evaluate("read")
        policy.evaluate("read")
        assert len(policy.audit_log) == 2


# --- Reliability: skill rollback never leaves a broken/half-applied state ---


class TestSkillRegistryHardening:
    def test_rolling_back_an_unknown_skill_leaves_the_registry_unchanged(self) -> None:
        registry = SkillRegistry()
        assert registry.rollback("never-existed", to_version=1) is False
        assert registry.current("never-existed") is None
