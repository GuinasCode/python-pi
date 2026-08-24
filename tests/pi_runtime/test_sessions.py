"""Tests for pi_runtime.sessions.RuntimeSessionStore. Covers Fase 12's
acceptance criterion from plan.md section 16:

"Uma tarefa interrompida pode ser retomada sem reconstruir manualmente
o estado" — plus fork/branch/replay/lineage, which the criterion implies
a real implementation needs to support that resume.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_coding_agent.session_manager import SessionManager
from pi_runtime.sessions import RuntimeSessionStore, state_from_dict, state_to_dict
from pi_runtime.state import (
    AgentState,
    Budget,
    Goal,
    Plan,
    PlanStep,
    RunStatus,
    StepStatus,
    StopReason,
    VerificationResult,
)


def _sample_state() -> AgentState:
    goal = Goal(objective="research X", constraints=["cite sources"], budget=Budget(max_iterations=5))
    step = PlanStep(objective="research X", action="research X", status=StepStatus.DONE, attempts=1)
    plan = Plan(goal=goal, steps=[step], current_step_index=1)
    return AgentState(
        goal=goal,
        plan=plan,
        decisions=["used source A"],
        evidence=[{"url": "https://a.example"}],
        unresolved_questions=["is source B reliable?"],
        budget=Budget(max_iterations=5, consumed_iterations=1),
        iteration=1,
        status=RunStatus.DONE,
        stop_reason=StopReason.COMPLETED,
        verification=VerificationResult(passed=True, score=1.0),
        final_text="X is true because of source A.",
    )


class TestStateSerializationRoundTrip:
    def test_round_trips_through_json_exactly(self) -> None:
        state = _sample_state()
        raw = state_to_dict(state)
        # Must actually be JSON-serializable — SessionManager.append_entry
        # calls json.dumps on the entry data, so this is the real bar.
        reparsed = json.loads(json.dumps(raw))
        restored = state_from_dict(reparsed)

        assert restored.run_id == state.run_id
        assert restored.status == RunStatus.DONE
        assert restored.stop_reason == StopReason.COMPLETED
        assert restored.decisions == ["used source A"]
        assert restored.evidence == [{"url": "https://a.example"}]
        assert restored.final_text == state.final_text
        assert restored.goal is not None
        assert restored.goal.objective == "research X"
        assert restored.goal.constraints == ["cite sources"]
        assert restored.plan is not None
        assert len(restored.plan.steps) == 1
        assert restored.plan.steps[0].status == StepStatus.DONE
        assert restored.plan.current_step_index == 1
        assert restored.verification is not None
        assert restored.verification.passed

    def test_state_with_no_goal_or_plan_round_trips(self) -> None:
        state = AgentState(status=RunStatus.PENDING)
        raw = json.loads(json.dumps(state_to_dict(state)))
        restored = state_from_dict(raw)
        assert restored.goal is None
        assert restored.plan is None
        assert restored.stop_reason is None


class TestSaveAndLoadLatestState:
    def test_load_latest_state_returns_none_for_a_chat_only_session(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="chat")
        store = RuntimeSessionStore(manager)
        assert store.load_latest_state(info.id) is None

    def test_saved_state_is_loadable(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="run")
        store = RuntimeSessionStore(manager)

        state = _sample_state()
        store.save_state(info.id, state, seq=0)

        loaded = store.load_latest_state(info.id)
        assert loaded is not None
        assert loaded.run_id == state.run_id
        assert loaded.final_text == state.final_text

    def test_multiple_saves_return_only_the_latest(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="run")
        store = RuntimeSessionStore(manager)

        first = _sample_state()
        first.iteration = 1
        store.save_state(info.id, first, seq=0)

        second = _sample_state()
        second.iteration = 2
        second.status = RunStatus.RUNNING
        store.save_state(info.id, second, seq=1)

        loaded = store.load_latest_state(info.id)
        assert loaded is not None
        assert loaded.iteration == 2


class TestResume:
    def test_resume_returns_session_info_and_state_together(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="interrupted-run")
        store = RuntimeSessionStore(manager)
        store.save_state(info.id, _sample_state(), seq=0)

        resumed = store.resume(info.id)
        assert resumed is not None
        resumed_info, resumed_state = resumed
        assert resumed_info.id == info.id
        assert resumed_state.final_text == _sample_state().final_text

    def test_resume_of_a_nonexistent_session_returns_none(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = RuntimeSessionStore(manager)
        assert store.resume("does-not-exist") is None

    def test_resume_of_a_chat_only_session_returns_none(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="chat")
        store = RuntimeSessionStore(manager)
        assert store.resume(info.id) is None


class TestReplayAndLineage:
    def test_replay_returns_every_snapshot_in_order(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="run")
        store = RuntimeSessionStore(manager)

        for i in range(3):
            state = _sample_state()
            state.iteration = i
            store.save_state(info.id, state, seq=i)

        history = store.replay(info.id)
        assert [s.iteration for s in history] == [0, 1, 2]

    def test_replay_of_a_chat_only_session_is_empty(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="chat")
        store = RuntimeSessionStore(manager)
        assert store.replay(info.id) == []


class TestForkAndBranch:
    def test_fork_copies_runtime_state_entries_too(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="original")
        store = RuntimeSessionStore(manager)
        store.save_state(info.id, _sample_state(), seq=0)

        forked = store.fork(info.id, name="forked")
        assert forked is not None
        assert forked.id != info.id

        forked_state = store.load_latest_state(forked.id)
        assert forked_state is not None
        assert forked_state.final_text == _sample_state().final_text

    def test_branch_truncates_at_the_given_seq(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="original")
        store = RuntimeSessionStore(manager)

        for i in range(3):
            state = _sample_state()
            state.iteration = i
            store.save_state(info.id, state, seq=i)

        branched = store.branch(info.id, from_seq=1, name="branch-at-1")
        assert branched is not None

        history = store.replay(branched.id)
        assert [s.iteration for s in history] == [0, 1]  # seq 2 excluded

    def test_branch_of_a_nonexistent_session_returns_none(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = RuntimeSessionStore(manager)
        assert store.branch("nope", from_seq=0) is None

    def test_original_session_is_untouched_by_forking(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        info = manager.create_session(name="original")
        store = RuntimeSessionStore(manager)
        store.save_state(info.id, _sample_state(), seq=0)

        store.fork(info.id, name="forked")

        original_history = store.replay(info.id)
        assert len(original_history) == 1
