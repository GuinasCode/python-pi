"""Sessions + Lineage + Replay — Fase 12 of the research-first-runtime
plan.

Does not replace pi_coding_agent.session_manager.SessionManager — it's
already a real, tested JSONL session store with create/open/list/fork
(SessionManager.fork_session already exists and is reused unchanged
here, not reimplemented). What's missing: AgentState (Fase 1) has never
been persisted into a session at all — only chat messages are. This
module adds that: serializing/deserializing AgentState to/from a
SessionEntry (kind="runtime_state"), so an interrupted run's *state*
(not just its conversation) can be resumed without rebuilding it by
hand, and a full run's sequence of runtime-state snapshots can be
replayed for inspection.

AgentState.working_memory intentionally has no cross-session
persistence path of its own here — same reasoning as Fase 7's
CognitiveMemoryType.WORKING: working memory is meant to not survive
past its own run. It's still serialized as part of the resumed run's
state (a resumed run picks up mid-flight, so its own working memory is
part of what's being resumed) — this only means there's no separate
"restore working memory into a *new* run" path, which would defeat the
point of the type.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pi_coding_agent.session_manager import SessionEntry, SessionInfo, SessionManager
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

_RUNTIME_STATE_KIND = "runtime_state"


def state_to_dict(state: AgentState) -> dict[str, Any]:
    """AgentState's own fields are all plain/serializable (Fase 1's own
    design goal) — dataclasses.asdict() handles the nesting (Goal,
    Plan/PlanStep, Budget, VerificationResult) automatically; enums
    (RunStatus, StopReason, StepStatus) need their .value pulled out
    explicitly since asdict() leaves them as enum instances, which
    json.dumps (SessionManager.append_entry's own serializer) can't
    encode on its own."""
    raw = asdict(state)
    raw["status"] = state.status.value
    raw["stop_reason"] = state.stop_reason.value if state.stop_reason is not None else None
    if raw.get("plan") is not None:
        for step in raw["plan"]["steps"]:
            step["status"] = step["status"].value if isinstance(step["status"], StepStatus) else step["status"]
    return raw


def state_from_dict(raw: dict[str, Any]) -> AgentState:
    """Inverse of state_to_dict — rebuilds real dataclass/enum instances,
    not a plain dict standing in for AgentState."""
    goal_raw = raw.get("goal")
    goal = None
    if goal_raw is not None:
        budget_raw = goal_raw.get("budget") or {}
        goal = Goal(
            objective=goal_raw["objective"],
            context=goal_raw.get("context", ""),
            success_criteria=list(goal_raw.get("success_criteria", [])),
            constraints=list(goal_raw.get("constraints", [])),
            priority=goal_raw.get("priority", 0),
            deadline_ts=goal_raw.get("deadline_ts"),
            budget=Budget(**budget_raw) if budget_raw else Budget(),
        )

    plan_raw = raw.get("plan")
    plan = None
    if plan_raw is not None:
        steps = [
            PlanStep(
                id=s["id"],
                objective=s["objective"],
                preconditions=list(s.get("preconditions", [])),
                action=s["action"],
                dependencies=list(s.get("dependencies", [])),
                expected_outcome=s.get("expected_outcome", ""),
                verification=s.get("verification", ""),
                status=StepStatus(s["status"]),
                attempts=s.get("attempts", 0),
                owner=s.get("owner", "principal"),
            )
            for s in plan_raw["steps"]
        ]
        plan = Plan(
            goal=goal or Goal(objective=""), steps=steps, current_step_index=plan_raw.get("current_step_index", 0)
        )

    verification_raw = raw.get("verification")
    verification = VerificationResult(**verification_raw) if verification_raw is not None else None

    budget_raw = raw.get("budget") or {}

    return AgentState(
        run_id=raw["run_id"],
        goal=goal,
        plan=plan,
        working_memory=list(raw.get("working_memory", [])),
        active_tasks=list(raw.get("active_tasks", [])),
        child_agent_handles=list(raw.get("child_agent_handles", [])),
        evidence=list(raw.get("evidence", [])),
        unresolved_questions=list(raw.get("unresolved_questions", [])),
        decisions=list(raw.get("decisions", [])),
        pending_actions=list(raw.get("pending_actions", [])),
        budget=Budget(**budget_raw) if budget_raw else Budget(),
        iteration=raw.get("iteration", 0),
        status=RunStatus(raw["status"]),
        stop_reason=StopReason(raw["stop_reason"]) if raw.get("stop_reason") else None,
        verification=verification,
        final_text=raw.get("final_text"),
        error_message=raw.get("error_message"),
        started_at=raw.get("started_at", 0.0),
        finished_at=raw.get("finished_at"),
    )


class RuntimeSessionStore:
    """Sits on top of an existing SessionManager (unchanged) rather than
    a new persistence layer — every method here either delegates
    straight to SessionManager or reads/writes SessionEntry objects the
    same way pi_coding_agent.interactive_mode._persist_message already
    does for chat messages."""

    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager

    def save_state(self, session_id: str, state: AgentState, *, seq: int) -> None:
        entry = SessionEntry(
            seq=seq, parent_seq=seq - 1 if seq > 0 else None, kind=_RUNTIME_STATE_KIND, data=state_to_dict(state)
        )
        self._manager.append_entry(session_id, entry)

    def load_latest_state(self, session_id: str) -> AgentState | None:
        """Fase 12's central acceptance criterion: "uma tarefa
        interrompida pode ser retomada sem reconstruir manualmente o
        estado." Returns the most recently saved runtime_state entry,
        fully reconstructed — or None if this session never ran through
        AgentRuntime at all (a plain chat-only session, which is still a
        completely valid, unaffected use of SessionManager)."""
        entries = [e for e in self._manager.get_entries(session_id) if e.kind == _RUNTIME_STATE_KIND]
        if not entries:
            return None
        latest = max(entries, key=lambda e: e.seq)
        return state_from_dict(latest.data)

    def replay(self, session_id: str) -> list[AgentState]:
        """Fase 12: a lineage of every runtime-state snapshot ever saved
        for this session, in order — for inspection, not just the
        final/latest one."""
        entries = sorted(
            (e for e in self._manager.get_entries(session_id) if e.kind == _RUNTIME_STATE_KIND), key=lambda e: e.seq
        )
        return [state_from_dict(e.data) for e in entries]

    def resume(self, session_id: str) -> tuple[SessionInfo, AgentState] | None:
        """Open a session and its most recent runtime state together —
        the single call a caller needs to actually continue an
        interrupted run."""
        info = self._manager.open_session(session_id)
        if info is None:
            return None
        state = self.load_latest_state(session_id)
        if state is None:
            return None
        return info, state

    def fork(self, session_id: str, *, name: str | None = None) -> SessionInfo | None:
        """Delegates straight to SessionManager.fork_session (unchanged,
        already copies every entry — chat messages *and*
        runtime_state entries alike, since it copies by kind-agnostic
        entry list) — a forked session is resumable via this same
        store immediately, with its own independent lineage from that
        point forward."""
        return self._manager.fork_session(session_id, name=name)

    def branch(self, session_id: str, *, from_seq: int, name: str | None = None) -> SessionInfo | None:
        """A fork truncated at a specific point in history — for
        replaying "what if we'd stopped/decided differently at step N"
        without carrying forward entries that happened after it."""
        original = self._manager.open_session(session_id)
        if original is None:
            return None
        new_info = self._manager.create_session(cwd=original.cwd, name=name or f"branch-{session_id}-{from_seq}")
        for entry in self._manager.get_entries(session_id):
            if entry.seq <= from_seq:
                self._manager.append_entry(new_info.id, entry)
        return new_info


__all__ = ["RuntimeSessionStore", "state_from_dict", "state_to_dict"]
