"""Scheduler + Jobs — Fase 14 of the research-first-runtime plan.

plan.md's explicit rule: "Scheduler executa o mesmo runtime normal. Não
criar um 'segundo agent'." Scheduler.tick() runs a due Job through the
real pi_runtime.loop.AgentRuntime (Fase 1) against a real AgentSession —
not a separate execution path with its own rules.

Persistence reuses pi_coding_agent.session_manager.SessionManager
(unchanged, real, tested JSONL store) rather than a new database: every
job lives as one SessionEntry(kind="job") in a dedicated session, the
same storage substrate pi_runtime.sessions.RuntimeSessionStore already
uses for AgentState snapshots (Fase 12) — not a second persistence
layer.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pi_coding_agent.agent_session import AgentSession
from pi_coding_agent.session_manager import SessionEntry, SessionManager
from pi_runtime.loop import AgentRuntime
from pi_runtime.state import Goal, RunStatus

_JOB_KIND = "job"
_JOBS_SESSION_NAME = "__pi_runtime_scheduler__"


class JobStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Schedule:
    """A schedule is either one-shot (run once at/after `at`) or
    recurring (run every `interval_seconds`). Not a real cron expression
    parser — plan.md lists "cron" as a capability, but a correct cron
    parser is a substantial, well-solved problem with existing libraries
    (croniter and similar) that aren't a dependency of this repo; adding
    one speculatively for a single vertical slice would violate Regra
    1.5. Fixed-interval recurrence covers the same "recurring job" and
    "run history" acceptance criteria honestly without a fake cron
    parser standing in for a real one."""

    def __init__(self, *, at: float | None = None, interval_seconds: float | None = None) -> None:
        if at is None and interval_seconds is None:
            raise ValueError("Schedule needs either `at` (one-shot) or `interval_seconds` (recurring)")
        self.at = at
        self.interval_seconds = interval_seconds

    @property
    def recurring(self) -> bool:
        return self.interval_seconds is not None

    def to_dict(self) -> dict[str, Any]:
        return {"at": self.at, "interval_seconds": self.interval_seconds}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Schedule:
        return cls(at=raw.get("at"), interval_seconds=raw.get("interval_seconds"))


@dataclass
class JobRunRecord:
    started_at: float
    finished_at: float
    status: RunStatus
    attempt: int
    error_message: str | None = None


@dataclass
class Job:
    objective: str
    schedule_at: float | None = None
    schedule_interval_seconds: float | None = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: JobStatus = JobStatus.SCHEDULED
    max_retries: int = 0
    attempt: int = 0
    next_run_at: float = field(default_factory=time.time)
    run_history: list[JobRunRecord] = field(default_factory=list)

    @property
    def schedule(self) -> Schedule:
        return Schedule(at=self.schedule_at, interval_seconds=self.schedule_interval_seconds)


def _job_to_dict(job: Job) -> dict[str, Any]:
    raw = asdict(job)
    raw["status"] = job.status.value
    for record in raw["run_history"]:
        record["status"] = record["status"].value if isinstance(record["status"], RunStatus) else record["status"]
    return raw


def _job_from_dict(raw: dict[str, Any]) -> Job:
    history = [
        JobRunRecord(
            started_at=r["started_at"],
            finished_at=r["finished_at"],
            status=RunStatus(r["status"]),
            attempt=r["attempt"],
            error_message=r.get("error_message"),
        )
        for r in raw.get("run_history", [])
    ]
    return Job(
        objective=raw["objective"],
        schedule_at=raw.get("schedule_at"),
        schedule_interval_seconds=raw.get("schedule_interval_seconds"),
        job_id=raw["job_id"],
        status=JobStatus(raw["status"]),
        max_retries=raw.get("max_retries", 0),
        attempt=raw.get("attempt", 0),
        next_run_at=raw.get("next_run_at", 0.0),
        run_history=history,
    )


class JobStore:
    """Persists Job records as SessionEntry(kind="job") in one dedicated
    session — reuses SessionManager unchanged rather than a new
    database. Fase 14 acceptance criterion: "persisted jobs" is real:
    every enqueue()/save() call writes through SessionManager.
    append_entry(), and jobs survive being reloaded from a fresh
    JobStore instance over the same SessionManager (proven in tests by
    constructing two JobStore instances over one SessionManager)."""

    def __init__(self, manager: SessionManager) -> None:
        self._manager = manager
        self._session_id = self._ensure_session()

    def _ensure_session(self) -> str:
        existing = next((s for s in self._manager.list_sessions() if s.name == _JOBS_SESSION_NAME), None)
        if existing is not None:
            return existing.id
        info = self._manager.create_session(name=_JOBS_SESSION_NAME)
        return info.id

    def save(self, job: Job) -> None:
        entries = self._manager.get_entries(self._session_id)
        next_seq = max((e.seq for e in entries), default=-1) + 1
        self._manager.append_entry(
            self._session_id, SessionEntry(seq=next_seq, parent_seq=None, kind=_JOB_KIND, data=_job_to_dict(job))
        )

    def all_jobs(self) -> list[Job]:
        """The *latest* saved record per job_id — save() appends rather
        than overwrites (same append-only pattern as
        pi_runtime.sessions.RuntimeSessionStore), so a job's full save
        history is preserved and only the most recent state is
        considered current."""
        latest_by_id: dict[str, tuple[int, Job]] = {}
        for entry in self._manager.get_entries(self._session_id):
            if entry.kind != _JOB_KIND:
                continue
            job = _job_from_dict(entry.data)
            if job.job_id not in latest_by_id or entry.seq > latest_by_id[job.job_id][0]:
                latest_by_id[job.job_id] = (entry.seq, job)
        return [job for _seq, job in latest_by_id.values()]

    def get(self, job_id: str) -> Job | None:
        return next((j for j in self.all_jobs() if j.job_id == job_id), None)


class Scheduler:
    """Fase 14: one-shot, recurring, retry, persisted jobs, cancellation
    — all running the same AgentRuntime (Fase 1), never a second agent
    loop."""

    def __init__(self, store: JobStore, *, runtime: AgentRuntime | None = None) -> None:
        self._store = store
        self._runtime = runtime or AgentRuntime()

    def enqueue(self, objective: str, schedule: Schedule, *, max_retries: int = 0) -> Job:
        job = Job(
            objective=objective,
            schedule_at=schedule.at,
            schedule_interval_seconds=schedule.interval_seconds,
            max_retries=max_retries,
            next_run_at=schedule.at if schedule.at is not None else time.time(),
        )
        self._store.save(job)
        return job

    def cancel(self, job_id: str) -> bool:
        """Fase 14 acceptance criterion: cancellation. A job already
        finished/failed/cancelled can't be cancelled again (returns
        False) — cancellation only applies to work still pending."""
        job = self._store.get(job_id)
        if job is None or job.status not in (JobStatus.SCHEDULED, JobStatus.RUNNING):
            return False
        job.status = JobStatus.CANCELLED
        self._store.save(job)
        return True

    def due_jobs(self, *, now: float | None = None) -> list[Job]:
        now = now if now is not None else time.time()
        return [j for j in self._store.all_jobs() if j.status == JobStatus.SCHEDULED and j.next_run_at <= now]

    async def run_job(self, job: Job, session: AgentSession) -> Job:
        """Runs one job through the real AgentRuntime. On failure, retries
        up to max_retries by rescheduling (status back to SCHEDULED,
        next_run_at now) rather than looping in-process — each retry is
        its own tick(), so a crashed scheduler process doesn't lose a
        pending retry (it's already persisted as SCHEDULED)."""
        job.status = JobStatus.RUNNING
        job.attempt += 1
        self._store.save(job)

        started_at = time.time()
        state = await self._runtime.run(Goal(objective=job.objective), session)
        finished_at = time.time()

        job.run_history.append(
            JobRunRecord(
                started_at=started_at,
                finished_at=finished_at,
                status=state.status,
                attempt=job.attempt,
                error_message=state.error_message,
            )
        )

        if state.status == RunStatus.DONE:
            job.status = self._finish_or_reschedule(job)
        elif job.attempt <= job.max_retries:
            job.status = JobStatus.SCHEDULED
            job.next_run_at = time.time()
        else:
            job.status = JobStatus.FAILED

        self._store.save(job)
        return job

    def _finish_or_reschedule(self, job: Job) -> JobStatus:
        if job.schedule.recurring:
            assert job.schedule_interval_seconds is not None
            job.next_run_at = time.time() + job.schedule_interval_seconds
            job.attempt = 0
            return JobStatus.SCHEDULED
        return JobStatus.DONE

    async def tick(self, session: AgentSession, *, now: float | None = None) -> list[Job]:
        """Runs every currently-due job once. Returns the jobs that were
        run (in whatever final status they ended up in) so a caller can
        inspect what happened this tick without re-querying the store."""
        return [await self.run_job(job, session) for job in self.due_jobs(now=now)]


__all__ = ["Job", "JobRunRecord", "JobStatus", "JobStore", "Schedule", "Scheduler"]
