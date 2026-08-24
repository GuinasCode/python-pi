"""Tests for pi_runtime.scheduler. Covers Fase 14's acceptance criteria
from plan.md section 18:

- one-shot
- recurring
- retry
- persisted jobs
- cancellation

Also confirms the explicit rule: "Scheduler executa o mesmo runtime
normal. Não criar um 'segundo agent'" — run_job() actually drives a real
AgentRuntime against a real AgentSession (faux provider, no network).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from pi_ai.models import MutableModels
from pi_ai.providers.faux import faux_assistant_message
from pi_ai.providers.faux import faux_provider as _faux_provider
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_coding_agent.session_manager import SessionManager
from pi_runtime.loop import AgentRuntime
from pi_runtime.scheduler import JobStatus, JobStore, Schedule, Scheduler
from pi_runtime.state import RunStatus


def _make_session(responses: list[Any]) -> AgentSession:
    handle = _faux_provider()
    handle.set_responses(responses)
    models = MutableModels()
    models.set_provider(handle.provider)
    model = handle.get_model()
    assert model is not None
    return AgentSession(AgentSessionOptions(models=models, model=model, cwd="/tmp", enable_subagents=False))


class TestScheduleValidation:
    def test_requires_at_or_interval(self) -> None:
        raised = False
        try:
            Schedule()
        except ValueError:
            raised = True
        assert raised

    def test_one_shot_is_not_recurring(self) -> None:
        assert not Schedule(at=time.time()).recurring

    def test_interval_is_recurring(self) -> None:
        assert Schedule(interval_seconds=60).recurring


class TestJobPersistence:
    """Fase 14 acceptance criterion: persisted jobs — a fresh JobStore
    instance over the same SessionManager sees jobs saved by another
    instance."""

    def test_enqueued_job_survives_a_fresh_store_instance(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store1 = JobStore(manager)
        scheduler = Scheduler(store1)
        job = scheduler.enqueue("do the thing", Schedule(at=time.time()))

        store2 = JobStore(manager)
        reloaded = store2.get(job.job_id)
        assert reloaded is not None
        assert reloaded.objective == "do the thing"

    def test_all_jobs_returns_only_the_latest_version_per_job(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        job = Scheduler(store).enqueue("x", Schedule(at=time.time()))

        job.attempt = 1
        store.save(job)
        job.attempt = 2
        store.save(job)

        all_jobs = store.all_jobs()
        assert len([j for j in all_jobs if j.job_id == job.job_id]) == 1
        assert store.get(job.job_id).attempt == 2  # type: ignore[union-attr]


class TestDueJobs:
    def test_future_job_is_not_due(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        scheduler.enqueue("x", Schedule(at=time.time() + 3600))
        assert scheduler.due_jobs() == []

    def test_past_job_is_due(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("x", Schedule(at=time.time() - 10))
        due = scheduler.due_jobs()
        assert [j.job_id for j in due] == [job.job_id]

    def test_cancelled_job_is_never_due(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("x", Schedule(at=time.time() - 10))
        scheduler.cancel(job.job_id)
        assert scheduler.due_jobs() == []


class TestCancellation:
    def test_scheduled_job_can_be_cancelled(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("x", Schedule(at=time.time()))

        assert scheduler.cancel(job.job_id) is True
        assert store.get(job.job_id).status == JobStatus.CANCELLED  # type: ignore[union-attr]

    def test_already_cancelled_job_cannot_be_cancelled_again(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("x", Schedule(at=time.time()))
        scheduler.cancel(job.job_id)
        assert scheduler.cancel(job.job_id) is False

    def test_unknown_job_cannot_be_cancelled(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        assert scheduler.cancel("nope") is False


class TestRunJobUsesTheRealRuntime:
    """Confirms plan.md's rule: "Scheduler executa o mesmo runtime
    normal" — run_job drives an actual AgentRuntime.run() call."""

    def test_successful_one_shot_job_completes(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store, runtime=AgentRuntime())
        job = scheduler.enqueue("say hi", Schedule(at=time.time()))
        session = _make_session([faux_assistant_message("hi there")])

        result = asyncio.run(scheduler.run_job(job, session))

        assert result.status == JobStatus.DONE
        assert len(result.run_history) == 1
        assert result.run_history[0].status == RunStatus.DONE


class TestRecurringJobs:
    def test_successful_recurring_job_reschedules_instead_of_finishing(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("check status", Schedule(interval_seconds=60))
        session = _make_session([faux_assistant_message("all good")])

        result = asyncio.run(scheduler.run_job(job, session))

        assert result.status == JobStatus.SCHEDULED  # not DONE — reschedules
        assert result.next_run_at > time.time()

    def test_recurring_job_resets_attempts_on_success(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("check status", Schedule(interval_seconds=60))
        job.attempt = 3
        session = _make_session([faux_assistant_message("all good")])

        result = asyncio.run(scheduler.run_job(job, session))
        assert result.attempt == 0


class TestRetry:
    def test_failed_job_within_retry_budget_is_rescheduled(self, tmp_path: Path) -> None:
        from pi_ai import StopReason as AiStopReason

        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("do the thing", Schedule(at=time.time()), max_retries=2)
        session = _make_session(
            [faux_assistant_message("boom", stop_reason=AiStopReason.ERROR, error_message="upstream failure")]
        )

        result = asyncio.run(scheduler.run_job(job, session))

        assert result.status == JobStatus.SCHEDULED  # retried, not failed
        assert result.attempt == 1

    def test_failed_job_beyond_retry_budget_is_marked_failed(self, tmp_path: Path) -> None:
        from pi_ai import StopReason as AiStopReason

        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        job = scheduler.enqueue("do the thing", Schedule(at=time.time()), max_retries=0)
        # AgentRuntime's own Replanner (Fase 1) already retries a failed
        # step internally up to max_attempts_per_step=2 before run()
        # returns FAILED — both responses need to fail for the job-level
        # retry check (this test) to actually see a FAILED run.
        session = _make_session(
            [
                faux_assistant_message("boom", stop_reason=AiStopReason.ERROR, error_message="upstream failure"),
                faux_assistant_message("boom again", stop_reason=AiStopReason.ERROR, error_message="upstream failure"),
            ]
        )

        result = asyncio.run(scheduler.run_job(job, session))

        assert result.status == JobStatus.FAILED
        assert result.run_history[-1].error_message == "upstream failure"


class TestTick:
    def test_tick_runs_every_due_job_and_returns_results(self, tmp_path: Path) -> None:
        manager = SessionManager(tmp_path)
        store = JobStore(manager)
        scheduler = Scheduler(store)
        scheduler.enqueue("a", Schedule(at=time.time() - 10))
        scheduler.enqueue("b", Schedule(at=time.time() + 3600))  # not due
        session = _make_session([faux_assistant_message("done a")])

        results = asyncio.run(scheduler.tick(session))

        assert len(results) == 1
        assert results[0].objective == "a"
        assert results[0].status == JobStatus.DONE
