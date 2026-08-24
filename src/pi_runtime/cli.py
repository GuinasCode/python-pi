"""CLI / UX — Fase 17 of the research-first-runtime plan.

plan.md: "Somente depois do runtime estar estável" (Fases 1-16 all
committed and tested first) and "Não crie agent loops paralelos" — this
CLI is a thin entrypoint over pi_runtime's own real components
(AgentRuntime, Scheduler, RuntimeSessionStore, ResearchEngine), not a
new execution path. Model/provider setup reuses
pi_coding_agent.print_mode._setup_models (unchanged, already tested)
rather than a second provider-selection implementation.

Deliberately a separate entrypoint (`python -m pi_runtime.cli`, not
merged into pi_coding_agent's own CLI/interactive_mode/TUI) — those are
large, delicate, already-shipped surfaces; plan.md's own rule is to wire
the runtime in only once it's stable, and folding 17 phases of new
surface into that CLI in one pass would be exactly the kind of
large-blast-radius change Regra 1.6 ("não quebrar invariantes
existentes") warns against. This keeps every new command in its own
contained surface, fully real and fully tested, without touching the
shipped CLI at all.

Every command prints one JSON object to stdout — plan.md section 21:
"Todos os estados críticos devem ser representáveis em eventos JSON."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

from pi_coding_agent import Args
from pi_coding_agent.agent_session import AgentSession, AgentSessionOptions
from pi_coding_agent.config import get_session_dir
from pi_coding_agent.print_mode import _setup_models
from pi_coding_agent.session_manager import SessionManager
from pi_runtime.loop import AgentRuntime
from pi_runtime.research import ResearchEngine, ResearchTask
from pi_runtime.scheduler import Job, JobStore, Schedule, Scheduler
from pi_runtime.sessions import RuntimeSessionStore
from pi_runtime.state import Goal


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=str))


def _build_session() -> AgentSession:
    models, model = _setup_models(Args())
    return AgentSession(AgentSessionOptions(models=models, model=model, cwd=".", enable_subagents=False))


def _job_to_json(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "objective": job.objective,
        "status": job.status.value,
        "attempt": job.attempt,
        "next_run_at": job.next_run_at,
        "run_count": len(job.run_history),
    }


def cmd_run(args: argparse.Namespace) -> int:
    session = _build_session()
    state = asyncio.run(AgentRuntime().run(Goal(objective=args.objective), session))
    _print_json(
        {
            "command": "run",
            "run_id": state.run_id,
            "status": state.status.value,
            "stop_reason": state.stop_reason.value if state.stop_reason else None,
            "final_text": state.final_text,
            "error_message": state.error_message,
            "iterations": state.iteration,
        }
    )
    return 0 if state.error_message is None else 1


def cmd_research(args: argparse.Namespace) -> int:
    engine = ResearchEngine()
    result = engine.research(ResearchTask(question=args.question, urls=list(args.urls)))
    _print_json(
        {
            "command": "research",
            "question": result.question,
            "coverage_note": result.coverage_note,
            "evidence_count": len(result.evidence),
            "failed_urls": result.failed_urls,
            "sources": [e.url for e in result.evidence],
        }
    )
    return 0


def _job_store() -> JobStore:
    return JobStore(SessionManager(get_session_dir()))


def cmd_jobs_enqueue(args: argparse.Namespace) -> int:
    store = _job_store()
    scheduler = Scheduler(store)
    schedule = Schedule(interval_seconds=args.interval_seconds) if args.interval_seconds else Schedule(at=time.time())
    job = scheduler.enqueue(args.objective, schedule, max_retries=args.max_retries)
    _print_json({"command": "jobs.enqueue", "job": _job_to_json(job)})
    return 0


def cmd_jobs_list(_args: argparse.Namespace) -> int:
    store = _job_store()
    _print_json({"command": "jobs.list", "jobs": [_job_to_json(j) for j in store.all_jobs()]})
    return 0


def cmd_jobs_cancel(args: argparse.Namespace) -> int:
    store = _job_store()
    scheduler = Scheduler(store)
    cancelled = scheduler.cancel(args.job_id)
    _print_json({"command": "jobs.cancel", "job_id": args.job_id, "cancelled": cancelled})
    return 0 if cancelled else 1


def cmd_jobs_tick(_args: argparse.Namespace) -> int:
    store = _job_store()
    scheduler = Scheduler(store)
    session = _build_session()
    results = asyncio.run(scheduler.tick(session))
    _print_json({"command": "jobs.tick", "ran": [_job_to_json(j) for j in results]})
    return 0


def cmd_sessions_resume(args: argparse.Namespace) -> int:
    manager = SessionManager(get_session_dir())
    store = RuntimeSessionStore(manager)
    resumed = store.resume(args.session_id)
    if resumed is None:
        _print_json({"command": "sessions.resume", "session_id": args.session_id, "found": False})
        return 1
    info, state = resumed
    _print_json(
        {
            "command": "sessions.resume",
            "session_id": info.id,
            "found": True,
            "status": state.status.value,
            "stop_reason": state.stop_reason.value if state.stop_reason else None,
            "final_text": state.final_text,
        }
    )
    return 0


def cmd_sessions_replay(args: argparse.Namespace) -> int:
    manager = SessionManager(get_session_dir())
    store = RuntimeSessionStore(manager)
    history = store.replay(args.session_id)
    _print_json(
        {
            "command": "sessions.replay",
            "session_id": args.session_id,
            "snapshot_count": len(history),
            "snapshots": [{"run_id": s.run_id, "status": s.status.value, "iteration": s.iteration} for s in history],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi-runtime", description="research-first-runtime CLI (Fase 17)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a goal through AgentRuntime")
    run_parser.add_argument("objective")
    run_parser.set_defaults(func=cmd_run)

    research_parser = subparsers.add_parser("research", help="Gather evidence for a question from a list of URLs")
    research_parser.add_argument("question")
    research_parser.add_argument("urls", nargs="*")
    research_parser.set_defaults(func=cmd_research)

    jobs_parser = subparsers.add_parser("jobs", help="Scheduler job management")
    jobs_sub = jobs_parser.add_subparsers(dest="jobs_command", required=True)

    enqueue_parser = jobs_sub.add_parser("enqueue")
    enqueue_parser.add_argument("objective")
    enqueue_parser.add_argument("--interval-seconds", type=float, default=None)
    enqueue_parser.add_argument("--max-retries", type=int, default=0)
    enqueue_parser.set_defaults(func=cmd_jobs_enqueue)

    list_parser = jobs_sub.add_parser("list")
    list_parser.set_defaults(func=cmd_jobs_list)

    cancel_parser = jobs_sub.add_parser("cancel")
    cancel_parser.add_argument("job_id")
    cancel_parser.set_defaults(func=cmd_jobs_cancel)

    tick_parser = jobs_sub.add_parser("tick")
    tick_parser.set_defaults(func=cmd_jobs_tick)

    sessions_parser = subparsers.add_parser("sessions", help="Runtime session inspection")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command", required=True)

    resume_parser = sessions_sub.add_parser("resume")
    resume_parser.add_argument("session_id")
    resume_parser.set_defaults(func=cmd_sessions_resume)

    replay_parser = sessions_sub.add_parser("replay")
    replay_parser.add_argument("session_id")
    replay_parser.set_defaults(func=cmd_sessions_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
