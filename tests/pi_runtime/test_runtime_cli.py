"""Tests for pi_runtime.cli. Covers Fase 17's acceptance criteria from
plan.md section 21: real commands over research/sessions/jobs, and every
command's output is representable as JSON (literally — every command
here is asserted to print valid, parseable JSON).

NVAPI_KEY/OPENAI_API_KEY are stripped so _setup_models (reused,
unchanged, from pi_coding_agent.print_mode) falls back to the
deterministic faux provider — no real network call anywhere in these
tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pi_runtime.cli import main


@pytest.fixture(autouse=True)
def _strip_api_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NVAPI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Route session storage (jobs/sessions commands both go through
    # get_session_dir()) into an isolated tmp dir, not the real ~/.pi.
    monkeypatch.setenv("PI_SESSION_DIR", str(tmp_path / "sessions"))


def _run_cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict[str, object]]:
    exit_code = main(argv)
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


class TestRunCommand:
    def test_run_prints_valid_json_with_a_stop_reason(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, payload = _run_cli(capsys, ["run", "say hi"])
        assert exit_code == 0
        assert payload["command"] == "run"
        assert payload["stop_reason"] is not None
        assert payload["run_id"]


class TestResearchCommand:
    def test_research_with_no_urls_reports_insufficient_evidence(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, payload = _run_cli(capsys, ["research", "what is X?"])
        assert exit_code == 0
        assert payload["evidence_count"] == 0
        assert "insufficient" in payload["coverage_note"]

    def test_research_with_a_real_fetched_url(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = httpx.Response(200, content=b"real content", headers={"content-type": "text/plain"})
        monkeypatch.setattr(httpx, "get", lambda *_a, **_k: response)

        exit_code, payload = _run_cli(capsys, ["research", "what is X?", "https://example.com"])
        assert exit_code == 0
        assert payload["evidence_count"] == 1
        assert payload["sources"] == ["https://example.com"]


class TestJobsCommands:
    def test_enqueue_then_list_shows_the_job(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, enqueue_payload = _run_cli(capsys, ["jobs", "enqueue", "do the thing"])
        assert exit_code == 0
        job_id = enqueue_payload["job"]["job_id"]

        _, list_payload = _run_cli(capsys, ["jobs", "list"])
        job_ids = [j["job_id"] for j in list_payload["jobs"]]
        assert job_id in job_ids

    def test_cancel_a_scheduled_job(self, capsys: pytest.CaptureFixture[str]) -> None:
        _, enqueue_payload = _run_cli(capsys, ["jobs", "enqueue", "do the thing"])
        job_id = enqueue_payload["job"]["job_id"]

        exit_code, cancel_payload = _run_cli(capsys, ["jobs", "cancel", job_id])
        assert exit_code == 0
        assert cancel_payload["cancelled"] is True

    def test_cancel_unknown_job_fails_explicitly(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, payload = _run_cli(capsys, ["jobs", "cancel", "nonexistent"])
        assert exit_code == 1
        assert payload["cancelled"] is False

    def test_tick_runs_a_due_job(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_cli(capsys, ["jobs", "enqueue", "say hi"])
        exit_code, payload = _run_cli(capsys, ["jobs", "tick"])
        assert exit_code == 0
        assert len(payload["ran"]) == 1


class TestSessionsCommands:
    def test_resume_of_unknown_session_reports_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, payload = _run_cli(capsys, ["sessions", "resume", "nonexistent"])
        assert exit_code == 1
        assert payload["found"] is False

    def test_resume_after_a_run_was_saved(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from pi_coding_agent.session_manager import SessionManager
        from pi_runtime.sessions import RuntimeSessionStore
        from pi_runtime.state import AgentState, RunStatus, StopReason

        session_dir = tmp_path / "sessions2"
        monkeypatch.setenv("PI_SESSION_DIR", str(session_dir))

        manager = SessionManager(session_dir)
        store = RuntimeSessionStore(manager)
        info = manager.create_session(name="test-run")
        state = AgentState(status=RunStatus.DONE, stop_reason=StopReason.COMPLETED, final_text="done")
        store.save_state(info.id, state, seq=0)

        exit_code, payload = _run_cli(capsys, ["sessions", "resume", info.id])

        assert exit_code == 0
        assert payload["found"] is True
        assert payload["final_text"] == "done"

    def test_replay_of_unknown_session_is_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code, payload = _run_cli(capsys, ["sessions", "replay", "nonexistent"])
        assert exit_code == 0
        assert payload["snapshot_count"] == 0


class TestJsonEveryCommand:
    """plan.md section 21: every critical state must be representable as
    a JSON event — this is proven mechanically for every command above
    via the shared _run_cli() helper (json.loads never raises), plus
    this one negative check that stdout is never anything else."""

    def test_run_output_has_no_extra_non_json_lines(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["run", "hi"])
        out = capsys.readouterr().out.strip()
        assert out.count("\n") == 0  # exactly one line, the JSON payload
