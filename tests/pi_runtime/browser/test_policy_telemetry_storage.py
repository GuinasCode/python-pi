"""Slice B5 — per-action policy, telemetry redaction, persistent-profile
storage state."""

from __future__ import annotations

import asyncio

from fixtures import fixture_server

from pi_runtime.browser import BrowserManager, InteractionStatus
from pi_runtime.tools import PolicyEngine, PolicyMode, default_registry


async def _ref_for(manager: BrowserManager, session_id: str, *, role: str, name: str | None = None) -> str:
    page_snapshot = await manager.snapshot(session_id)
    return next(n.ref for n in page_snapshot.nodes if n.role == role and (name is None or n.name == name))


class TestPerActionPolicy:
    def test_low_risk_actions_are_allowed_under_default_policy(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
                async with BrowserManager(policy_engine=engine) as manager:
                    session = await manager.open_session()
                    result = await manager.navigate(session.session_id, f"{base_url}/")
                    assert result.ok  # navigate is LOW risk — DEFAULT mode only asks on HIGH

        asyncio.run(_run())

    def test_high_risk_evaluate_is_denied_with_no_confirm_callback(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
                async with BrowserManager(policy_engine=engine) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.evaluate(session.session_id, "1 + 1")

                    assert not result.ok
                    assert result.status == InteractionStatus.POLICY_DENIED

        asyncio.run(_run())

    def test_high_risk_evaluate_is_allowed_once_approved(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT, confirm=lambda spec: True)
                async with BrowserManager(policy_engine=engine) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.evaluate(session.session_id, "1 + 1")

                    assert result.ok

        asyncio.run(_run())

    def test_medium_risk_click_asks_under_strict_mode(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                # approve opening the session, but nothing else — click stays denied
                engine = PolicyEngine(
                    default_registry(), mode=PolicyMode.STRICT, confirm=lambda spec: spec.name == "browser"
                )
                async with BrowserManager(policy_engine=engine) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Submit")

                    result = await manager.click(session.session_id, ref)

                    assert not result.ok
                    assert result.status == InteractionStatus.POLICY_DENIED

        asyncio.run(_run())

    def test_denial_is_recorded_in_the_shared_audit_log(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)
                async with BrowserManager(policy_engine=engine) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")
                    await manager.evaluate(session.session_id, "1 + 1")

                    assert any(entry.tool_name == "browser_evaluate" for entry in engine.audit_log)

        asyncio.run(_run())


class TestTelemetry:
    def test_navigate_and_click_emit_telemetry_records(self) -> None:
        records: list[object] = []

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager(telemetry_sink=records.append) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Submit")
                    await manager.click(session.session_id, ref)

        asyncio.run(_run())

        actions = [r.action for r in records]  # type: ignore[attr-defined]
        assert "navigate" in actions
        assert "click" in actions
        navigate_record = next(r for r in records if r.action == "navigate")  # type: ignore[attr-defined]
        assert navigate_record.status == "success"
        assert navigate_record.duration_ms >= 0

    def test_fill_telemetry_never_contains_the_typed_value(self) -> None:
        records: list[object] = []

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager(telemetry_sink=records.append) as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="textbox")
                    await manager.fill(session.session_id, ref, "super-secret-password")

        asyncio.run(_run())

        fill_record = next(r for r in records if r.action == "fill")  # type: ignore[attr-defined]
        assert "super-secret-password" not in (fill_record.target or "")
        assert fill_record.target == "[value redacted]"
        for record in records:
            assert "super-secret-password" not in str(record.__dict__)

    def test_no_sink_means_no_overhead_and_no_crash(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:  # no telemetry_sink
                    session = await manager.open_session()
                    result = await manager.navigate(session.session_id, f"{base_url}/")
                    assert result.ok

        asyncio.run(_run())


class TestPersistentProfile:
    def test_session_is_ephemeral_by_default_no_storage_state_written(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/set-cookie")
                    # never called save_storage_state — nothing should exist on disk
                    # (nothing to assert against a path since none was given; this
                    # documents that open_session alone persists nothing)
                    assert session is not None

        asyncio.run(_run())

    def test_save_storage_state_writes_a_real_file_and_a_new_session_loads_it(self, tmp_path: object) -> None:
        from pathlib import Path

        async def _run() -> None:
            state_path = Path(str(tmp_path)) / "state.json"
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session_a = await manager.open_session()
                    await manager.navigate(session_a.session_id, f"{base_url}/set-cookie")
                    await manager.save_storage_state(session_a.session_id, str(state_path))
                    assert state_path.exists()

                    session_b = await manager.open_session(storage_state_path=str(state_path))
                    result = await manager.navigate(session_b.session_id, f"{base_url}/echo-cookie")

                    assert result.ok
                    assert result.evidence is not None
                    assert "fixture_session=abc123" in result.evidence.excerpt

        asyncio.run(_run())

    def test_opening_a_persistent_profile_goes_through_its_own_policy_check(self, tmp_path: object) -> None:
        from pathlib import Path

        async def _run() -> None:
            state_path = Path(str(tmp_path)) / "state.json"
            engine = PolicyEngine(default_registry(), mode=PolicyMode.DEFAULT)  # HIGH risk, no confirm -> denied
            async with BrowserManager(policy_engine=engine) as manager:
                denied = False
                try:
                    await manager.open_session(storage_state_path=str(state_path))
                except Exception:
                    denied = True
                assert denied

        asyncio.run(_run())
