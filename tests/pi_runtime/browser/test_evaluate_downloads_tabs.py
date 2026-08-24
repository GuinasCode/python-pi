"""Slice B4 — browser_evaluate, upload, download, tabs."""

from __future__ import annotations

import asyncio

from fixtures import fixture_server

from pi_runtime.browser import BrowserManager, InteractionStatus


async def _ref_for(manager: BrowserManager, session_id: str, *, role: str, name: str | None = None) -> str:
    page_snapshot = await manager.snapshot(session_id)
    return next(n.ref for n in page_snapshot.nodes if n.role == role and (name is None or n.name == name))


class TestEvaluate:
    def test_evaluate_returns_a_small_json_preview(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.evaluate(session.session_id, "1 + 1")

                    assert result.ok
                    assert result.preview == "2"
                    assert not result.truncated

        asyncio.run(_run())

    def test_evaluate_can_read_dom_state(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    script = "document.getElementById('heading').textContent"
                    result = await manager.evaluate(session.session_id, script)

                    assert result.ok
                    assert "Fixture Home" in result.preview

        asyncio.run(_run())

    def test_large_evaluate_result_is_bounded_with_an_artifact(self, tmp_path: object) -> None:
        from pathlib import Path

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.evaluate(
                        session.session_id,
                        "Array.from({length: 50000}, (_, i) => i)",
                        artifacts_dir=Path(str(tmp_path)),
                    )

                    assert result.ok
                    assert result.truncated
                    assert result.artifact_path is not None
                    assert Path(result.artifact_path).exists()
                    assert len(result.preview) < result.total_chars

        asyncio.run(_run())

    def test_evaluate_syntax_error_is_reported_not_crashed(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/")

                    result = await manager.evaluate(session.session_id, "this is not valid js (((")

                    assert not result.ok
                    assert result.status == InteractionStatus.ERROR

        asyncio.run(_run())


class TestUpload:
    def test_upload_sets_the_file_input(self, tmp_path: object) -> None:
        from pathlib import Path

        async def _run() -> None:
            upload_file = Path(str(tmp_path)) / "hello.txt"
            upload_file.write_text("hello")

            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="button", name="Choose File")

                    result = await manager.upload(session.session_id, ref, [str(upload_file)])

                    assert result.ok
                    page = session.get_page()
                    assert page is not None
                    file_names = await page.locator("#upload").evaluate("el => Array.from(el.files).map(f => f.name)")
                    assert file_names == ["hello.txt"]

        asyncio.run(_run())


class TestDownload:
    def test_download_produces_an_artifact_with_provenance(self, tmp_path: object) -> None:
        from pathlib import Path

        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    await manager.navigate(session.session_id, f"{base_url}/form")
                    ref = await _ref_for(manager, session.session_id, role="link")

                    result = await manager.download_via_click(
                        session.session_id, ref, artifacts_dir=Path(str(tmp_path))
                    )

                    assert result.ok
                    assert result.artifact_path is not None
                    assert Path(result.artifact_path).read_bytes() == b"file contents for download test"
                    assert result.size_bytes == len(b"file contents for download test")
                    assert result.sha256 is not None
                    assert len(result.sha256) == 64

        asyncio.run(_run())


class TestTabs:
    def test_new_page_becomes_the_active_page(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    original_page_id = session.active_page_id

                    new_page_id = await manager.new_page(session.session_id, url=f"{base_url}/form")

                    assert new_page_id != original_page_id
                    assert session.active_page_id == new_page_id
                    pages = manager.list_pages(session.session_id)
                    assert len(pages) == 2

        asyncio.run(_run())

    def test_switch_page_changes_the_active_page(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    original_page_id = session.active_page_id
                    assert original_page_id is not None
                    await manager.new_page(session.session_id, url=f"{base_url}/form")

                    switched = manager.switch_page(session.session_id, original_page_id)

                    assert switched
                    assert session.active_page_id == original_page_id

        asyncio.run(_run())

    def test_close_page_removes_it_and_promotes_another_active_page(self) -> None:
        async def _run() -> None:
            with fixture_server() as base_url:
                async with BrowserManager() as manager:
                    session = await manager.open_session()
                    original_page_id = session.active_page_id
                    assert original_page_id is not None
                    await manager.new_page(session.session_id, url=f"{base_url}/form")

                    closed = await manager.close_page(session.session_id, original_page_id)

                    assert closed
                    assert original_page_id not in session.pages
                    assert session.active_page_id is not None

        asyncio.run(_run())

    def test_switch_to_unknown_page_returns_false(self) -> None:
        async def _run() -> None:
            async with BrowserManager() as manager:
                session = await manager.open_session()
                assert manager.switch_page(session.session_id, "nope") is False

        asyncio.run(_run())
