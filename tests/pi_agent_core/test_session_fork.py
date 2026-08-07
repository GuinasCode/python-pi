"""Tests for JsonlRepository.fork() entry replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent_core.session import JsonlRepository, SessionForkOptions


@pytest.mark.asyncio
async def test_fork_copies_source_entries(tmp_path: Path) -> None:
    repo = JsonlRepository(str(tmp_path))
    session = await repo.create()
    await session.append_message({"role": "user", "content": "hello"})
    await session.append_message({"role": "assistant", "content": "hi there"})

    sessions = await repo.list()
    assert len(sessions) == 1
    source_metadata = sessions[0]

    forked = await repo.fork(source_metadata, SessionForkOptions())

    entries = await forked.get_entries()
    assert len(entries) == 2
    assert await forked.get_leaf_id() is not None
