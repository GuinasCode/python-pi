"""Session-wide test isolation.

Several tests construct real InteractiveSession/AgentSession/SettingsManager
objects. SettingsManager.get_memory_db_path() falls back to the module-level
pi_coding_agent.config.get_config_dir(), which reads the PI_AGENT_DIR env var
(or ~/.pi) — it does NOT use the agent_dir passed into SettingsManager.create().
Without this fixture, any test that doesn't explicitly stub the memory store
ends up reading/writing the real user's ~/.pi/memory.db instead of an
isolated per-test database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolated_pi_agent_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Function-scoped (not session-scoped): SettingsManager.get_memory_db_path()
    ignores the agent_dir passed to SettingsManager.create() and always falls
    back to the global PI_AGENT_DIR env var. A session-scoped override would
    make every test in the run share one memory.db, causing cross-test state
    leakage; a fresh directory per test keeps tests isolated from each other
    as well as from the real ~/.pi."""
    agent_dir = tmp_path_factory.mktemp("pi-agent-dir")
    original = os.environ.get("PI_AGENT_DIR")
    os.environ["PI_AGENT_DIR"] = str(agent_dir)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PI_AGENT_DIR", None)
        else:
            os.environ["PI_AGENT_DIR"] = original
