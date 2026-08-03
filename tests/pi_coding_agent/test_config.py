"""Tests for pi_coding_agent config module."""

from __future__ import annotations

import os
from pathlib import Path

from pi_coding_agent.config import (
    APP_NAME,
    CONFIG_DIR_NAME,
    detect_install_method,
    get_config_dir,
    get_default_session_name,
    get_home_dir,
    get_package_dir,
    get_self_update_command,
    get_session_dir,
    is_bun_binary,
    is_bun_runtime,
    normalize_path,
)


def test_app_name() -> None:
    assert APP_NAME == "pi"


def test_config_dir_name() -> None:
    assert CONFIG_DIR_NAME == ".pi"


def test_get_home_dir() -> None:
    home = get_home_dir()
    assert isinstance(home, Path)
    assert home.exists()


def test_get_config_dir_default() -> None:
    os.environ.pop("PI_AGENT_DIR", None)
    config_dir = get_config_dir()
    assert config_dir == get_home_dir() / ".pi"


def test_get_config_dir_env_override() -> None:
    os.environ["PI_AGENT_DIR"] = "/tmp/test-pi-config"
    config_dir = get_config_dir()
    assert config_dir == Path("/tmp/test-pi-config")
    os.environ.pop("PI_AGENT_DIR", None)


def test_get_session_dir_default() -> None:
    os.environ.pop("PI_SESSION_DIR", None)
    os.environ.pop("PI_AGENT_DIR", None)
    session_dir = get_session_dir()
    assert session_dir == get_home_dir() / ".pi" / "sessions"


def test_get_session_dir_env_override() -> None:
    os.environ["PI_SESSION_DIR"] = "/tmp/test-sessions"
    session_dir = get_session_dir()
    assert session_dir == Path("/tmp/test-sessions")
    os.environ.pop("PI_SESSION_DIR", None)


def test_get_package_dir() -> None:
    pkg_dir = get_package_dir()
    assert isinstance(pkg_dir, Path)


def test_detect_install_method() -> None:
    method = detect_install_method()
    assert method in ("pip", "uv", "unknown")


def test_get_self_update_command() -> None:
    cmd = get_self_update_command()
    assert "install" in cmd
    assert "python-pi" in cmd


def test_normalize_path() -> None:
    result = normalize_path("/tmp")
    assert isinstance(result, str)
    assert len(result) > 0


def test_is_bun_binary() -> None:
    assert is_bun_binary() is False


def test_is_bun_runtime() -> None:
    assert is_bun_runtime() is False


def test_get_default_session_name() -> None:
    name = get_default_session_name()
    assert isinstance(name, str)
    assert len(name) > 0
