"""Configuration for the Pi coding agent.

Mirrors the Python-relevant parts of packages/coding-agent/src/config.ts.
Node/Bun-specific detection is replaced with Python equivalents.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "pi"
CONFIG_DIR_NAME = ".pi"
ENV_AGENT_DIR = "PI_AGENT_DIR"
ENV_SESSION_DIR = "PI_SESSION_DIR"

InstallMethod = str  # "pip", "uv", "unknown"


def get_home_dir() -> Path:
    """Get the user home directory."""
    return Path.home()


def get_config_dir() -> Path:
    """Get the Pi configuration directory (~/.pi)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        return Path(env_dir)
    return get_home_dir() / CONFIG_DIR_NAME


def get_session_dir() -> Path:
    """Get the session storage directory."""
    env_dir = os.environ.get(ENV_SESSION_DIR)
    if env_dir:
        return Path(env_dir)
    return get_config_dir() / "sessions"


def get_package_dir() -> Path:
    """Get the package installation directory."""
    return Path(__file__).parent.parent.parent


def get_docs_path() -> Path:
    """Directory of bundled pi documentation (referenced from the system
    prompt's "Pi documentation" section). Resolved relative to this
    installed package, not the dev-checkout repo root, so it works the
    same whether running from a git clone or a pip/uv install."""
    return Path(__file__).resolve().parent / "docs"


def get_examples_path() -> Path:
    """Directory of bundled pi examples (extensions, custom tools, ...)."""
    return Path(__file__).resolve().parent / "examples"


def detect_install_method() -> InstallMethod:
    """Detect how Pi was installed."""
    if os.environ.get("UV_TOOL_BIN_DIR") or os.environ.get("VIRTUAL_ENV"):
        return "uv"
    if "site-packages" in str(Path(__file__).resolve()):
        return "pip"
    return "unknown"


def get_self_update_command() -> str:
    """Get the command to self-update Pi."""
    method = detect_install_method()
    if method == "uv":
        return "uv pip install --upgrade python-pi"
    if method == "pip":
        return "pip install --upgrade python-pi"
    return "pip install --upgrade python-pi"


def normalize_path(path: str | Path) -> str:
    """Normalize a filesystem path."""
    return str(Path(path).resolve())


def is_bun_binary() -> bool:
    """Always False in Python port."""
    return False


def is_bun_runtime() -> bool:
    """Always False in Python port."""
    return False


def get_default_session_name() -> str:
    """Generate a default session name based on the current directory."""
    cwd = Path.cwd()
    return cwd.name or "session"


def ensure_config_dir() -> Path:
    """Ensure the config directory exists and return it."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def ensure_session_dir() -> Path:
    """Ensure the session directory exists and return it."""
    session_dir = get_session_dir()
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir
