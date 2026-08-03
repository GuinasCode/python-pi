"""Settings management for the Pi coding agent.

Mirrors packages/coding-agent/src/core/settings-manager.ts.

Reads and writes ~/.pi/settings.json (global) and .pi/settings.json (project)
with deep-merge semantics, tracking modified fields for incremental persistence.

The TypeScript original uses proper-lockfile for concurrent access. This Python
port uses filelock (already a dependency) for the same purpose. An in-memory
storage backend is also provided for testing.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from filelock import FileLock

from pi_coding_agent.config import CONFIG_DIR_NAME, get_config_dir

# --- Defaults ---

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 120_000

# --- Settings sub-structures ---


@dataclass
class CompactionSettings:
    """Compaction configuration."""

    enabled: bool | None = None
    reserve_tokens: int | None = None
    keep_recent_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in (
                ("enabled", self.enabled),
                ("reserveTokens", self.reserve_tokens),
                ("keepRecentTokens", self.keep_recent_tokens),
            ) if v is not None
        }


@dataclass
class BranchSummarySettings:
    """Branch summary configuration."""

    reserve_tokens: int | None = None
    skip_prompt: bool | None = None


@dataclass
class ProviderRetrySettings:
    """Provider-level retry configuration."""

    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None


@dataclass
class RetrySettings:
    """Top-level retry configuration."""

    enabled: bool | None = None
    max_retries: int | None = None
    base_delay_ms: int | None = None
    provider: ProviderRetrySettings | None = None


@dataclass
class TerminalSettings:
    """Terminal display configuration."""

    show_images: bool | None = None
    image_width_cells: int | None = None
    clear_on_shrink: bool | None = None
    show_terminal_progress: bool | None = None


@dataclass
class ImageSettings:
    """Image handling configuration."""

    auto_resize: bool | None = None
    block_images: bool | None = None


@dataclass
class ThinkingBudgetsSettings:
    """Custom token budgets for thinking levels."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


@dataclass
class MarkdownSettings:
    """Markdown rendering configuration."""

    code_block_indent: str | None = None


@dataclass
class WarningSettings:
    """Warning display configuration."""

    anthropic_extra_usage: bool | None = None


DefaultProjectTrust = str  # "ask" | "always" | "never"
TransportSetting = str  # "sse" | "websocket" | "websocket-cached" | "auto"
PackageSource = str | dict[str, Any]
"""Package source: string form loads all resources; object form filters."""


class Settings:
    """Pi settings, stored as JSON at ~/.pi/settings.json (global) and .pi/settings.json (project).

    Uses a dict-based representation to allow flexible partial updates and
    migration from older formats. Keys mirror the TypeScript camelCase names
    so settings files remain compatible across both implementations.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data) if data else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the underlying dict."""
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    def remove(self, key: str) -> None:
        self._data.pop(key, None)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Settings):
            return NotImplemented
        return self._data == other._data

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Settings({self._data!r})"


# --- Deep merge ---


def deep_merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two settings dicts: overrides take precedence, nested dicts merge recursively.

    Arrays and primitives are replaced wholesale (override wins).
    """
    result = dict(base)
    for key, override_value in overrides.items():
        if override_value is None:
            continue
        base_value = result.get(key)
        if isinstance(override_value, dict) and isinstance(base_value, dict) and not isinstance(base_value, list):
            result[key] = {**base_value, **override_value}
        else:
            result[key] = override_value
    return result


# --- Settings storage backends ---


SettingsScope = str  # "global" | "project"


class SettingsStorage(Protocol):
    """Storage backend for settings with locking."""

    def with_lock(self, scope: SettingsScope, fn: Any) -> Any:
        """Run fn under an exclusive lock for the given scope.

        fn receives the current file content (str | None) and returns the new
        content (str | None). Returning None means no write.
        """
        ...


class FileSettingsStorage:
    """File-based settings storage with file locking."""

    def __init__(self, cwd: str | Path, agent_dir: str | Path) -> None:
        resolved_cwd = Path(cwd).resolve()
        resolved_agent_dir = Path(agent_dir).resolve()
        self.global_settings_path = resolved_agent_dir / "settings.json"
        self.project_settings_path = resolved_cwd / CONFIG_DIR_NAME / "settings.json"

    def _lock_path(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    def with_lock(self, scope: SettingsScope, fn: Any) -> Any:
        path = self.global_settings_path if scope == "global" else self.project_settings_path
        path_parent = path.parent
        file_exists = path.exists()
        lock = FileLock(str(self._lock_path(path)), timeout=10)
        try:
            lock.acquire()
            current = path.read_text(encoding="utf-8") if file_exists else None
            next_content = fn(current)
            if next_content is not None:
                path_parent.mkdir(parents=True, exist_ok=True)
                path.write_text(next_content, encoding="utf-8")
            return next_content
        finally:
            if lock.is_locked:
                lock.release()
            # Clean up lock file if it exists and is stale
            lock_path = self._lock_path(path)
            with contextlib.suppress(OSError):
                if lock_path.exists():
                    lock_path.unlink()


class InMemorySettingsStorage:
    """In-memory settings storage for testing."""

    def __init__(self) -> None:
        self._global: str | None = None
        self._project: str | None = None

    def with_lock(self, scope: SettingsScope, fn: Any) -> Any:
        current = self._global if scope == "global" else self._project
        next_content = fn(current)
        if next_content is not None:
            if scope == "global":
                self._global = next_content
            else:
                self._project = next_content
        return next_content


# --- Errors ---


@dataclass
class SettingsError:
    """An error that occurred while loading or writing settings."""

    scope: SettingsScope
    error: Exception


# --- Migration ---


def migrate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Migrate old settings format to new format. Returns a new dict."""
    result = dict(settings)

    # Migrate queueMode -> steeringMode
    if "queueMode" in result and "steeringMode" not in result:
        result["steeringMode"] = result.pop("queueMode")

    # Migrate legacy websockets boolean -> transport enum
    if "transport" not in result and isinstance(result.get("websockets"), bool):
        result["transport"] = "websocket" if result["websockets"] else "sse"
        result.pop("websockets", None)

    # Migrate old skills object format to new array format
    skills = result.get("skills")
    if isinstance(skills, dict) and not isinstance(skills, list):
        skills_settings = skills
        if (
            skills_settings.get("enableSkillCommands") is not None
            and result.get("enableSkillCommands") is None
        ):
            result["enableSkillCommands"] = skills_settings["enableSkillCommands"]
        custom_dirs = skills_settings.get("customDirectories")
        if isinstance(custom_dirs, list) and len(custom_dirs) > 0:
            result["skills"] = custom_dirs
        else:
            result.pop("skills", None)

    # Migrate retry.maxDelayMs -> retry.provider.maxRetryDelayMs
    retry = result.get("retry")
    if isinstance(retry, dict) and not isinstance(retry, list):
        provider = retry.get("provider")
        provider_settings = provider if isinstance(provider, dict) else None
        max_delay = retry.get("maxDelayMs")
        if isinstance(max_delay, (int, float)) and (
            provider_settings is None or provider_settings.get("maxRetryDelayMs") is None
        ):
            new_provider = dict(provider_settings or {})
            new_provider["maxRetryDelayMs"] = max_delay
            retry["provider"] = new_provider
        retry.pop("maxDelayMs", None)

    return result


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    """Parse an HTTP idle timeout value. Returns ms or None if unset/invalid."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subtype of int, reject it
        return None
    if isinstance(value, (int, float)):
        if value < 0 or value != value:  # NaN check
            return None
        return int(value)
    return None


def parse_timeout_setting(value: Any, setting_name: str) -> int | None:
    """Parse a timeout setting, raising on invalid non-None values."""
    timeout_ms = parse_http_idle_timeout_ms(value)
    if timeout_ms is not None:
        return timeout_ms
    if value is not None:
        raise ValueError(f"Invalid {setting_name} setting: {value!r}")
    return None


# --- SettingsManager ---


@dataclass
class SettingsManagerCreateOptions:
    project_trusted: bool | None = None


class SettingsManager:
    """Manages Pi settings with global/project merge and incremental persistence.

    Mirrors the SettingsManager class from settings-manager.ts. Tracks which
    fields were modified during a session so only those fields are written
    back, preserving unrelated keys in the settings file.
    """

    def __init__(
        self,
        storage: SettingsStorage,
        initial_global: Settings,
        initial_project: Settings,
        global_load_error: Exception | None = None,
        project_load_error: Exception | None = None,
        initial_errors: list[SettingsError] | None = None,
        project_trusted: bool = True,
    ) -> None:
        self._storage = storage
        self._global_settings = initial_global
        self._project_settings = initial_project
        self._project_trusted = project_trusted
        self._global_settings_load_error = global_load_error
        self._project_settings_load_error = project_load_error
        self._errors: list[SettingsError] = list(initial_errors or [])
        self._modified_fields: set[str] = set()
        self._modified_nested_fields: dict[str, set[str]] = {}
        self._modified_project_fields: set[str] = set()
        self._modified_project_nested_fields: dict[str, set[str]] = {}
        self._settings = Settings(deep_merge_settings(
            self._global_settings.to_dict(),
            self._project_settings.to_dict(),
        ))

    # --- Factory methods ---

    @classmethod
    def create(
        cls,
        cwd: str | Path,
        agent_dir: str | Path | None = None,
        options: SettingsManagerCreateOptions | None = None,
    ) -> SettingsManager:
        """Create a SettingsManager that loads from files."""
        resolved_agent_dir = Path(agent_dir) if agent_dir else get_config_dir()
        storage = FileSettingsStorage(cwd, resolved_agent_dir)
        return cls.from_storage(storage, options)

    @classmethod
    def from_storage(
        cls,
        storage: SettingsStorage,
        options: SettingsManagerCreateOptions | None = None,
    ) -> SettingsManager:
        """Create a SettingsManager from an arbitrary storage backend."""
        project_trusted = (options.project_trusted if options and options.project_trusted is not None else True)
        global_load = cls._try_load_from_storage(storage, "global")
        project_load = cls._try_load_from_storage(storage, "project", project_trusted)
        initial_errors: list[SettingsError] = []
        if global_load[1]:
            initial_errors.append(SettingsError(scope="global", error=global_load[1]))
        if project_load[1]:
            initial_errors.append(SettingsError(scope="project", error=project_load[1]))
        return cls(
            storage,
            global_load[0],
            project_load[0],
            global_load[1],
            project_load[1],
            initial_errors,
            project_trusted,
        )

    @classmethod
    def in_memory(
        cls,
        settings: dict[str, Any] | None = None,
        options: SettingsManagerCreateOptions | None = None,
    ) -> SettingsManager:
        """Create an in-memory SettingsManager (no file I/O)."""
        storage = InMemorySettingsStorage()
        migrated = migrate_settings(dict(settings) if settings else {})
        storage.with_lock("global", lambda _: json.dumps(migrated, indent=2))
        return cls.from_storage(storage, options)

    # --- Internal load helpers ---

    @staticmethod
    def _load_from_storage(
        storage: SettingsStorage,
        scope: SettingsScope,
        project_trusted: bool = True,
    ) -> Settings:
        if scope == "project" and not project_trusted:
            return Settings()
        content: str | None = None

        def capture(current: str | None) -> None:
            nonlocal content
            content = current

        storage.with_lock(scope, capture)
        if not content:
            return Settings()
        parsed = json.loads(content)
        return Settings(migrate_settings(parsed))

    @staticmethod
    def _try_load_from_storage(
        storage: SettingsStorage,
        scope: SettingsScope,
        project_trusted: bool = True,
    ) -> tuple[Settings, Exception | None]:
        try:
            return (SettingsManager._load_from_storage(storage, scope, project_trusted), None)
        except Exception as exc:
            return (Settings(), exc)

    # --- Accessors ---

    def get_global_settings(self) -> Settings:
        return Settings(self._global_settings.to_dict())

    def get_project_settings(self) -> Settings:
        return Settings(self._project_settings.to_dict())

    def is_project_trusted(self) -> bool:
        return self._project_trusted

    def set_project_trusted(self, trusted: bool) -> None:
        if self._project_trusted == trusted:
            return
        self._project_trusted = trusted
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()
        if not trusted:
            self._project_settings = Settings()
            self._project_settings_load_error = None
            self._settings = Settings(deep_merge_settings(
                self._global_settings.to_dict(),
                self._project_settings.to_dict(),
            ))
            return
        project_load = self._try_load_from_storage(self._storage, "project", trusted)
        self._project_settings = project_load[0]
        self._project_settings_load_error = project_load[1]
        if project_load[1]:
            self._record_error("project", project_load[1])
        self._settings = Settings(deep_merge_settings(
            self._global_settings.to_dict(),
            self._project_settings.to_dict(),
        ))

    def reload(self) -> None:
        """Reload settings from storage."""
        global_load = self._try_load_from_storage(self._storage, "global")
        if not global_load[1]:
            self._global_settings = global_load[0]
            self._global_settings_load_error = None
        else:
            self._global_settings_load_error = global_load[1]
            self._record_error("global", global_load[1])

        self._modified_fields.clear()
        self._modified_nested_fields.clear()
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

        project_load = self._try_load_from_storage(self._storage, "project", self._project_trusted)
        if not project_load[1]:
            self._project_settings = project_load[0]
            self._project_settings_load_error = None
        else:
            self._project_settings_load_error = project_load[1]
            self._record_error("project", project_load[1])

        self._settings = Settings(deep_merge_settings(
            self._global_settings.to_dict(),
            self._project_settings.to_dict(),
        ))

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply additional overrides on top of current settings."""
        self._settings = Settings(deep_merge_settings(self._settings.to_dict(), overrides))

    # --- Modified field tracking ---

    def _mark_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_fields.add(field)
        if nested_key:
            self._modified_nested_fields.setdefault(field, set()).add(nested_key)

    def _mark_project_modified(self, field: str, nested_key: str | None = None) -> None:
        self._modified_project_fields.add(field)
        if nested_key:
            self._modified_project_nested_fields.setdefault(field, set()).add(nested_key)

    def _assert_project_trusted_for_write(self) -> None:
        if not self._project_trusted:
            raise RuntimeError("Project is not trusted; refusing to write project settings")

    def _record_error(self, scope: SettingsScope, error: Any) -> None:
        normalized = error if isinstance(error, Exception) else Exception(str(error))
        self._errors.append(SettingsError(scope=scope, error=normalized))

    def _clear_modified_scope(self, scope: SettingsScope) -> None:
        if scope == "global":
            self._modified_fields.clear()
            self._modified_nested_fields.clear()
            return
        self._modified_project_fields.clear()
        self._modified_project_nested_fields.clear()

    def _persist_scoped_settings(
        self,
        scope: SettingsScope,
        snapshot_settings: Settings,
        modified_fields: set[str],
        modified_nested_fields: dict[str, set[str]],
    ) -> None:
        def transform(current: str | None) -> str | None:
            current_file_settings: dict[str, Any] = {}
            if current:
                current_file_settings = migrate_settings(json.loads(current))
            merged: dict[str, Any] = dict(current_file_settings)
            for field_ in modified_fields:
                value = snapshot_settings.get(field_)
                if field_ in modified_nested_fields and isinstance(value, dict):
                    nested_modified = modified_nested_fields[field_]
                    base_nested = current_file_settings.get(field_, {})
                    if not isinstance(base_nested, dict):
                        base_nested = {}
                    merged_nested = dict(base_nested)
                    for nested_key in nested_modified:
                        merged_nested[nested_key] = value.get(nested_key)
                    merged[field_] = merged_nested
                else:
                    merged[field_] = value
            return json.dumps(merged, indent=2)

        self._storage.with_lock(scope, transform)

    def _save(self) -> None:
        self._settings = Settings(deep_merge_settings(
            self._global_settings.to_dict(),
            self._project_settings.to_dict(),
        ))
        if self._global_settings_load_error:
            return
        snapshot = Settings(self._global_settings.to_dict())
        modified_fields = set(self._modified_fields)
        modified_nested = {k: set(v) for k, v in self._modified_nested_fields.items()}
        self._persist_scoped_settings("global", snapshot, modified_fields, modified_nested)
        self._clear_modified_scope("global")

    def _save_project_settings(self, settings: Settings) -> None:
        self._assert_project_trusted_for_write()
        self._project_settings = Settings(settings.to_dict())
        self._settings = Settings(deep_merge_settings(
            self._global_settings.to_dict(),
            self._project_settings.to_dict(),
        ))
        if self._project_settings_load_error:
            return
        snapshot = Settings(self._project_settings.to_dict())
        modified_fields = set(self._modified_project_fields)
        modified_nested = {k: set(v) for k, v in self._modified_project_nested_fields.items()}
        self._persist_scoped_settings("project", snapshot, modified_fields, modified_nested)
        self._clear_modified_scope("project")

    def _update_project_settings(self, field: str, update: Any) -> None:
        self._assert_project_trusted_for_write()
        project_settings = Settings(self._project_settings.to_dict())
        update(project_settings)
        self._mark_project_modified(field)
        self._save_project_settings(project_settings)

    def drain_errors(self) -> list[SettingsError]:
        drained = list(self._errors)
        self._errors = []
        return drained

    # --- Typed accessors ---

    def get_last_changelog_version(self) -> str | None:
        return self._settings.get("lastChangelogVersion")

    def set_last_changelog_version(self, version: str) -> None:
        self._global_settings.set("lastChangelogVersion", version)
        self._mark_modified("lastChangelogVersion")
        self._save()

    def get_session_dir(self) -> str | None:
        session_dir = self._settings.get("sessionDir")
        if session_dir is None:
            return None
        return str(Path(session_dir).resolve())

    def get_default_provider(self) -> str | None:
        return self._settings.get("defaultProvider")

    def get_default_model(self) -> str | None:
        return self._settings.get("defaultModel")

    def set_default_provider(self, provider: str) -> None:
        self._global_settings.set("defaultProvider", provider)
        self._mark_modified("defaultProvider")
        self._save()

    def set_default_model(self, model_id: str) -> None:
        self._global_settings.set("defaultModel", model_id)
        self._mark_modified("defaultModel")
        self._save()

    def set_default_model_and_provider(self, provider: str, model_id: str) -> None:
        self._global_settings.set("defaultProvider", provider)
        self._global_settings.set("defaultModel", model_id)
        self._mark_modified("defaultProvider")
        self._mark_modified("defaultModel")
        self._save()

    def get_steering_mode(self) -> str:
        return self._settings.get("steeringMode") or "one-at-a-time"

    def set_steering_mode(self, mode: str) -> None:
        self._global_settings.set("steeringMode", mode)
        self._mark_modified("steeringMode")
        self._save()

    def get_follow_up_mode(self) -> str:
        return self._settings.get("followUpMode") or "one-at-a-time"

    def set_follow_up_mode(self, mode: str) -> None:
        self._global_settings.set("followUpMode", mode)
        self._mark_modified("followUpMode")
        self._save()

    def get_theme_setting(self) -> str | None:
        value = self._settings.get("theme")
        return value if isinstance(value, str) else None

    def get_theme(self) -> str | None:
        theme = self.get_theme_setting()
        return None if (theme and "/" in theme) else theme

    def set_theme(self, theme: str) -> None:
        self._global_settings.set("theme", theme)
        self._mark_modified("theme")
        self._save()

    def get_default_thinking_level(self) -> str | None:
        return self._settings.get("defaultThinkingLevel")

    def set_default_thinking_level(self, level: str) -> None:
        self._global_settings.set("defaultThinkingLevel", level)
        self._mark_modified("defaultThinkingLevel")
        self._save()

    def get_transport(self) -> TransportSetting:
        return self._settings.get("transport") or "auto"

    def set_transport(self, transport: TransportSetting) -> None:
        self._global_settings.set("transport", transport)
        self._mark_modified("transport")
        self._save()

    def get_compaction_enabled(self) -> bool:
        compaction = self._settings.get("compaction")
        return compaction.get("enabled", True) if isinstance(compaction, dict) else True

    def set_compaction_enabled(self, enabled: bool) -> None:
        compaction = self._global_settings.get("compaction")
        if not isinstance(compaction, dict):
            compaction = {}
        compaction["enabled"] = enabled
        self._global_settings.set("compaction", compaction)
        self._mark_modified("compaction", "enabled")
        self._save()

    def get_compaction_reserve_tokens(self) -> int:
        compaction = self._settings.get("compaction")
        return compaction.get("reserveTokens", 16384) if isinstance(compaction, dict) else 16384

    def get_compaction_keep_recent_tokens(self) -> int:
        compaction = self._settings.get("compaction")
        return compaction.get("keepRecentTokens", 20000) if isinstance(compaction, dict) else 20000

    def get_compaction_settings(self) -> dict[str, Any]:
        return {
            "enabled": self.get_compaction_enabled(),
            "reserveTokens": self.get_compaction_reserve_tokens(),
            "keepRecentTokens": self.get_compaction_keep_recent_tokens(),
        }

    def get_branch_summary_settings(self) -> dict[str, Any]:
        branch_summary = self._settings.get("branchSummary")
        if not isinstance(branch_summary, dict):
            branch_summary = {}
        return {
            "reserveTokens": branch_summary.get("reserveTokens", 16384),
            "skipPrompt": branch_summary.get("skipPrompt", False),
        }

    def get_branch_summary_skip_prompt(self) -> bool:
        branch_summary = self._settings.get("branchSummary")
        return branch_summary.get("skipPrompt", False) if isinstance(branch_summary, dict) else False

    def get_retry_enabled(self) -> bool:
        retry = self._settings.get("retry")
        return retry.get("enabled", True) if isinstance(retry, dict) else True

    def set_retry_enabled(self, enabled: bool) -> None:
        retry = self._global_settings.get("retry")
        if not isinstance(retry, dict):
            retry = {}
        retry["enabled"] = enabled
        self._global_settings.set("retry", retry)
        self._mark_modified("retry", "enabled")
        self._save()

    def get_retry_settings(self) -> dict[str, Any]:
        retry = self._settings.get("retry")
        if not isinstance(retry, dict):
            retry = {}
        return {
            "enabled": retry.get("enabled", True),
            "maxRetries": retry.get("maxRetries", 3),
            "baseDelayMs": retry.get("baseDelayMs", 2000),
        }

    def get_http_idle_timeout_ms(self) -> int:
        value = self._settings.get("httpIdleTimeoutMs")
        try:
            parsed = parse_timeout_setting(value, "httpIdleTimeoutMs")
        except ValueError:
            parsed = None
        return parsed if parsed is not None else DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def set_http_idle_timeout_ms(self, timeout_ms: int) -> None:
        if not isinstance(timeout_ms, (int, float)) or timeout_ms < 0 or timeout_ms != timeout_ms:
            raise ValueError(f"Invalid httpIdleTimeoutMs setting: {timeout_ms!r}")
        self._global_settings.set("httpIdleTimeoutMs", int(timeout_ms))
        self._mark_modified("httpIdleTimeoutMs")
        self._save()

    def get_provider_retry_settings(self) -> dict[str, Any]:
        retry = self._settings.get("retry")
        provider = retry.get("provider") if isinstance(retry, dict) else None
        if not isinstance(provider, dict):
            provider = {}
        return {
            "timeoutMs": provider.get("timeoutMs"),
            "maxRetries": provider.get("maxRetries"),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs", 60000),
        }

    def get_websocket_connect_timeout_ms(self) -> int | None:
        value = self._settings.get("websocketConnectTimeoutMs")
        try:
            return parse_timeout_setting(value, "websocketConnectTimeoutMs")
        except ValueError:
            return None

    def get_hide_thinking_block(self) -> bool:
        return bool(self._settings.get("hideThinkingBlock", False))

    def get_show_cache_miss_notices(self) -> bool:
        return bool(self._settings.get("showCacheMissNotices", False))

    def get_external_editor_command(self) -> str:
        configured = self._settings.get("externalEditor")
        if isinstance(configured, str) and configured.strip():
            return configured
        env_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if env_editor:
            return env_editor
        return "notepad" if os.name == "nt" else "nano"

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._global_settings.set("hideThinkingBlock", hide)
        self._mark_modified("hideThinkingBlock")
        self._save()

    def set_show_cache_miss_notices(self, show: bool) -> None:
        self._global_settings.set("showCacheMissNotices", show)
        self._mark_modified("showCacheMissNotices")
        self._save()

    def get_shell_path(self) -> str | None:
        shell_path = self._settings.get("shellPath")
        return str(Path(shell_path).resolve()) if shell_path else shell_path

    def set_shell_path(self, path: str | None) -> None:
        self._global_settings.set("shellPath", path)
        self._mark_modified("shellPath")
        self._save()

    def get_quiet_startup(self) -> bool:
        return bool(self._settings.get("quietStartup", False))

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings.set("quietStartup", quiet)
        self._mark_modified("quietStartup")
        self._save()

    def get_default_project_trust(self) -> DefaultProjectTrust:
        value = self._global_settings.get("defaultProjectTrust")
        return value if value in ("always", "never") else "ask"

    def set_default_project_trust(self, default_project_trust: DefaultProjectTrust) -> None:
        self._global_settings.set("defaultProjectTrust", default_project_trust)
        self._mark_modified("defaultProjectTrust")
        self._save()

    def get_shell_command_prefix(self) -> str | None:
        return self._settings.get("shellCommandPrefix")

    def set_shell_command_prefix(self, prefix: str | None) -> None:
        self._global_settings.set("shellCommandPrefix", prefix)
        self._mark_modified("shellCommandPrefix")
        self._save()

    def get_npm_command(self) -> list[str] | None:
        cmd = self._settings.get("npmCommand")
        return list(cmd) if isinstance(cmd, list) else None

    def set_npm_command(self, command: list[str] | None) -> None:
        self._global_settings.set("npmCommand", list(command) if command else None)
        self._mark_modified("npmCommand")
        self._save()

    def get_collapse_changelog(self) -> bool:
        return bool(self._settings.get("collapseChangelog", False))

    def set_collapse_changelog(self, collapse: bool) -> None:
        self._global_settings.set("collapseChangelog", collapse)
        self._mark_modified("collapseChangelog")
        self._save()

    def get_enable_install_telemetry(self) -> bool:
        return bool(self._settings.get("enableInstallTelemetry", True))

    def set_enable_install_telemetry(self, enabled: bool) -> None:
        self._global_settings.set("enableInstallTelemetry", enabled)
        self._mark_modified("enableInstallTelemetry")
        self._save()

    def get_enable_analytics(self) -> bool:
        return bool(self._settings.get("enableAnalytics", False))

    def get_tracking_id(self) -> str | None:
        return self._settings.get("trackingId")

    def set_enable_analytics(self, enabled: bool) -> None:
        self._global_settings.set("enableAnalytics", enabled)
        self._mark_modified("enableAnalytics")
        if enabled and not self._global_settings.get("trackingId"):
            tracking_id = str(uuid.uuid4())
            self._global_settings.set("trackingId", tracking_id)
            self._mark_modified("trackingId")
        self._save()

    def get_packages(self) -> list[PackageSource]:
        packages = self._settings.get("packages")
        return list(packages) if isinstance(packages, list) else []

    def set_packages(self, packages: list[PackageSource]) -> None:
        self._global_settings.set("packages", packages)
        self._mark_modified("packages")
        self._save()

    def set_project_packages(self, packages: list[PackageSource]) -> None:
        def update(settings: Settings) -> None:
            settings.set("packages", packages)
        self._update_project_settings("packages", update)

    def get_extension_paths(self) -> list[str]:
        exts = self._settings.get("extensions")
        return list(exts) if isinstance(exts, list) else []

    def set_extension_paths(self, paths: list[str]) -> None:
        self._global_settings.set("extensions", paths)
        self._mark_modified("extensions")
        self._save()

    def set_project_extension_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings.set("extensions", paths)
        self._update_project_settings("extensions", update)

    def get_skill_paths(self) -> list[str]:
        skills = self._settings.get("skills")
        return list(skills) if isinstance(skills, list) else []

    def set_skill_paths(self, paths: list[str]) -> None:
        self._global_settings.set("skills", paths)
        self._mark_modified("skills")
        self._save()

    def set_project_skill_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings.set("skills", paths)
        self._update_project_settings("skills", update)

    def get_prompt_template_paths(self) -> list[str]:
        prompts = self._settings.get("prompts")
        return list(prompts) if isinstance(prompts, list) else []

    def set_prompt_template_paths(self, paths: list[str]) -> None:
        self._global_settings.set("prompts", paths)
        self._mark_modified("prompts")
        self._save()

    def set_project_prompt_template_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings.set("prompts", paths)
        self._update_project_settings("prompts", update)

    def get_theme_paths(self) -> list[str]:
        themes = self._settings.get("themes")
        return list(themes) if isinstance(themes, list) else []

    def set_theme_paths(self, paths: list[str]) -> None:
        self._global_settings.set("themes", paths)
        self._mark_modified("themes")
        self._save()

    def set_project_theme_paths(self, paths: list[str]) -> None:
        def update(settings: Settings) -> None:
            settings.set("themes", paths)
        self._update_project_settings("themes", update)

    def get_enable_skill_commands(self) -> bool:
        return bool(self._settings.get("enableSkillCommands", True))

    def set_enable_skill_commands(self, enabled: bool) -> None:
        self._global_settings.set("enableSkillCommands", enabled)
        self._mark_modified("enableSkillCommands")
        self._save()

    def get_thinking_budgets(self) -> ThinkingBudgetsSettings | None:
        budgets = self._settings.get("thinkingBudgets")
        if not isinstance(budgets, dict):
            return None
        return ThinkingBudgetsSettings(
            minimal=budgets.get("minimal"),
            low=budgets.get("low"),
            medium=budgets.get("medium"),
            high=budgets.get("high"),
        )

    def get_show_images(self) -> bool:
        terminal = self._settings.get("terminal")
        return terminal.get("showImages", True) if isinstance(terminal, dict) else True

    def set_show_images(self, show: bool) -> None:
        terminal = self._global_settings.get("terminal")
        if not isinstance(terminal, dict):
            terminal = {}
        terminal["showImages"] = show
        self._global_settings.set("terminal", terminal)
        self._mark_modified("terminal", "showImages")
        self._save()

    def get_image_width_cells(self) -> int:
        terminal = self._settings.get("terminal")
        if not isinstance(terminal, dict):
            return 60
        width = terminal.get("imageWidthCells")
        if not isinstance(width, (int, float)) or width != width:
            return 60
        return max(1, int(width))

    def set_image_width_cells(self, width: int) -> None:
        terminal = self._global_settings.get("terminal")
        if not isinstance(terminal, dict):
            terminal = {}
        terminal["imageWidthCells"] = max(1, int(width))
        self._global_settings.set("terminal", terminal)
        self._mark_modified("terminal", "imageWidthCells")
        self._save()

    def get_clear_on_shrink(self) -> bool:
        terminal = self._settings.get("terminal")
        if isinstance(terminal, dict) and terminal.get("clearOnShrink") is not None:
            return bool(terminal["clearOnShrink"])
        return os.environ.get("PI_CLEAR_ON_SHRINK") == "1"

    def set_clear_on_shrink(self, enabled: bool) -> None:
        terminal = self._global_settings.get("terminal")
        if not isinstance(terminal, dict):
            terminal = {}
        terminal["clearOnShrink"] = enabled
        self._global_settings.set("terminal", terminal)
        self._mark_modified("terminal", "clearOnShrink")
        self._save()

    def get_show_terminal_progress(self) -> bool:
        terminal = self._settings.get("terminal")
        return bool(terminal.get("showTerminalProgress", False)) if isinstance(terminal, dict) else False

    def set_show_terminal_progress(self, enabled: bool) -> None:
        terminal = self._global_settings.get("terminal")
        if not isinstance(terminal, dict):
            terminal = {}
        terminal["showTerminalProgress"] = enabled
        self._global_settings.set("terminal", terminal)
        self._mark_modified("terminal", "showTerminalProgress")
        self._save()

    def get_ui_mode(self) -> str:
        return "fullscreen" if self._settings.get("uiMode") == "fullscreen" else "regular"

    def set_ui_mode(self, mode: str) -> None:
        self._global_settings.set("uiMode", mode)
        self._mark_modified("uiMode")
        self._save()

    def get_fullscreen_scrollbar(self) -> str:
        mode = self._settings.get("fullscreenScrollbar")
        return mode if mode in ("always", "hidden") else "auto"

    def set_fullscreen_scrollbar(self, mode: str) -> None:
        self._global_settings.set("fullscreenScrollbar", mode)
        self._mark_modified("fullscreenScrollbar")
        self._save()

    def get_image_auto_resize(self) -> bool:
        images = self._settings.get("images")
        return images.get("autoResize", True) if isinstance(images, dict) else True

    def set_image_auto_resize(self, enabled: bool) -> None:
        images = self._global_settings.get("images")
        if not isinstance(images, dict):
            images = {}
        images["autoResize"] = enabled
        self._global_settings.set("images", images)
        self._mark_modified("images", "autoResize")
        self._save()

    def get_block_images(self) -> bool:
        images = self._settings.get("images")
        return bool(images.get("blockImages", False)) if isinstance(images, dict) else False

    def set_block_images(self, blocked: bool) -> None:
        images = self._global_settings.get("images")
        if not isinstance(images, dict):
            images = {}
        images["blockImages"] = blocked
        self._global_settings.set("images", images)
        self._mark_modified("images", "blockImages")
        self._save()

    def get_enabled_models(self) -> list[str] | None:
        models = self._settings.get("enabledModels")
        return list(models) if isinstance(models, list) else None

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._global_settings.set("enabledModels", patterns)
        self._mark_modified("enabledModels")
        self._save()

    def get_double_escape_action(self) -> str:
        return self._settings.get("doubleEscapeAction") or "tree"

    def set_double_escape_action(self, action: str) -> None:
        self._global_settings.set("doubleEscapeAction", action)
        self._mark_modified("doubleEscapeAction")
        self._save()

    def get_tree_filter_mode(self) -> str:
        mode = self._settings.get("treeFilterMode")
        valid = ["default", "no-tools", "user-only", "labeled-only", "all"]
        return mode if mode in valid else "default"

    def set_tree_filter_mode(self, mode: str) -> None:
        self._global_settings.set("treeFilterMode", mode)
        self._mark_modified("treeFilterMode")
        self._save()

    def get_show_hardware_cursor(self) -> bool:
        return bool(self._settings.get("showHardwareCursor", os.environ.get("PI_HARDWARE_CURSOR") == "1"))

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        self._global_settings.set("showHardwareCursor", enabled)
        self._mark_modified("showHardwareCursor")
        self._save()

    def get_editor_padding_x(self) -> int:
        return int(self._settings.get("editorPaddingX", 0))

    def set_editor_padding_x(self, padding: int) -> None:
        self._global_settings.set("editorPaddingX", max(0, min(3, int(padding))))
        self._mark_modified("editorPaddingX")
        self._save()

    def get_output_pad(self) -> int:
        return 0 if self._settings.get("outputPad") == 0 else 1

    def set_output_pad(self, padding: int) -> None:
        self._global_settings.set("outputPad", padding)
        self._mark_modified("outputPad")
        self._save()

    def get_autocomplete_max_visible(self) -> int:
        return int(self._settings.get("autocompleteMaxVisible", 5))

    def set_autocomplete_max_visible(self, max_visible: int) -> None:
        self._global_settings.set("autocompleteMaxVisible", max(3, min(20, int(max_visible))))
        self._mark_modified("autocompleteMaxVisible")
        self._save()

    def get_code_block_indent(self) -> str:
        markdown = self._settings.get("markdown")
        return markdown.get("codeBlockIndent", "  ") if isinstance(markdown, dict) else "  "

    def get_warnings(self) -> dict[str, Any]:
        warnings = self._settings.get("warnings")
        return dict(warnings) if isinstance(warnings, dict) else {}

    def set_warnings(self, warnings: dict[str, Any]) -> None:
        self._global_settings.set("warnings", dict(warnings))
        self._mark_modified("warnings")
        self._save()
