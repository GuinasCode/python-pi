"""Tests for pi_coding_agent.settings_manager module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_coding_agent.settings_manager import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    InMemorySettingsStorage,
    Settings,
    SettingsManager,
    SettingsManagerCreateOptions,
    deep_merge_settings,
    migrate_settings,
    parse_http_idle_timeout_ms,
    parse_timeout_setting,
)

# --- Settings dataclass ---


class TestSettings:
    def test_empty(self) -> None:
        s = Settings()
        assert s.to_dict() == {}
        assert s.get("missing") is None
        assert s.get("missing", "default") == "default"

    def test_with_data(self) -> None:
        s = Settings({"a": 1, "b": "two"})
        assert s.get("a") == 1
        assert s.get("b") == "two"
        assert s.has("a")
        assert not s.has("c")

    def test_set_and_remove(self) -> None:
        s = Settings()
        s.set("x", 42)
        assert s.get("x") == 42
        s.remove("x")
        assert not s.has("x")

    def test_equality(self) -> None:
        a = Settings({"k": 1})
        b = Settings({"k": 1})
        c = Settings({"k": 2})
        assert a == b
        assert a != c


# --- deep_merge_settings ---


class TestDeepMerge:
    def test_primitives_override(self) -> None:
        result = deep_merge_settings({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_none_skipped(self) -> None:
        result = deep_merge_settings({"a": 1}, {"a": None, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dicts_merge(self) -> None:
        base = {"compaction": {"enabled": True, "reserveTokens": 100}}
        overrides = {"compaction": {"reserveTokens": 200}}
        result = deep_merge_settings(base, overrides)
        assert result == {"compaction": {"enabled": True, "reserveTokens": 200}}

    def test_arrays_replace(self) -> None:
        result = deep_merge_settings({"skills": ["a", "b"]}, {"skills": ["c"]})
        assert result == {"skills": ["c"]}

    def test_empty_overrides(self) -> None:
        base = {"a": 1, "b": 2}
        result = deep_merge_settings(base, {})
        assert result == base


# --- migrate_settings ---


class TestMigrateSettings:
    def test_queue_mode_migration(self) -> None:
        result = migrate_settings({"queueMode": "all"})
        assert result.get("steeringMode") == "all"
        assert "queueMode" not in result

    def test_websockets_bool_migration(self) -> None:
        result = migrate_settings({"websockets": True})
        assert result.get("transport") == "websocket"
        result2 = migrate_settings({"websockets": False})
        assert result2.get("transport") == "sse"

    def test_skills_object_migration(self) -> None:
        result = migrate_settings({"skills": {"enableSkillCommands": False, "customDirectories": ["/path"]}})
        assert result.get("skills") == ["/path"]
        assert result.get("enableSkillCommands") is False

    def test_skills_object_no_dirs(self) -> None:
        result = migrate_settings({"skills": {"enableSkillCommands": True}})
        assert "skills" not in result
        assert result.get("enableSkillCommands") is True

    def test_retry_max_delay_migration(self) -> None:
        result = migrate_settings({"retry": {"maxDelayMs": 5000, "enabled": True}})
        assert result["retry"]["provider"]["maxRetryDelayMs"] == 5000
        assert "maxDelayMs" not in result["retry"]
        assert result["retry"]["enabled"] is True

    def test_no_migration_needed(self) -> None:
        original = {"steeringMode": "all", "transport": "sse"}
        result = migrate_settings(original)
        assert result == original


# --- Timeout parsing ---


class TestTimeoutParsing:
    def test_parse_http_idle_timeout_ms_valid(self) -> None:
        assert parse_http_idle_timeout_ms(30000) == 30000
        assert parse_http_idle_timeout_ms(0) == 0

    def test_parse_http_idle_timeout_ms_none(self) -> None:
        assert parse_http_idle_timeout_ms(None) is None

    def test_parse_http_idle_timeout_ms_invalid(self) -> None:
        assert parse_http_idle_timeout_ms(-1) is None
        assert parse_http_idle_timeout_ms("abc") is None
        assert parse_http_idle_timeout_ms(True) is None  # bool rejected

    def test_parse_timeout_setting_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_timeout_setting("abc", "testSetting")

    def test_parse_timeout_setting_none(self) -> None:
        assert parse_timeout_setting(None, "testSetting") is None

    def test_default_http_idle_timeout(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS


# --- InMemorySettingsStorage ---


class TestInMemoryStorage:
    def test_global_persistence(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("global", lambda _: json.dumps({"a": 1}))
        result = storage.with_lock("global", lambda c: c)
        assert result == json.dumps({"a": 1})

    def test_project_persistence(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("project", lambda _: json.dumps({"b": 2}))
        result = storage.with_lock("project", lambda c: c)
        assert result == json.dumps({"b": 2})

    def test_no_write_on_none(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("global", lambda _: json.dumps({"x": 1}))
        storage.with_lock("global", lambda c: None)  # no write
        assert storage.with_lock("global", lambda c: c) == json.dumps({"x": 1})


# --- SettingsManager in-memory ---


class TestSettingsManagerInMemory:
    def test_empty(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_default_provider() is None
        assert sm.get_default_model() is None
        assert sm.get_steering_mode() == "one-at-a-time"
        assert sm.get_compaction_enabled() is True

    def test_with_initial_data(self) -> None:
        sm = SettingsManager.in_memory({"defaultProvider": "anthropic", "defaultModel": "claude-3"})
        assert sm.get_default_provider() == "anthropic"
        assert sm.get_default_model() == "claude-3"

    def test_set_default_provider(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_default_provider("openai")
        assert sm.get_default_provider() == "openai"
        # Verify persisted to storage
        sm.reload()
        assert sm.get_default_provider() == "openai"

    def test_set_default_model_and_provider(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_default_model_and_provider("anthropic", "claude-4")
        assert sm.get_default_provider() == "anthropic"
        assert sm.get_default_model() == "claude-4"
        sm.reload()
        assert sm.get_default_provider() == "anthropic"
        assert sm.get_default_model() == "claude-4"

    def test_set_thinking_level(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_default_thinking_level("high")
        assert sm.get_default_thinking_level() == "high"
        sm.reload()
        assert sm.get_default_thinking_level() == "high"

    def test_set_transport(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_transport() == "auto"
        sm.set_transport("websocket")
        assert sm.get_transport() == "websocket"
        sm.reload()
        assert sm.get_transport() == "websocket"

    def test_compaction_settings(self) -> None:
        sm = SettingsManager.in_memory({"compaction": {"enabled": False, "reserveTokens": 5000}})
        assert sm.get_compaction_enabled() is False
        assert sm.get_compaction_reserve_tokens() == 5000
        assert sm.get_compaction_keep_recent_tokens() == 20000
        settings = sm.get_compaction_settings()
        assert settings == {"enabled": False, "reserveTokens": 5000, "keepRecentTokens": 20000}

    def test_set_compaction_enabled(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_compaction_enabled(False)
        assert sm.get_compaction_enabled() is False
        sm.reload()
        assert sm.get_compaction_enabled() is False

    def test_memory_settings_defaults(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_memory_enabled() is True
        assert sm.get_memory_top_k() == 3
        assert sm.get_memory_auto_capture() is True
        assert sm.get_memory_db_path().endswith("memory.db")

    def test_memory_settings_overrides(self) -> None:
        sm = SettingsManager.in_memory({"memory": {"enabled": False, "topK": 5, "dbPath": "/tmp/m.db"}})
        assert sm.get_memory_enabled() is False
        assert sm.get_memory_top_k() == 5
        assert sm.get_memory_db_path() == "/tmp/m.db"
        settings = sm.get_memory_settings()
        assert settings["enabled"] is False
        assert settings["topK"] == 5

    def test_set_memory_enabled(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_memory_enabled(False)
        assert sm.get_memory_enabled() is False
        sm.reload()
        assert sm.get_memory_enabled() is False

    def test_skill_paths(self) -> None:
        sm = SettingsManager.in_memory({"skills": ["/path/to/skills"]})
        assert sm.get_skill_paths() == ["/path/to/skills"]
        sm.set_skill_paths(["/new/path"])
        assert sm.get_skill_paths() == ["/new/path"]
        sm.reload()
        assert sm.get_skill_paths() == ["/new/path"]

    def test_extension_paths(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_extension_paths(["/ext1", "/ext2"])
        assert sm.get_extension_paths() == ["/ext1", "/ext2"]
        sm.reload()
        assert sm.get_extension_paths() == ["/ext1", "/ext2"]

    def test_enabled_models(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_enabled_models(["claude-*", "gpt-*"])
        assert sm.get_enabled_models() == ["claude-*", "gpt-*"]

    def test_theme(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_theme("dark")
        assert sm.get_theme() == "dark"
        # Themes with / are not returned by get_theme
        sm.set_theme("path/to/theme")
        assert sm.get_theme() is None
        assert sm.get_theme_setting() == "path/to/theme"

    def test_ui_mode(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_ui_mode() == "regular"
        sm.set_ui_mode("fullscreen")
        assert sm.get_ui_mode() == "fullscreen"

    def test_thinking_budgets(self) -> None:
        sm = SettingsManager.in_memory({"thinkingBudgets": {"minimal": 1000, "high": 8000}})
        budgets = sm.get_thinking_budgets()
        assert budgets is not None
        assert budgets.minimal == 1000
        assert budgets.high == 8000

    def test_terminal_settings(self) -> None:
        sm = SettingsManager.in_memory({"terminal": {"showImages": False, "imageWidthCells": 80}})
        assert sm.get_show_images() is False
        assert sm.get_image_width_cells() == 80
        sm.set_show_images(True)
        assert sm.get_show_images() is True
        sm.set_image_width_cells(100)
        assert sm.get_image_width_cells() == 100

    def test_image_settings(self) -> None:
        sm = SettingsManager.in_memory({"images": {"autoResize": False, "blockImages": True}})
        assert sm.get_image_auto_resize() is False
        assert sm.get_block_images() is True
        sm.set_image_auto_resize(True)
        assert sm.get_image_auto_resize() is True
        sm.set_block_images(False)
        assert sm.get_block_images() is False

    def test_apply_overrides(self) -> None:
        sm = SettingsManager.in_memory({"defaultProvider": "openai"})
        sm.apply_overrides({"defaultModel": "gpt-4"})
        assert sm.get_default_provider() == "openai"
        # defaultModel only in merged settings, not persisted
        assert sm.get_default_model() == "gpt-4"

    def test_drain_errors_empty(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.drain_errors() == []

    def test_is_project_trusted_default(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.is_project_trusted() is True

    def test_set_project_trusted_false(self) -> None:
        sm = SettingsManager.in_memory({"defaultProvider": "openai"})
        sm.set_project_trusted(False)
        assert sm.is_project_trusted() is False

    def test_analytics_tracking_id(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_enable_analytics(True)
        assert sm.get_enable_analytics() is True
        assert sm.get_tracking_id() is not None
        # Re-enabling shouldn't regenerate tracking ID
        first_id = sm.get_tracking_id()
        sm.set_enable_analytics(False)
        sm.set_enable_analytics(True)
        assert sm.get_tracking_id() == first_id

    def test_double_escape_action(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_double_escape_action() == "tree"
        sm.set_double_escape_action("fork")
        assert sm.get_double_escape_action() == "fork"

    def test_tree_filter_mode(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_tree_filter_mode() == "default"
        sm.set_tree_filter_mode("all")
        assert sm.get_tree_filter_mode() == "all"
        sm.set_tree_filter_mode("invalid")
        assert sm.get_tree_filter_mode() == "default"

    def test_editor_padding_x_clamped(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_editor_padding_x(10)
        assert sm.get_editor_padding_x() == 3
        sm.set_editor_padding_x(-5)
        assert sm.get_editor_padding_x() == 0

    def test_autocomplete_max_visible_clamped(self) -> None:
        sm = SettingsManager.in_memory()
        sm.set_autocomplete_max_visible(100)
        assert sm.get_autocomplete_max_visible() == 20
        sm.set_autocomplete_max_visible(1)
        assert sm.get_autocomplete_max_visible() == 3

    def test_code_block_indent(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_code_block_indent() == "  "
        sm = SettingsManager.in_memory({"markdown": {"codeBlockIndent": "    "}})
        assert sm.get_code_block_indent() == "    "

    def test_warnings(self) -> None:
        sm = SettingsManager.in_memory({"warnings": {"anthropicExtraUsage": False}})
        assert sm.get_warnings() == {"anthropicExtraUsage": False}
        sm.set_warnings({"anthropicExtraUsage": True})
        assert sm.get_warnings() == {"anthropicExtraUsage": True}

    def test_external_editor_command(self) -> None:
        sm = SettingsManager.in_memory({"externalEditor": "vim"})
        assert sm.get_external_editor_command() == "vim"

    def test_external_editor_command_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VISUAL", "code")
        monkeypatch.delenv("EDITOR", raising=False)
        sm = SettingsManager.in_memory()
        assert sm.get_external_editor_command() == "code"

    def test_quiet_startup(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_quiet_startup() is False
        sm.set_quiet_startup(True)
        assert sm.get_quiet_startup() is True

    def test_default_project_trust(self) -> None:
        sm = SettingsManager.in_memory()
        assert sm.get_default_project_trust() == "ask"
        sm.set_default_project_trust("always")
        assert sm.get_default_project_trust() == "always"

    def test_http_idle_timeout_ms(self) -> None:
        sm = SettingsManager.in_memory({"httpIdleTimeoutMs": 60000})
        assert sm.get_http_idle_timeout_ms() == 60000
        sm.set_http_idle_timeout_ms(0)
        assert sm.get_http_idle_timeout_ms() == 0

    def test_http_idle_timeout_ms_invalid(self) -> None:
        sm = SettingsManager.in_memory({"httpIdleTimeoutMs": "invalid"})
        # Falls back to default
        assert sm.get_http_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS

    def test_set_http_idle_timeout_ms_invalid(self) -> None:
        sm = SettingsManager.in_memory()
        with pytest.raises(ValueError):
            sm.set_http_idle_timeout_ms(-1)

    def test_project_settings_untrusted_not_loaded(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("project", lambda _: json.dumps({"defaultModel": "project-model"}))
        sm = SettingsManager.from_storage(storage, SettingsManagerCreateOptions(project_trusted=False))
        assert sm.is_project_trusted() is False
        # Project settings not loaded
        assert sm.get_default_model() is None

    def test_project_settings_trusted_loaded(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("global", lambda _: json.dumps({"defaultProvider": "global-provider"}))
        storage.with_lock("project", lambda _: json.dumps({"defaultModel": "project-model"}))
        sm = SettingsManager.from_storage(storage, SettingsManagerCreateOptions(project_trusted=True))
        assert sm.get_default_provider() == "global-provider"
        assert sm.get_default_model() == "project-model"

    def test_incremental_persistence_preserves_unrelated_keys(self) -> None:
        storage = InMemorySettingsStorage()
        storage.with_lock("global", lambda _: json.dumps({"unrelatedKey": "preserved", "defaultProvider": "old"}))
        sm = SettingsManager.from_storage(storage)
        sm.set_default_provider("new")
        # Reload from storage to verify
        sm2 = SettingsManager.from_storage(storage)
        assert sm2.get_default_provider() == "new"
        # The unrelated key should still be there
        assert sm2.get_global_settings().get("unrelatedKey") == "preserved"


# --- SettingsManager with file storage ---


class TestSettingsManagerFileStorage:
    def test_create_and_reload(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".pi"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        sm = SettingsManager.create(cwd, agent_dir)
        sm.set_default_provider("anthropic")
        sm.set_default_model("claude-4")

        # Create a new manager from the same paths
        sm2 = SettingsManager.create(cwd, agent_dir)
        assert sm2.get_default_provider() == "anthropic"
        assert sm2.get_default_model() == "claude-4"

    def test_project_settings(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".pi"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        sm = SettingsManager.create(cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=True))
        sm.set_project_skill_paths(["/project/skills"])
        assert sm.get_skill_paths() == ["/project/skills"]

        sm2 = SettingsManager.create(cwd, agent_dir)
        assert sm2.get_skill_paths() == ["/project/skills"]

    def test_project_untrusted_blocks_write(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".pi"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        sm = SettingsManager.create(cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=False))
        with pytest.raises(RuntimeError):
            sm.set_project_skill_paths(["/path"])

    def test_nested_field_persistence(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".pi"
        agent_dir.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()
        sm = SettingsManager.create(cwd, agent_dir)
        sm.set_compaction_enabled(False)
        sm.set_show_images(False)

        sm2 = SettingsManager.create(cwd, agent_dir)
        assert sm2.get_compaction_enabled() is False
        assert sm2.get_show_images() is False
