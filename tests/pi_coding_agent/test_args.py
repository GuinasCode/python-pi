"""Tests for pi_coding_agent CLI argument parsing."""

from __future__ import annotations

from pi_coding_agent import Args, parse_args


class TestParseArgs:
    def test_help_flag(self) -> None:
        result = parse_args(["--help"])
        assert result.help is True

    def test_help_short(self) -> None:
        result = parse_args(["-h"])
        assert result.help is True

    def test_version_flag(self) -> None:
        result = parse_args(["--version"])
        assert result.version is True

    def test_version_short(self) -> None:
        result = parse_args(["-v"])
        assert result.version is True

    def test_print_flag(self) -> None:
        result = parse_args(["--print", "hello world"])
        assert result.print is True
        assert result.messages == ["hello world"]

    def test_print_short(self) -> None:
        result = parse_args(["-p", "hello"])
        assert result.print is True
        assert result.messages == ["hello"]

    def test_provider(self) -> None:
        result = parse_args(["--provider", "anthropic"])
        assert result.provider == "anthropic"

    def test_model(self) -> None:
        result = parse_args(["--model", "claude-sonnet-4"])
        assert result.model == "claude-sonnet-4"

    def test_api_key(self) -> None:
        result = parse_args(["--api-key", "secret123"])
        assert result.api_key == "secret123"

    def test_thinking_valid(self) -> None:
        result = parse_args(["--thinking", "high"])
        assert result.thinking == "high"

    def test_thinking_invalid(self) -> None:
        result = parse_args(["--thinking", "ultra"])
        assert result.thinking is None
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0]["type"] == "warning"

    def test_continue(self) -> None:
        result = parse_args(["--continue"])
        assert result.continue_session is True

    def test_continue_short(self) -> None:
        result = parse_args(["-c"])
        assert result.continue_session is True

    def test_resume(self) -> None:
        result = parse_args(["--resume"])
        assert result.resume is True

    def test_name(self) -> None:
        result = parse_args(["--name", "my-session"])
        assert result.name == "my-session"

    def test_no_session(self) -> None:
        result = parse_args(["--no-session"])
        assert result.no_session is True

    def test_session_id(self) -> None:
        result = parse_args(["--session-id", "abc123"])
        assert result.session_id == "abc123"

    def test_session(self) -> None:
        result = parse_args(["--session", "abc123"])
        assert result.session == "abc123"

    def test_tools(self) -> None:
        result = parse_args(["--tools", "read,write,bash"])
        assert result.tools == ["read", "write", "bash"]

    def test_exclude_tools(self) -> None:
        result = parse_args(["--exclude-tools", "bash,grep"])
        assert result.exclude_tools == ["bash", "grep"]

    def test_no_tools(self) -> None:
        result = parse_args(["--no-tools"])
        assert result.no_tools is True

    def test_extension(self) -> None:
        result = parse_args(["--extension", "my-ext"])
        assert result.extensions == ["my-ext"]

    def test_multiple_extensions(self) -> None:
        result = parse_args(["-e", "ext1", "-e", "ext2"])
        assert result.extensions == ["ext1", "ext2"]

    def test_skill(self) -> None:
        result = parse_args(["--skill", "python"])
        assert result.skills == ["python"]

    def test_list_models(self) -> None:
        result = parse_args(["--list-models"])
        assert result.list_models is True

    def test_list_models_with_pattern(self) -> None:
        result = parse_args(["--list-models", "claude"])
        assert result.list_models == "claude"

    def test_ui_mode(self) -> None:
        result = parse_args(["--ui-mode", "fullscreen"])
        assert result.ui_mode == "fullscreen"

    def test_alt_alias(self) -> None:
        result = parse_args(["--alt"])
        assert result.ui_mode == "fullscreen"

    def test_verbose(self) -> None:
        result = parse_args(["--verbose"])
        assert result.verbose is True

    def test_offline(self) -> None:
        result = parse_args(["--offline"])
        assert result.offline is True

    def test_file_arg(self) -> None:
        result = parse_args(["@myfile.txt"])
        assert result.file_args == ["myfile.txt"]

    def test_plain_message(self) -> None:
        result = parse_args(["hello world"])
        assert result.messages == ["hello world"]

    def test_unknown_flag(self) -> None:
        result = parse_args(["--custom-flag"])
        assert result.unknown_flags == {"custom-flag": True}

    def test_unknown_flag_with_value(self) -> None:
        result = parse_args(["--custom-flag=value"])
        assert result.unknown_flags == {"custom-flag": "value"}

    def test_mode_text(self) -> None:
        result = parse_args(["--mode", "text"])
        assert result.mode == "text"

    def test_mode_json(self) -> None:
        result = parse_args(["--mode", "json"])
        assert result.mode == "json"

    def test_mode_invalid(self) -> None:
        result = parse_args(["--mode", "invalid"])
        assert result.mode is None

    def test_export(self) -> None:
        result = parse_args(["--export", "output.html"])
        assert result.export == "output.html"

    def test_no_skills(self) -> None:
        result = parse_args(["--no-skills"])
        assert result.no_skills is True

    def test_no_extensions(self) -> None:
        result = parse_args(["--no-extensions"])
        assert result.no_extensions is True

    def test_no_context_files(self) -> None:
        result = parse_args(["--no-context-files"])
        assert result.no_context_files is True

    def test_empty_args(self) -> None:
        result = parse_args([])
        assert result.help is False
        assert result.version is False
        assert result.messages == []
        assert result.file_args == []

    def test_default_args(self) -> None:
        args = Args()
        assert args.provider is None
        assert args.model is None
        assert args.messages == []
        assert args.unknown_flags == {}
        assert args.diagnostics == []
