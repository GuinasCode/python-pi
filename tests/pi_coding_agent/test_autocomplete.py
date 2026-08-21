"""Tests for pi_coding_agent.autocomplete.command_suggestions."""

from __future__ import annotations

from pi_coding_agent.autocomplete import command_suggestions


class TestCommandSuggestions:
    def test_matches_builtin_prefix(self) -> None:
        assert "/model" in command_suggestions("/mo", [])

    def test_no_match_returns_empty(self) -> None:
        assert command_suggestions("/zzz", []) == []

    def test_not_a_slash_command_returns_empty(self) -> None:
        assert command_suggestions("hello", []) == []

    def test_a_space_means_arguments_started_no_more_completion(self) -> None:
        assert command_suggestions("/model ", []) == []

    def test_exact_match_offers_nothing_left_to_complete(self) -> None:
        assert command_suggestions("/model", []) == []

    def test_includes_extension_commands(self) -> None:
        assert command_suggestions("/gr", ["greet"]) == ["/greet"]

    def test_bare_slash_lists_every_command(self) -> None:
        matches = command_suggestions("/", ["greet"])
        assert "/help" in matches
        assert "/greet" in matches
