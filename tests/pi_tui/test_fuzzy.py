from __future__ import annotations

from pi_tui.fuzzy import FuzzyMatch, fuzzy_filter, fuzzy_match


class TestFuzzyMatch:
    def test_empty_query_matches_everything_with_score_0(self) -> None:
        result = fuzzy_match("", "anything")
        assert result.matches is True
        assert result.score == 0

    def test_query_longer_than_text_does_not_match(self) -> None:
        result = fuzzy_match("longquery", "short")
        assert result.matches is False

    def test_exact_match_has_good_score(self) -> None:
        result = fuzzy_match("test", "test")
        assert result.matches is True
        assert result.score < 0  # Should be negative due to consecutive bonuses

    def test_characters_must_appear_in_order(self) -> None:
        match_in_order = fuzzy_match("abc", "aXbXc")
        assert match_in_order.matches is True

        match_out_of_order = fuzzy_match("abc", "cba")
        assert match_out_of_order.matches is False

    def test_case_insensitive_matching(self) -> None:
        result = fuzzy_match("ABC", "abc")
        assert result.matches is True

        result2 = fuzzy_match("abc", "ABC")
        assert result2.matches is True

    def test_consecutive_matches_score_better_than_scattered_matches(self) -> None:
        consecutive = fuzzy_match("foo", "foobar")
        scattered = fuzzy_match("foo", "f_o_o_bar")

        assert consecutive.matches is True
        assert scattered.matches is True
        assert consecutive.score < scattered.score

    def test_word_boundary_matches_score_better(self) -> None:
        at_boundary = fuzzy_match("fb", "foo-bar")
        not_at_boundary = fuzzy_match("fb", "afbx")

        assert at_boundary.matches is True
        assert not_at_boundary.matches is True
        assert at_boundary.score < not_at_boundary.score

    def test_matches_swapped_alpha_numeric_tokens(self) -> None:
        result = fuzzy_match("codex52", "gpt-5.2-codex")
        assert result.matches is True


class TestFuzzyFilter:
    def test_empty_query_returns_all_items_unchanged(self) -> None:
        items = ["apple", "banana", "cherry"]
        result = fuzzy_filter(items, "", lambda x: x)
        assert result == items

    def test_filters_out_non_matching_items(self) -> None:
        items = ["apple", "banana", "cherry"]
        result = fuzzy_filter(items, "an", lambda x: x)
        assert "banana" in result
        assert "apple" not in result
        assert "cherry" not in result

    def test_sorts_results_by_match_quality(self) -> None:
        items = ["a_p_p", "app", "application"]
        result = fuzzy_filter(items, "app", lambda x: x)

        # "app" should be first (exact consecutive match at start)
        assert result[0] == "app"

    def test_prioritizes_exact_matches_over_longer_prefix_matches(self) -> None:
        items = ["clone", "cl"]
        result = fuzzy_filter(items, "cl", lambda x: x)

        assert result == ["cl", "clone"]

    def test_works_with_custom_get_text_function(self) -> None:
        items = [
            {"name": "foo", "id": 1},
            {"name": "bar", "id": 2},
            {"name": "foobar", "id": 3},
        ]
        result = fuzzy_filter(items, "foo", lambda item: item["name"])

        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "foo" in names
        assert "foobar" in names

    def test_matches_slash_separated_provider_model_queries_against_reordered_text(self) -> None:
        item = {"id": "gpt-5.5", "provider": "openai-codex"}
        result = fuzzy_filter([item], "openai-codex/gpt-5.5", lambda model: f"{model['id']} {model['provider']}")

        assert result == [item]


def test_fuzzy_match_dataclass_shape() -> None:
    m = fuzzy_match("a", "a")
    assert isinstance(m, FuzzyMatch)
    assert m.matches is True
    assert isinstance(m.score, float)
