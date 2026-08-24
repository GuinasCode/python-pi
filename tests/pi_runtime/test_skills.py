"""Tests for pi_runtime.skills. Covers Fase 9's acceptance criteria from
plan.md section 13:

- skills são carregadas sob demanda (only selected skills become
  ContextItems — see TestContextEngineSkillIntegration in
  test_context.py for the real end-to-end proof)
- skills irrelevantes não entram no contexto
- selection é observável
- usage e success rate são registrados
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_runtime.skills import SkillSelector, SkillUsageTracker


@dataclass
class _FakeSkill:
    name: str
    description: str


class TestSkillSelector:
    def test_relevant_skill_scores_higher_than_irrelevant_one(self) -> None:
        selector = SkillSelector()
        relevant = _FakeSkill(name="database-migrations", description="how to write safe SQL migrations")
        irrelevant = _FakeSkill(name="poetry-writing", description="how to write a sonnet")

        relevant_score = selector.score("write a database migration", relevant)
        irrelevant_score = selector.score("write a database migration", irrelevant)

        assert relevant_score > irrelevant_score

    def test_select_marks_only_top_k_as_selected(self) -> None:
        """Fase 9 acceptance criterion: irrelevant skills don't enter the
        context — only the top_k highest-scoring candidates above
        min_score are marked selected."""
        selector = SkillSelector(top_k=1, min_score=0.0)
        skills = [
            _FakeSkill(name="database-migrations", description="safe SQL migrations"),
            _FakeSkill(name="poetry-writing", description="write a sonnet"),
        ]
        selections = selector.select("write a database migration safely", skills)

        selected = [s for s in selections if s.selected]
        assert len(selected) == 1
        assert selected[0].name == "database-migrations"

    def test_selection_is_observable_every_candidate_is_scored(self) -> None:
        """Fase 9 acceptance criterion: selection é observável — the full
        ranking is returned, not just the winners."""
        selector = SkillSelector(top_k=1)
        skills = [
            _FakeSkill(name="a", description="alpha"),
            _FakeSkill(name="b", description="beta"),
            _FakeSkill(name="c", description="gamma"),
        ]
        selections = selector.select("alpha task", skills)
        assert len(selections) == 3
        assert {s.name for s in selections} == {"a", "b", "c"}

    def test_no_skills_returns_empty_list(self) -> None:
        selector = SkillSelector()
        assert selector.select("anything", []) == []

    def test_empty_query_scores_zero_for_everything(self) -> None:
        selector = SkillSelector()
        skill = _FakeSkill(name="x", description="y")
        assert selector.score("", skill) == 0.0

    def test_min_score_excludes_weak_matches_even_within_top_k(self) -> None:
        selector = SkillSelector(top_k=5, min_score=0.5)
        skills = [_FakeSkill(name="totally-unrelated", description="nothing in common")]
        selections = selector.select("database migration", skills)
        assert not selections[0].selected


class TestSkillUsageTracker:
    def test_usage_is_recorded_on_selection(self) -> None:
        from pi_runtime.skills import SkillSelection

        tracker = SkillUsageTracker()
        tracker.record_selection(
            "query",
            [SkillSelection(name="a", description="", score=0.9, selected=True)],
        )
        assert tracker.usage_count("a") == 1

    def test_unselected_candidates_are_not_recorded_as_usage(self) -> None:
        from pi_runtime.skills import SkillSelection

        tracker = SkillUsageTracker()
        tracker.record_selection(
            "query",
            [SkillSelection(name="a", description="", score=0.1, selected=False)],
        )
        assert tracker.usage_count("a") == 0

    def test_success_rate_reflects_recorded_outcomes(self) -> None:
        from pi_runtime.skills import SkillSelection

        tracker = SkillUsageTracker()
        for _ in range(3):
            tracker.record_selection("q", [SkillSelection(name="a", description="", score=1.0, selected=True)])
        outcomes = [True, True, False]
        # attach outcomes to the 3 recorded (still-open) usages, in order
        for outcome in outcomes:
            assert tracker.record_outcome("a", succeeded=outcome) is True

        assert tracker.success_rate("a") == 2 / 3

    def test_success_rate_is_none_with_no_recorded_outcomes(self) -> None:
        tracker = SkillUsageTracker()
        assert tracker.success_rate("never-used") is None

    def test_record_outcome_for_unknown_skill_returns_false_not_a_crash(self) -> None:
        tracker = SkillUsageTracker()
        assert tracker.record_outcome("never-selected", succeeded=True) is False

    def test_all_records_reflects_full_history(self) -> None:
        from pi_runtime.skills import SkillSelection

        tracker = SkillUsageTracker()
        tracker.record_selection("q1", [SkillSelection(name="a", description="", score=1.0, selected=True)])
        tracker.record_selection("q2", [SkillSelection(name="b", description="", score=1.0, selected=True)])
        assert len(tracker.all_records()) == 2
