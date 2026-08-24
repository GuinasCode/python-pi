"""Tests for pi_runtime.context.ContextEngine. Covers Fase 2's acceptance
criteria from plan.md section 6:

- context assembly reproduz comportamento atual para casos básicos
- contexto irrelevante é removido
- decisões importantes sobrevivem à compactação
- constraints sobrevivem
- evidência citada não desaparece silenciosamente
"""

from __future__ import annotations

from pi_ai import UserMessage
from pi_runtime.context import ContextEngine, ContextItem
from pi_runtime.skills import SkillSelector
from pi_runtime.state import AgentState, Goal


class TestContextItem:
    def test_score_rewards_priority_over_relevance_and_freshness(self) -> None:
        high_priority = ContextItem(content="x", source="decision", priority=80, relevance=0.1, freshness=0.1)
        low_priority = ContextItem(content="y", source="conversation", priority=10, relevance=1.0, freshness=1.0)
        assert high_priority.score() > low_priority.score()


class TestCollectItems:
    def test_empty_state_and_messages_produce_no_items(self) -> None:
        engine = ContextEngine()
        state = AgentState()
        assert engine.collect_items(state, []) == []

    def test_goal_and_constraints_become_protected_items(self) -> None:
        engine = ContextEngine()
        goal = Goal(objective="ship the feature", constraints=["never touch prod directly"])
        state = AgentState(goal=goal)
        items = engine.collect_items(state, [])
        sources = {item.source for item in items}
        assert "goal" in sources
        assert "constraint" in sources

    def test_decisions_evidence_and_questions_become_items(self) -> None:
        engine = ContextEngine()
        state = AgentState(
            decisions=["use SQLite"],
            evidence=[{"claim": "sqlite is fast enough"}],
            unresolved_questions=["what about concurrent writers?"],
        )
        items = engine.collect_items(state, [])
        sources = {item.source for item in items}
        assert sources == {"decision", "evidence", "unresolved_question"}

    def test_blank_messages_are_skipped(self) -> None:
        engine = ContextEngine()
        state = AgentState()
        items = engine.collect_items(state, [UserMessage(content="   ")])
        assert items == []


class TestAssembleWorkingSet:
    def test_reproduces_current_behavior_for_the_basic_case(self) -> None:
        """No decisions/constraints/evidence accumulated yet — the
        working set is just the conversation, nothing invented."""
        engine = ContextEngine()
        state = AgentState()
        messages = [UserMessage(content="hello")]
        working_set = engine.assemble_working_set(state, messages)
        assert len(working_set) == 1
        assert working_set[0].source == "conversation"

    def test_protected_items_always_included_regardless_of_budget(self) -> None:
        engine = ContextEngine(budget_tokens=1)  # tiny budget
        goal = Goal(objective="x", constraints=["c1"])
        state = AgentState(goal=goal, decisions=["d1"], evidence=[{"e": 1}])
        working_set = engine.assemble_working_set(state, [])
        sources = {item.source for item in working_set}
        assert sources == {"goal", "constraint", "decision", "evidence"}

    def test_low_relevance_conversation_is_dropped_under_tight_budget(self) -> None:
        engine = ContextEngine(budget_tokens=1)
        state = AgentState()
        messages = [UserMessage(content="a fairly long message that costs several tokens to keep around")]
        working_set = engine.assemble_working_set(state, messages)
        assert working_set == []  # pure noise, no protected content to preserve

    def test_generous_budget_keeps_everything(self) -> None:
        engine = ContextEngine(budget_tokens=100_000)
        state = AgentState()
        messages = [UserMessage(content=f"message {i}") for i in range(5)]
        working_set = engine.assemble_working_set(state, messages)
        assert len(working_set) == 5


class TestRenderWorkingSetNote:
    def test_none_when_nothing_but_the_goal(self) -> None:
        engine = ContextEngine()
        state = AgentState(goal=Goal(objective="x"))
        assert engine.render_working_set_note(state, []) is None

    def test_none_for_a_completely_empty_state(self) -> None:
        engine = ContextEngine()
        assert engine.render_working_set_note(AgentState(), []) is None

    def test_surfaces_decisions_constraints_and_evidence(self) -> None:
        engine = ContextEngine()
        goal = Goal(objective="x", constraints=["never delete without confirmation"])
        state = AgentState(goal=goal, decisions=["chose SQLite over Postgres"], evidence=[{"source": "benchmark.md"}])
        note = engine.render_working_set_note(state, [])
        assert note is not None
        assert "never delete without confirmation" in note
        assert "chose SQLite over Postgres" in note
        assert "benchmark.md" in note
        assert "GOAL:" not in note  # the bare goal itself isn't re-surfaced, only protected extras

    def test_unresolved_questions_are_not_lost(self) -> None:
        engine = ContextEngine()
        state = AgentState(unresolved_questions=["does this handle concurrent writers?"])
        note = engine.render_working_set_note(state, [])
        assert note is not None
        assert "concurrent writers" in note


class TestContextEngineSkillIntegration:
    """Fase 9's real consumer: skills passed to collect_items()/
    assemble_working_set() are selected via SkillSelector, not injected
    wholesale — this is what "skills são carregadas sob demanda" and
    "skills irrelevantes não entram no contexto" mean at the Context
    Engine level."""

    def test_no_skills_argument_behaves_exactly_like_before(self) -> None:
        engine = ContextEngine()
        state = AgentState(goal=Goal(objective="write a database migration"))
        items = engine.collect_items(state, [])
        assert not any(item.source == "skill" for item in items)

    def test_relevant_skill_becomes_a_context_item(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Skill:
            name: str
            description: str

        engine = ContextEngine()
        state = AgentState(goal=Goal(objective="write a database migration"))
        skills = [_Skill(name="database-migrations", description="safe SQL migration guidance")]

        items = engine.collect_items(state, [], skills)
        skill_items = [item for item in items if item.source == "skill"]
        assert len(skill_items) == 1
        assert "database-migrations" in skill_items[0].content

    def test_irrelevant_skill_never_becomes_a_context_item(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Skill:
            name: str
            description: str

        engine = ContextEngine(skill_selector=SkillSelector(top_k=1, min_score=0.3))
        state = AgentState(goal=Goal(objective="write a database migration"))
        skills = [
            _Skill(name="database-migrations", description="safe SQL migration guidance"),
            _Skill(name="poetry-writing", description="how to write a sonnet"),
        ]

        items = engine.collect_items(state, [], skills)
        skill_items = [item for item in items if item.source == "skill"]
        assert len(skill_items) == 1
        assert "database-migrations" in skill_items[0].content

    def test_skills_survive_into_the_working_set(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Skill:
            name: str
            description: str

        engine = ContextEngine()
        state = AgentState(goal=Goal(objective="write a database migration"))
        skills = [_Skill(name="database-migrations", description="safe SQL migration guidance")]

        working_set = engine.assemble_working_set(state, [], skills)
        assert any(item.source == "skill" for item in working_set)
