"""Tests for parenthetical_asides conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.parenthetical_asides import (
    _score_parenthetical_density,
    _score_no_parentheses,
    ParentheticalAsidesConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("parenthetical_asides")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "parenthetical_asides"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_arg_keys_empty(self, conflict):
        assert conflict.arg_keys == [] or conflict.get_instruction_args_keys() == []

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b")
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_templates_no_placeholders(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_threshold_set(self, conflict):
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_verify_fns_are_float(self, conflict):
        result_sys = conflict.score_system("Hello world", direction="a")
        result_usr = conflict.score_user("Hello world", direction="a")
        assert isinstance(result_sys, float)
        assert isinstance(result_usr, float)

    def test_inverted_flag(self):
        assert getattr(_score_no_parentheses, "is_inverted", False) is True
        assert not getattr(_score_parenthetical_density, "is_inverted", False)


# -- Scorer: _score_parenthetical_density --


class TestScoreParentheticalDensity:
    def test_multiple_asides(self):
        text = "Machine learning (ML) is a subset of artificial intelligence (AI)."
        score = _score_parenthetical_density(text)
        assert score > 0.5  # 2 parens in 1 sentence = high density

    def test_single_aside_one_sentence(self):
        text = "Machine learning (ML) is a powerful tool."
        score = _score_parenthetical_density(text)
        assert score == pytest.approx(1.0)  # 1 match / 1 sentence = 1.0

    def test_single_aside_two_sentences(self):
        text = "Machine learning (ML) is a powerful tool. It works well."
        score = _score_parenthetical_density(text)
        assert score == pytest.approx(0.5)  # 1 match / 2 sentences

    def test_no_parens(self):
        text = "Machine learning is a powerful tool for data analysis."
        score = _score_parenthetical_density(text)
        assert score == 0.0

    def test_numbered_references(self):
        text = "The water cycle is continuous (1). It sustains life (2). Evaporation is key (3)."
        score = _score_parenthetical_density(text)
        assert score == pytest.approx(1.0)  # 3 matches / 3 sentences

    def test_long_asides(self):
        text = (
            "Think of it (like a child learning to recognize objects) as a way "
            "to train algorithms (using large datasets of examples)."
        )
        score = _score_parenthetical_density(text)
        assert score > 0.5

    def test_empty_parens_not_counted(self):
        text = "Something () and another () here."
        score = _score_parenthetical_density(text)
        assert score == 0.0

    def test_whitespace_only_parens_not_counted(self):
        text = "Something (  ) and another (  ) here."
        score = _score_parenthetical_density(text)
        assert score == 0.0


# -- Scorer: _score_no_parentheses --


class TestScoreNoParentheses:
    def test_no_parens(self):
        text = "This text has no parentheses at all."
        assert _score_no_parentheses(text) == 1.0

    def test_heavy_parens(self):
        text = "Machine learning (ML) is a subset of artificial intelligence (AI)."
        assert _score_no_parentheses(text) < 0.5

    def test_is_complement(self):
        text = "Something (here) and more (there). Another sentence."
        density = _score_parenthetical_density(text)
        no_parens = _score_no_parentheses(text)
        assert density + no_parens == pytest.approx(1.0)


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_parenthetical_density("") == 0.0
        assert _score_no_parentheses("") == 1.0

    def test_very_short_response(self):
        assert _score_parenthetical_density("ok") == 0.0
        assert _score_no_parentheses("ok") == 1.0

    def test_threshold_integration_direction_a(self):
        """Integration: verify_followed_system with direction='a' (system=asides)."""
        c = ParentheticalAsidesConflict()
        with_asides = "Machine learning (ML) is a subset of AI (artificial intelligence)."
        assert c.verify_followed_system(with_asides, direction="a")
        without = "Machine learning is a subset of AI."
        assert not c.verify_followed_system(without, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration: verify_followed_system with direction='b' (system=no parens)."""
        c = ParentheticalAsidesConflict()
        clean = "Machine learning is a powerful tool for data analysis."
        assert c.verify_followed_system(clean, direction="b")
        with_parens = "Machine learning (ML) is great (really great)."
        assert not c.verify_followed_system(with_parens, direction="b")

    def test_nested_parens_ignored(self):
        """Nested parens are not common but should not crash."""
        text = "Something ((nested)) and another (thing) here (also)."
        score = _score_parenthetical_density(text)
        assert score > 0.0
