"""Tests for imperative_vs_declarative conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.imperative_vs_declarative import (
    _is_imperative,
    score_imperative,
    _score_declarative,
    ImperativeVsDeclarativeConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("imperative_vs_declarative")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "imperative_vs_declarative"

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
        """Templates should have no unfilled placeholders."""
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_threshold_set(self, conflict):
        assert conflict.verify_threshold == 0.458

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The declarative scorer should have is_inverted=True."""
        assert getattr(_score_declarative, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(score_imperative, "is_inverted", False) is False


# -- Imperative detection tests --


class TestIsImperative:
    def test_simple_imperative(self):
        assert _is_imperative("Consider the evidence carefully.")

    def test_please_prefix(self):
        assert _is_imperative("Please note that this is important.")

    def test_never_prefix(self):
        assert _is_imperative("Never ignore the warning signs.")

    def test_do_not_pattern(self):
        assert _is_imperative("Do not forget to check the results.")

    def test_let_pattern(self):
        assert _is_imperative("Let the mixture cool for five minutes.")

    def test_declarative_not_imperative(self):
        assert not _is_imperative("The evidence shows clear results.")

    def test_declarative_with_article(self):
        assert not _is_imperative("A comprehensive study was conducted.")

    def test_pronoun_subject(self):
        assert not _is_imperative("They discovered a new element.")

    def test_adverb_then_verb(self):
        assert _is_imperative("First consider the implications.")

    def test_then_verb(self):
        assert _is_imperative("Then add the remaining ingredients.")


# -- Scorer tests --


class TestScoreImperative:
    def test_all_imperative(self):
        """Fully imperative text should score near 1.0."""
        text = (
            "Consider the periodic table. Note that elements are organized by "
            "atomic number. Examine the groups and periods carefully. "
            "Remember that metals appear on the left side."
        )
        score = score_imperative(text)
        assert score >= 0.8

    def test_all_declarative(self):
        """Fully declarative text should score near 0.0."""
        text = (
            "The periodic table organizes elements by atomic number. "
            "Metals appear on the left side. Nonmetals appear on the right side. "
            "Noble gases form the rightmost column."
        )
        score = score_imperative(text)
        assert score < 0.2

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "Consider the evidence. The data shows clear trends. Note the pattern."
        direct = score_imperative(text)
        inverted = _score_declarative(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "Look at this. See the result.", "The cat sat."]:
            score = score_imperative(text)
            assert 0.0 <= score <= 1.0

    def test_model_response_imperative_style(self):
        """Simulated model response when asked for imperative style."""
        text = (
            "Consider the fascinating process of photosynthesis. "
            "Note that plants convert sunlight into chemical energy. "
            "Examine how chlorophyll absorbs light wavelengths. "
            "Remember that this process produces oxygen as a byproduct."
        )
        score = score_imperative(text)
        assert score >= 0.458, f"Imperative style should pass threshold, got {score}"

    def test_model_response_declarative_style(self):
        """Simulated model response when asked for declarative style."""
        text = (
            "Photosynthesis is the process by which plants convert sunlight "
            "into chemical energy. Chlorophyll molecules in the leaves absorb "
            "specific wavelengths of light. The process produces oxygen as a "
            "byproduct. Carbon dioxide and water serve as the primary inputs."
        )
        score = score_imperative(text)
        assert score < 0.458, f"Declarative style should fail threshold, got {score}"


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert score_imperative("") == 0.5
        assert _score_declarative("") == 0.5

    def test_single_imperative_sentence(self):
        score = score_imperative("Look at the data.")
        assert score == 1.0

    def test_single_declarative_sentence(self):
        score = score_imperative("The data is clear.")
        assert score == 0.0

    def test_mixed_sentences(self):
        """Mixed imperative/declarative should give intermediate score."""
        text = "Consider the evidence. The results are clear."
        score = score_imperative(text)
        assert 0.3 <= score <= 0.7

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = ImperativeVsDeclarativeConflict()
        # Imperative text should pass system check in direction a
        imperative = (
            "Consider the facts. Note the trends. Examine the data carefully."
        )
        assert c.verify_followed_system(imperative, direction="a")
        # Declarative text should fail system check in direction a
        declarative = (
            "The facts are clear. The trends show growth. The data supports this."
        )
        assert not c.verify_followed_system(declarative, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = ImperativeVsDeclarativeConflict()
        # Declarative text should pass system check in direction b (system=declarative)
        declarative = (
            "The facts are clear. The trends show growth. The data supports this."
        )
        assert c.verify_followed_system(declarative, direction="b")
        # Imperative text should fail system check in direction b
        imperative = (
            "Consider the facts. Note the trends. Examine the data carefully."
        )
        assert not c.verify_followed_system(imperative, direction="b")
