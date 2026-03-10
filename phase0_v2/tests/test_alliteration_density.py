"""Tests for alliteration_density conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.alliteration_density import (
    _score_alliteration,
    _score_no_alliteration,
    AlliterationDensityConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("alliteration_density")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "alliteration_density"

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
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The no-alliteration scorer should have is_inverted=True."""
        assert getattr(_score_no_alliteration, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(_score_alliteration, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreAlliteration:
    def test_heavy_alliteration(self):
        """Text with many alliterative pairs should score high."""
        text = (
            "Beautiful blue butterflies bring bright blossoms. "
            "Perfect pristine paradise provides peaceful pleasure."
        )
        score = _score_alliteration(text)
        assert score >= 0.5

    def test_no_alliteration(self):
        """Text carefully avoiding alliteration should score very low."""
        text = (
            "The cat ran over a big yellow fence. "
            "She put down her old worn basket."
        )
        score = _score_alliteration(text)
        assert score <= 0.15

    def test_mixed_alliteration(self):
        """Some alliterative phrases mixed with normal text."""
        text = (
            "The big brown bear walked through the forest. "
            "Clever cats climb carefully up tall trees."
        )
        score = _score_alliteration(text)
        assert 0.1 <= score <= 0.5

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "Beautiful blue birds bring bright blossoms to the garden."
        direct = _score_alliteration(text)
        inverted = _score_no_alliteration(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "big blue bird", "The cat sat on the mat."]:
            score = _score_alliteration(text)
            assert 0.0 <= score <= 1.0

    def test_case_insensitive(self):
        """Alliteration should be detected regardless of case."""
        text = "BEAUTIFUL Blue BIRDS bring BRIGHT blossoms."
        score = _score_alliteration(text)
        assert score >= 0.5


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_alliteration("") == 0.0
        assert _score_no_alliteration("") == 1.0

    def test_single_word(self):
        assert _score_alliteration("hello") == 0.0

    def test_two_words_alliterative(self):
        score = _score_alliteration("big blue")
        assert score == 1.0

    def test_two_words_non_alliterative(self):
        score = _score_alliteration("big cat")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = AlliterationDensityConflict()
        # Text with heavy alliteration should pass system check in direction a
        alliterative = (
            "Beautiful blue butterflies bring bright blossoms. "
            "Perfect pristine paradise provides peaceful pleasure."
        )
        assert c.verify_followed_system(alliterative, direction="a")
        # Normal text should fail system check in direction a
        normal = "The cat sat on the mat. A dog ran in the yard."
        assert not c.verify_followed_system(normal, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = AlliterationDensityConflict()
        # Normal text should pass system check in direction b (system=avoid alliteration)
        normal = "The cat sat on the mat. A dog ran in the yard."
        assert c.verify_followed_system(normal, direction="b")
        # Heavily alliterative text should fail system check in direction b
        alliterative = (
            "Beautiful blue butterflies bring bright blossoms. "
            "Perfect pristine paradise provides peaceful pleasure."
        )
        assert not c.verify_followed_system(alliterative, direction="b")

    def test_punctuation_stripped(self):
        """Words with punctuation should still be compared by first letter."""
        text = '"beautiful" blue birds'
        score = _score_alliteration(text)
        assert score >= 0.5
