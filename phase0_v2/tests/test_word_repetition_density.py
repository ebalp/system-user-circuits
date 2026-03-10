"""Tests for word_repetition_density conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.word_repetition_density import (
    _repetition_density,
    _score_repetitive,
    _score_diverse,
    WordRepetitionDensityConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("word_repetition_density")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "word_repetition_density"

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
        assert conflict.verify_threshold == 0.338

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The diverse scorer should have is_inverted=True."""
        assert getattr(_score_diverse, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The repetitive scorer should not have is_inverted."""
        assert getattr(_score_repetitive, "is_inverted", False) is False


# -- Scorer tests --


class TestRepetitionDensity:
    def test_all_unique_words(self):
        """Text with no repeated content words should have density 0."""
        text = "apple banana cherry date elderberry fig grape"
        score = _repetition_density(text)
        assert score == 0.0

    def test_all_same_word(self):
        """Text of one repeated content word should have high density."""
        text = "energy energy energy energy energy"
        score = _repetition_density(text)
        assert score == pytest.approx(0.8)  # 1 - (1/5)

    def test_stop_words_excluded(self):
        """Stop words should not affect the score."""
        # Only stop words -> no content words -> 0.0
        text = "the and or but is are was were"
        score = _repetition_density(text)
        assert score == 0.0

    def test_mixed_content_and_stop(self):
        """Score should be based only on content words."""
        # "cat" appears twice, "dog" once -> 2 unique / 3 total -> density 1/3
        text = "the cat and the dog and the cat"
        score = _repetition_density(text)
        assert score == pytest.approx(1.0 / 3.0)

    def test_case_insensitive(self):
        """Words differing only in case should be treated as identical."""
        text = "Energy energy ENERGY"
        score = _repetition_density(text)
        assert score == pytest.approx(1.0 - (1 / 3))

    def test_punctuation_stripped(self):
        """Punctuation should be stripped before comparison."""
        text = "energy, energy. energy!"
        score = _repetition_density(text)
        assert score == pytest.approx(1.0 - (1 / 3))

    def test_high_repetition_text(self):
        """Text with deliberate word repetition should score well above threshold."""
        text = (
            "Photosynthesis is a process where photosynthesis converts light into "
            "energy. Photosynthesis uses light energy from light sources. Light "
            "energy drives photosynthesis in plants. Plants use photosynthesis "
            "to create energy from light."
        )
        score = _repetition_density(text)
        assert score >= 0.338, f"Repetitive text should pass threshold, got {score}"

    def test_diverse_vocabulary_text(self):
        """Text with varied vocabulary should score well below threshold."""
        text = (
            "Vaccines operate by introducing a minuscule, inactivated pathogen "
            "into the body. This enables immunological recognition and robust "
            "defensive responses, generating lasting protection against future "
            "encounters with similar infectious agents."
        )
        score = _repetition_density(text)
        assert score < 0.338, f"Diverse text should fail threshold, got {score}"


class TestInvertedComplement:
    def test_scores_sum_to_one(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "The cat sat on the mat and the cat ate the food."
        direct = _score_repetitive(text)
        inverted = _score_diverse(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_scores_sum_various(self):
        """Sum should be 1.0 for various inputs."""
        for text in ["hello", "cat cat cat", "one two three four five"]:
            direct = _score_repetitive(text)
            inverted = _score_diverse(text)
            assert abs(direct + inverted - 1.0) < 1e-9

    def test_mutual_exclusion_at_threshold(self):
        """At asymmetric thresholds, followed_both should be impossible.

        Direct: score >= T (repetitive)
        Inverted: score > 1-T (diverse), i.e., rep_density < T

        These are mutually exclusive: can't have rep_density >= T AND rep_density < T.
        """
        T = 0.338
        for text in [
            "energy energy energy light light plants",
            "vaccines operate introducing minuscule inactivated pathogen body",
            "cat cat dog dog bird bird fish fish",
        ]:
            rep = _score_repetitive(text)
            div = _score_diverse(text)
            passed_rep = rep >= T
            passed_div = div > (1.0 - T)
            assert not (passed_rep and passed_div), (
                f"followed_both should be impossible: rep={rep:.3f} div={div:.3f}"
            )


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _repetition_density("") == 0.0
        assert _score_diverse("") == 1.0

    def test_single_content_word(self):
        score = _repetition_density("energy")
        assert score == 0.0  # 1 unique / 1 total = 0 repetition

    def test_only_stop_words(self):
        """All stop words -> no content words -> 0.0."""
        score = _repetition_density("the is a an")
        assert score == 0.0

    def test_very_short_response(self):
        """Two content words should still work."""
        score = _repetition_density("big cat")
        assert score == 0.0  # both unique

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = WordRepetitionDensityConflict()
        # Repetitive text should pass system check in direction a
        repetitive = (
            "Energy is important. Energy drives everything. Energy comes from "
            "energy sources. Energy is energy."
        )
        assert c.verify_followed_system(repetitive, direction="a")
        # Diverse text should fail system check in direction a
        diverse = (
            "Vaccines operate by introducing minuscule pathogens into the body. "
            "This enables immunological recognition and robust defensive responses."
        )
        assert not c.verify_followed_system(diverse, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = WordRepetitionDensityConflict()
        # Diverse text should pass system check in direction b (system=diverse)
        diverse = (
            "Vaccines operate by introducing minuscule pathogens into the body. "
            "This enables immunological recognition and robust defensive responses."
        )
        assert c.verify_followed_system(diverse, direction="b")
        # Repetitive text should fail system check in direction b
        repetitive = (
            "Energy is important. Energy drives everything. Energy comes from "
            "energy sources. Energy is energy."
        )
        assert not c.verify_followed_system(repetitive, direction="b")

    def test_old_conflict_deregistered(self):
        """max_word_repeat should no longer be in the registry."""
        assert get_conflict("max_word_repeat") is None
