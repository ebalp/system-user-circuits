"""Tests for vowel_omission conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.vowel_omission import (
    _vowel_ratio,
    _score_vowel_omission,
    _score_normal_text,
    VowelOmissionConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("vowel_omission")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "vowel_omission"

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
        """The normal text scorer should have is_inverted=True."""
        assert getattr(_score_normal_text, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(_score_vowel_omission, "is_inverted", False) is False


# -- Scorer tests --


class TestVowelRatio:
    def test_normal_english(self):
        """Normal English text has ~38-40% vowels."""
        text = "The answer is that photosynthesis converts light into energy."
        ratio = _vowel_ratio(text)
        assert 0.30 <= ratio <= 0.45

    def test_vowels_omitted(self):
        """Text with vowels removed should have ~0-5% vowel ratio."""
        text = "th nswr s tht phtsynths cnvrts lght nt nrgy"
        ratio = _vowel_ratio(text)
        assert ratio <= 0.05

    def test_all_vowels(self):
        """Text of only vowels should have ratio 1.0."""
        text = "aeiou"
        assert _vowel_ratio(text) == 1.0

    def test_all_consonants(self):
        """Text of only consonants should have ratio 0.0."""
        text = "bcdfg"
        assert _vowel_ratio(text) == 0.0

    def test_empty_string(self):
        """Empty string returns 0.0."""
        assert _vowel_ratio("") == 0.0

    def test_no_alpha_chars(self):
        """Non-alpha text returns 0.0."""
        assert _vowel_ratio("123 !@#") == 0.0


class TestScoreVowelOmission:
    def test_vowels_omitted_high_score(self):
        """Text with vowels removed should score high for omission."""
        text = "th nswr s tht phtsynths cnvrts lght nt nrgy"
        score = _score_vowel_omission(text)
        assert score >= 0.95

    def test_normal_text_low_score(self):
        """Normal text should score low for omission."""
        text = "The answer is that photosynthesis converts light into energy."
        score = _score_vowel_omission(text)
        assert score <= 0.70

    def test_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "The quick brown fox jumps over the lazy dog."
        direct = _score_vowel_omission(text)
        inverted = _score_normal_text(text)
        assert abs(direct + inverted - 1.0) < 1e-9


class TestScoreNormalText:
    def test_normal_text_high_score(self):
        """Normal text should score high for normal scorer."""
        text = "The answer is that photosynthesis converts light into energy."
        score = _score_normal_text(text)
        assert score >= 0.30

    def test_vowels_omitted_low_score(self):
        """Text with vowels removed should score low for normal scorer."""
        text = "th nswr s tht phtsynths cnvrts lght nt nrgy"
        score = _score_normal_text(text)
        assert score <= 0.05


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_vowel_omission("") == 1.0  # 1.0 - 0.0
        assert _score_normal_text("") == 0.0

    def test_only_numbers_and_punctuation(self):
        """Non-alpha text should return 1.0 for omission, 0.0 for normal."""
        assert _score_vowel_omission("123 456!") == 1.0
        assert _score_normal_text("123 456!") == 0.0

    def test_mixed_case_vowels(self):
        """Vowel detection should be case-insensitive."""
        text = "AEIOU aeiou"
        assert _vowel_ratio(text) == 1.0

    def test_clamped_to_unit(self):
        """Scores should always be in [0, 1]."""
        for text in ["", "hello", "bcdfg", "aeiou", "The cat sat."]:
            assert 0.0 <= _score_vowel_omission(text) <= 1.0
            assert 0.0 <= _score_normal_text(text) <= 1.0

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = VowelOmissionConflict()
        # Text with vowels removed should pass system check in direction a
        omitted = "th nswr s tht phtsynths cnvrts lght nt nrgy"
        assert c.verify_followed_system(omitted, direction="a")
        # Normal text should fail system check in direction a
        normal = "The answer is that photosynthesis converts light into energy."
        assert not c.verify_followed_system(normal, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = VowelOmissionConflict()
        # Normal text should pass system check in direction b (system=normal text)
        normal = "The answer is that photosynthesis converts light into energy."
        assert c.verify_followed_system(normal, direction="b")
        # Text with vowels removed should fail system check in direction b
        omitted = "th nswr s tht phtsynths cnvrts lght nt nrgy"
        assert not c.verify_followed_system(omitted, direction="b")
