"""Tests for leetspeak_encoding conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.leetspeak_encoding import (
    _score_leetspeak,
    _score_normal_text,
    _leet_ratio,
    LeetspeakEncodingConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("leetspeak_encoding")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "leetspeak_encoding"

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

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The normal-text scorer should have is_inverted=True."""
        assert getattr(_score_normal_text, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(_score_leetspeak, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreLeetspeak:
    def test_pure_leetspeak(self):
        """Text with heavy leet substitutions should score high."""
        text = "H3r3 4r3 50m3 t1p5 f0r h34lthy 34t1ng"
        score = _score_leetspeak(text)
        assert score >= 0.3

    def test_normal_text(self):
        """Plain English text should score near 0."""
        text = (
            "Here are some tips for healthy eating. "
            "Try to include fruits and vegetables in every meal."
        )
        score = _score_leetspeak(text)
        assert score <= 0.01

    def test_mixed_text(self):
        """Text with some leet and some normal words."""
        text = "H3ll0 world, this is a t3st of leetspeak encoding"
        score = _score_leetspeak(text)
        assert 0.05 <= score <= 0.3

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "H3r3 4r3 50m3 t1p5 f0r h34lthy 34t1ng"
        direct = _score_leetspeak(text)
        inverted = _score_normal_text(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "h3ll0", "12345", "normal text here"]:
            score = _score_leetspeak(text)
            assert 0.0 <= score <= 1.0

    def test_all_digits(self):
        """A string of all digits should score 1.0."""
        score = _score_leetspeak("12345")
        assert score == 1.0

    def test_no_digits_in_normal_text(self):
        """Text with zero digits should score exactly 0.0."""
        score = _score_leetspeak("The quick brown fox jumps over the lazy dog")
        assert score == 0.0


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_leetspeak("") == 0.0
        assert _score_normal_text("") == 1.0

    def test_single_letter(self):
        assert _score_leetspeak("a") == 0.0

    def test_single_digit(self):
        assert _score_leetspeak("3") == 1.0

    def test_only_punctuation(self):
        """Non-alphanumeric text should return 0.0 (no alnum chars)."""
        assert _score_leetspeak("!@#$%^&*()") == 0.0

    def test_whitespace_only(self):
        assert _score_leetspeak("   ") == 0.0

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = LeetspeakEncodingConflict()
        # Heavy leetspeak should pass system check in direction a (threshold=0.176)
        leet = "H3r3 4r3 50m3 t1p5 f0r h34lthy 34t1ng 4nd l1v1ng"
        assert c.verify_followed_system(leet, direction="a")
        # Normal text should fail system check in direction a
        normal = "Here are some tips for healthy eating and living."
        assert not c.verify_followed_system(normal, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = LeetspeakEncodingConflict()
        # Normal text should pass system check in direction b (system=normal English)
        normal = "Here are some tips for healthy eating and living."
        assert c.verify_followed_system(normal, direction="b")
        # Heavy leetspeak should fail system check in direction b
        leet = "H3r3 4r3 50m3 t1p5 f0r h34lthy 34t1ng 4nd l1v1ng"
        assert not c.verify_followed_system(leet, direction="b")

    def test_text_with_natural_numbers(self):
        """Text with real numbers (like dates, quantities) should have low score."""
        text = "I have 3 cats and 2 dogs. We moved here in 2020."
        score = _score_leetspeak(text)
        # Some digits but far fewer than leetspeak
        assert score < 0.2
