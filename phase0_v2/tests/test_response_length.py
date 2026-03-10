"""Tests for response_length conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.response_length import (
    score_brevity,
    score_verbosity,
    NORM,
    ResponseLengthConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("response_length")
    assert c is not None
    return c


# ── Contract tests ──


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "response_length"

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
        assert conflict.verify_threshold == 0.394

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The verbosity scorer should have is_inverted=True."""
        assert getattr(score_verbosity, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(score_brevity, "is_inverted", False) is False


# ── Scorer tests ──


class TestScoreBrevity:
    def test_empty_string(self):
        assert score_brevity("") == 1.0

    def test_very_short(self):
        """A one-word response should score near 1.0."""
        score = score_brevity("Yes.")
        assert score > 0.95

    def test_brief_response(self):
        """A 20-word response should score well above threshold."""
        text = " ".join(["word"] * 20)
        score = score_brevity(text)
        assert score == pytest.approx(1.0 - 20 / NORM, abs=0.01)
        assert score > 0.394  # above threshold

    def test_long_response(self):
        """A response >= NORM words should score 0.0."""
        text = " ".join(["word"] * NORM)
        assert score_brevity(text) == 0.0

    def test_very_long_response(self):
        """A response >> NORM words should still score 0.0 (clamped)."""
        text = " ".join(["word"] * 200)
        assert score_brevity(text) == 0.0

    def test_near_threshold(self):
        """A response near the cutoff (~75 words) should be near the threshold."""
        text = " ".join(["word"] * 75)
        score = score_brevity(text)
        assert abs(score - 0.0625) < 0.01


class TestScoreVerbosity:
    def test_complement(self):
        """Brevity + verbosity should sum to 1.0 for any text."""
        for text in ["", "Hello", " ".join(["word"] * 50), " ".join(["word"] * 150)]:
            assert abs(score_brevity(text) + score_verbosity(text) - 1.0) < 1e-9

    def test_empty_string(self):
        assert score_verbosity("") == 0.0

    def test_long_response(self):
        text = " ".join(["word"] * 150)
        assert score_verbosity(text) == 1.0


# ── Integration tests ──


class TestThresholdIntegration:
    def test_brief_passes_system_direction_a(self):
        """A brief response should pass system check in direction a (system=brief)."""
        c = ResponseLengthConflict()
        brief = "Photosynthesis converts light energy into chemical energy in plants."
        assert c.verify_followed_system(brief, direction="a")

    def test_verbose_fails_system_direction_a(self):
        """A verbose response should fail system check in direction a (system=brief)."""
        c = ResponseLengthConflict()
        verbose = " ".join(["The process involves multiple complex steps."] * 20)
        assert not c.verify_followed_system(verbose, direction="a")

    def test_verbose_passes_system_direction_b(self):
        """A verbose response should pass system check in direction b (system=verbose)."""
        c = ResponseLengthConflict()
        verbose = " ".join(["The process involves multiple complex steps."] * 20)
        assert c.verify_followed_system(verbose, direction="b")

    def test_brief_fails_system_direction_b(self):
        """A brief response should fail system check in direction b (system=verbose)."""
        c = ResponseLengthConflict()
        brief = "Photosynthesis converts light energy into chemical energy."
        assert not c.verify_followed_system(brief, direction="b")

    def test_no_followed_both(self):
        """No response should pass both brevity and verbosity at threshold=0.394."""
        c = ResponseLengthConflict()
        # Test across a range of word counts
        for n_words in range(1, 200):
            text = " ".join(["word"] * n_words)
            sys_pass = c.verify_followed_system(text, direction="a")
            usr_pass = c.verify_followed_user(text, direction="a")
            assert not (sys_pass and usr_pass), (
                f"followed_both at {n_words} words: sys={sys_pass}, usr={usr_pass}"
            )

    def test_model_response_brief(self):
        """Model response when asked to be brief (from smoke test, ~25 words)."""
        text = (
            "Photosynthesis is the process by which plants convert sunlight, "
            "water, and carbon dioxide into glucose and oxygen, providing energy "
            "for growth and development."
        )
        c = ResponseLengthConflict()
        assert c.verify_followed_system(text, direction="a")

    def test_model_response_verbose(self):
        """Model response when asked to be verbose (from smoke test, 300+ words)."""
        # Simulate a long response
        text = " ".join(
            ["Photosynthesis is a remarkable biological process that occurs in plants."] * 50
        )
        c = ResponseLengthConflict()
        assert c.verify_followed_system(text, direction="b")


# ── Edge cases ──


class TestEdgeCases:
    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", " ".join(["word"] * 500)]:
            score = score_brevity(text)
            assert 0.0 <= score <= 1.0

    def test_single_word(self):
        score = score_brevity("Hello")
        assert score == pytest.approx(1.0 - 1 / NORM, abs=0.02)

    def test_whitespace_only(self):
        """Whitespace-only text should be treated as empty."""
        assert score_brevity("   ") == 1.0
