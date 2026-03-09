"""Tests for pronoun_density conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.pronoun_density import (
    pronoun_count,
    score_pronoun_density,
    _score_no_pronouns,
    PronounDensityConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("pronoun_density")
    assert c is not None
    return c


# ── Contract tests ──


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "pronoun_density"

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
        assert conflict.verify_threshold == 0.040

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The no-pronouns scorer should have is_inverted=True."""
        assert getattr(_score_no_pronouns, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(score_pronoun_density, "is_inverted", False) is False


# ── Pronoun count tests ──


class TestPronounCount:
    def test_basic(self):
        assert pronoun_count("I went to the store") >= 1

    def test_impersonal_excluded(self):
        text = "It is raining. Its handle broke."
        assert pronoun_count(text, exclude_impersonal=True) == 0
        assert pronoun_count(text, exclude_impersonal=False) >= 2

    def test_multiple_pronouns(self):
        text = "I told you that we should go and she agreed with them."
        count = pronoun_count(text, exclude_impersonal=True)
        # I, you, we, she, them = 5
        assert count == 5

    def test_case_insensitive(self):
        assert pronoun_count("I said I would", exclude_impersonal=True) == 2
        assert pronoun_count("MY house YOUR car", exclude_impersonal=True) == 2


# ── Scorer tests ──


class TestScorePronounDensity:
    def test_high_pronoun_text(self):
        """Conversational text with many pronouns should score well above threshold."""
        text = (
            "I think you should know that we really enjoyed our time together. "
            "My favorite part was when you shared your story with me."
        )
        score = score_pronoun_density(text)
        assert score > 0.05  # well above 0.022 threshold

    def test_no_pronoun_text(self):
        """Impersonal text should score near zero."""
        text = (
            "The periodic table organizes elements by atomic number. "
            "Metals appear on the left side. Nonmetals appear on the right side."
        )
        score = score_pronoun_density(text)
        assert score < 0.01

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "I went to the store and bought some milk for my family."
        direct = score_pronoun_density(text)
        inverted = _score_no_pronouns(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "I I I I I", "The cat sat on the mat"]:
            score = score_pronoun_density(text)
            assert 0.0 <= score <= 1.0

    # Tests based on actual model responses from smoke testing

    def test_model_response_personal_style(self):
        """Model response when asked for personal style (from smoke test, ~0.04-0.09 density)."""
        text = (
            "Let's dive into the fascinating world of electricity generation and "
            "distribution. I'm excited to share this with you. The process of "
            "generating electricity starts with power plants, which are essentially "
            "large factories that convert various forms of energy into electricity."
        )
        score = score_pronoun_density(text)
        assert score >= 0.040, f"Personal style should pass threshold, got {score}"

    def test_model_response_impersonal_style(self):
        """Model response when asked for impersonal style (from smoke test, ~0.0 density)."""
        text = (
            "Vaccines utilize Immunology to stimulate Immune System responses. "
            "Process begins with introduction of Antigens, derived from weakened "
            "or inactivated Pathogens, such as Influenza Virus or Measles Virus. "
            "Antigens interact with Immune Cells, including Dendritic Cells and "
            "T-Cells, in Lymph Nodes."
        )
        score = score_pronoun_density(text)
        assert score < 0.040, f"Impersonal style should fail threshold, got {score}"


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_response(self):
        assert score_pronoun_density("") == 0.0
        assert _score_no_pronouns("") == 1.0

    def test_single_pronoun(self):
        score = score_pronoun_density("I")
        assert score == 1.0  # 1 pronoun / 1 word

    def test_single_non_pronoun(self):
        score = score_pronoun_density("hello")
        assert score == 0.0

    def test_very_short_response(self):
        """Two-word response should still work."""
        score = score_pronoun_density("I agree")
        assert score == 0.5  # 1 pronoun / 2 words

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a'."""
        c = PronounDensityConflict()
        # Personal text should pass system check in direction a
        personal = "I think you should know that we all believe in this."
        assert c.verify_followed_system(personal, direction="a")
        # Impersonal text should fail system check in direction a
        impersonal = "The analysis reveals significant findings in the dataset."
        assert not c.verify_followed_system(impersonal, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (inverted)."""
        c = PronounDensityConflict()
        # Impersonal text should pass system check in direction b (system=no pronouns)
        impersonal = "The analysis reveals significant findings in the dataset."
        assert c.verify_followed_system(impersonal, direction="b")
        # Personal text should fail system check in direction b
        personal = "I think you should know that we all believe in this."
        assert not c.verify_followed_system(personal, direction="b")
