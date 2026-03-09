"""Tests for address_reader_directly conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.address_reader_directly import (
    score_direct_address,
    _score_impersonal,
    AddressReaderDirectlyConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("address_reader_directly")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "address_reader_directly"

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
        assert conflict.verify_threshold == 0.022

    def test_verify_fns_are_float(self, conflict):
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        assert getattr(_score_impersonal, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        assert getattr(score_direct_address, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreDirectAddress:
    def test_high_you_usage(self):
        text = (
            "You're about to learn one of the most fascinating processes in nature. "
            "This incredible process allows you to understand how plants convert "
            "sunlight into energy. Let yourself explore how it works for you."
        )
        score = score_direct_address(text)
        assert score > 0.05, f"Direct-address text should score high, got {score}"

    def test_no_you_usage(self):
        text = (
            "Machine learning is a subfield of artificial intelligence that involves "
            "the development of algorithms and statistical models that enable computers "
            "to learn from data without being explicitly programmed."
        )
        score = score_direct_address(text)
        assert score == 0.0

    def test_inverted_complement(self):
        text = "You should know that your efforts matter to yourself and your team."
        direct = score_direct_address(text)
        inverted = _score_impersonal(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        for text in ["", "hello", "you you you you you", "The cat sat on the mat"]:
            score = score_direct_address(text)
            assert 0.0 <= score <= 1.0

    def test_all_you_words(self):
        """All you-tokens should be detected."""
        text = "you your yours yourself yourselves"
        score = score_direct_address(text)
        # 5 you-tokens / 5 words = 1.0 raw, scaled = min(1.0, 10.0) = 1.0
        assert score == 1.0

    def test_case_insensitive(self):
        assert score_direct_address("YOU should know") > 0
        assert score_direct_address("Your idea") > 0

    # Tests based on actual model responses from smoke testing

    def test_model_response_direct_address(self):
        """Model response when asked to address reader directly (from smoke test)."""
        text = (
            "You're about to learn one of the most fascinating processes in nature - "
            "photosynthesis. This incredible process allows plants, algae, and some "
            "bacteria to convert sunlight into energy, which they use to grow and "
            "thrive. So, let's dive in and explore how it works."
        )
        score = score_direct_address(text)
        assert score >= 0.022, f"Direct-address response should pass threshold, got {score}"

    def test_model_response_impersonal(self):
        """Model response when asked for impersonal language (from smoke test)."""
        text = (
            "Machine learning is a subfield of artificial intelligence that involves "
            "the development of algorithms and statistical models that enable computers "
            "to learn from data without being explicitly programmed. This process "
            "typically involves the use of large datasets, which are analyzed to "
            "identify patterns and relationships."
        )
        score = score_direct_address(text)
        assert score < 0.022, f"Impersonal response should fail threshold, got {score}"


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert score_direct_address("") == 0.0
        assert _score_impersonal("") == 1.0

    def test_single_you(self):
        score = score_direct_address("you")
        assert score == 1.0  # 1 you-word / 1 word, scaled 10x, clamped

    def test_single_non_you(self):
        score = score_direct_address("hello")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """Integration: verify_followed_system with direction='a' (system=direct address)."""
        c = AddressReaderDirectlyConflict()
        direct = "You should know that your health matters. Take care of yourself."
        assert c.verify_followed_system(direct, direction="a")
        impersonal = "The analysis reveals significant findings in the dataset."
        assert not c.verify_followed_system(impersonal, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration: verify_followed_system with direction='b' (system=impersonal)."""
        c = AddressReaderDirectlyConflict()
        impersonal = "The analysis reveals significant findings in the dataset."
        assert c.verify_followed_system(impersonal, direction="b")
        direct = "You should know that your health matters. Take care of yourself."
        assert not c.verify_followed_system(direct, direction="b")

    def test_you_embedded_in_word_not_matched(self):
        """Words containing 'you' as substring should not match (e.g., 'youth')."""
        # 'youth' contains 'you' but \b boundary prevents match
        score = score_direct_address("The youth gathered at the fountain")
        assert score == 0.0
