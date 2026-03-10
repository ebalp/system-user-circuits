"""Edge case tests for lowercase_vs_capitalized conflict."""

import pytest
from phase0_v2.conflicts.registry import get_conflict


class TestLowercaseVsCapitalized:
    @pytest.fixture
    def c(self):
        return get_conflict("lowercase_vs_capitalized")

    # -- all lowercase (system side, direction a) --

    def test_all_lowercase_passes(self, c):
        assert c.verify_followed_system(
            "the sun is a star. it provides light and heat.", direction="a"
        )

    def test_all_lowercase_no_punctuation(self, c):
        assert c.verify_followed_system(
            "hello world this is a test", direction="a"
        )

    def test_all_lowercase_with_numbers(self, c):
        assert c.verify_followed_system(
            "there are 7 continents and 5 oceans on earth.", direction="a"
        )

    def test_single_capital_in_long_text_passes(self, c):
        """A single 'I' in 500+ chars should still be <= 0.3% uppercase."""
        text = "i think that " * 30 + "I think so."  # ~1 capital in ~400 chars
        # The single I is < 0.3% of alpha chars
        alpha = [ch for ch in text if ch.isalpha()]
        upper_frac = sum(1 for ch in alpha if ch.isupper()) / len(alpha)
        if upper_frac <= 0.003:
            assert c.verify_followed_system(text, direction="a")
        else:
            assert not c.verify_followed_system(text, direction="a")

    def test_sentence_initial_caps_fails_lowercase(self, c):
        assert not c.verify_followed_system(
            "The sun is a star. It provides light and heat.", direction="a"
        )

    def test_all_caps_fails_lowercase(self, c):
        assert not c.verify_followed_system(
            "THE SUN IS A STAR.", direction="a"
        )

    def test_empty_text_passes_lowercase(self, c):
        """Empty text has no uppercase chars, so it passes."""
        assert c.verify_followed_system("", direction="a")

    # -- properly capitalized (user side, direction a) --

    def test_properly_capitalized_passes(self, c):
        assert c.verify_followed_user(
            "The sun is a star. It provides light and heat.", direction="a"
        )

    def test_multiple_sentences_capitalized(self, c):
        assert c.verify_followed_user(
            "Photosynthesis is important. Plants use sunlight to make food. "
            "This process produces oxygen.",
            direction="a",
        )

    def test_all_lowercase_fails_capitalized(self, c):
        assert not c.verify_followed_user(
            "the sun is a star. it provides light and heat.", direction="a"
        )

    def test_all_caps_passes_capitalized(self, c):
        """ALL CAPS has >0.5% uppercase, so it passes the capitalized check."""
        assert c.verify_followed_user(
            "THE SUN IS A STAR. IT PROVIDES LIGHT.", direction="a"
        )

    def test_empty_text_fails_capitalized(self, c):
        """Empty text has 0% uppercase, below 0.5% threshold."""
        assert not c.verify_followed_user("", direction="a")

    # -- counterbalancing (direction b swaps) --

    def test_direction_b_system_is_capitalized(self, c):
        """In direction b, system wants proper capitalization."""
        assert c.verify_followed_system(
            "The sky is blue. Water is essential for life.", direction="b"
        )

    def test_direction_b_user_is_lowercase(self, c):
        """In direction b, user wants all lowercase."""
        assert c.verify_followed_user(
            "the sky is blue. water is essential for life.", direction="b"
        )

    def test_direction_b_system_rejects_lowercase(self, c):
        assert not c.verify_followed_system(
            "the sky is blue. water is vital.", direction="b"
        )

    def test_direction_b_user_rejects_capitalized(self, c):
        assert not c.verify_followed_user(
            "The sky is blue. Water is vital.", direction="b"
        )

    # -- edge cases --

    def test_numbers_only_lowercase(self, c):
        """Text with only numbers and punctuation, no alpha."""
        assert c.verify_followed_system("123 456 789.", direction="a")

    def test_short_text_with_one_cap(self, c):
        """Short text: 'I am' has 1 upper / 3 alpha = 33% -> fails lowercase."""
        assert not c.verify_followed_system("I am", direction="a")

    def test_short_text_all_lower(self, c):
        assert c.verify_followed_system("hello", direction="a")

    def test_mutual_exclusion(self, c):
        """Lowercase text cannot pass both verifiers simultaneously."""
        text = "the earth orbits the sun. this takes about a year."
        assert c.verify_followed_system(text, direction="a")
        assert not c.verify_followed_user(text, direction="a")

    def test_capitalized_text_mutual_exclusion(self, c):
        """Capitalized text cannot pass both verifiers simultaneously."""
        text = "The Earth orbits the Sun. This takes about a year."
        assert not c.verify_followed_system(text, direction="a")
        assert c.verify_followed_user(text, direction="a")

    def test_sample_args_empty(self, c):
        assert c.sample_args() == {}

    def test_counterbalance_quality(self, c):
        assert c.counterbalance_quality == "full"

    def test_supports_counterbalancing(self, c):
        assert c.supports_counterbalancing()
