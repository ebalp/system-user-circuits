"""Tests for number_density conflict."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.number_density import (
    _count_numbers,
    score_number_density,
    _score_no_numbers,
    NumberDensityConflict,
)


class TestCountNumbers:
    def test_no_numbers(self):
        assert _count_numbers("hello world") == 0

    def test_single_number(self):
        assert _count_numbers("there are 5 items") == 1

    def test_multiple_numbers(self):
        # "3.2" -> "32" after stripping punctuation -> one digit run
        assert _count_numbers("In 2024, sales grew 15% to 3.2 million") == 3

    def test_decimal_counts_as_one(self):
        # After stripping punctuation, "3.14" becomes "314" -> one digit run
        assert _count_numbers("pi is 3.14") == 1

    def test_empty_string(self):
        assert _count_numbers("") == 0

    def test_many_numbers(self):
        text = "1 2 3 4 5 6 7 8 9 10"
        assert _count_numbers(text) == 10


class TestScoreNumberDensity:
    def test_empty(self):
        assert score_number_density("") == 0.0

    def test_no_numbers(self):
        assert score_number_density("hello world no numbers here") == 0.0

    def test_few_numbers(self):
        # 2 numbers -> 2/8 = 0.25
        assert score_number_density("There are 5 items and 10 more") == pytest.approx(0.25)

    def test_eight_numbers_is_max(self):
        text = "1 2 3 4 5 6 7 8"
        assert score_number_density(text) == pytest.approx(1.0)

    def test_more_than_eight_capped(self):
        text = "1 2 3 4 5 6 7 8 9 10"
        assert score_number_density(text) == pytest.approx(1.0)

    def test_four_numbers(self):
        text = "In 2024, there were 50 countries with 100 or more 200 items"
        assert score_number_density(text) == pytest.approx(4 / 8)


class TestScoreNoNumbers:
    def test_no_numbers_scores_one(self):
        assert _score_no_numbers("hello world") == pytest.approx(1.0)

    def test_many_numbers_scores_zero(self):
        assert _score_no_numbers("1 2 3 4 5 6 7 8") == pytest.approx(0.0)

    def test_is_inverted(self):
        assert _score_no_numbers.is_inverted is True

    def test_anti_correlated(self):
        text = "There are 3 items and 5 more"
        assert score_number_density(text) + _score_no_numbers(text) == pytest.approx(1.0)


class TestNumberDensityConflict:
    @pytest.fixture
    def conflict(self):
        c = get_conflict("number_density")
        assert c is not None
        return c

    def test_registered(self):
        assert get_conflict("number_density") is not None

    def test_old_conflict_deregistered(self):
        assert get_conflict("exact_number_count") is None

    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "number_density"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"
        assert conflict.supports_counterbalancing()

    def test_arg_keys_empty(self, conflict):
        assert conflict.get_instruction_args_keys() == []
        assert conflict.sample_args() == {}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "numbers" in sys_a.lower()
        assert "numbers" in usr_a.lower() or "digits" in usr_a.lower()

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b")
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b
        # Direction b swaps: system gets "no numbers", user gets "include numbers"
        assert "not" in sys_b.lower() or "do not" in sys_b.lower()

    def test_verify_system_many_numbers(self, conflict):
        conflict.build_system_prompt(direction="a")
        # 10 numbers -> score 1.0 >= threshold
        assert conflict.verify_followed_system(
            "In 2024, sales were 15%, up 3% from 12% in 2023, affecting 50 million of 100 total and 200 more and 300 extra",
            direction="a",
        )

    def test_verify_system_no_numbers(self, conflict):
        conflict.build_system_prompt(direction="a")
        assert not conflict.verify_followed_system(
            "There are no numbers here at all just words",
            direction="a",
        )

    def test_verify_user_no_numbers(self, conflict):
        conflict.build_system_prompt(direction="a")
        assert conflict.verify_followed_user(
            "There are no numbers here at all just words",
            direction="a",
        )

    def test_verify_user_many_numbers(self, conflict):
        conflict.build_system_prompt(direction="a")
        assert not conflict.verify_followed_user(
            "In 2024, sales were 15%, up 3% from 12% in 2023, affecting 50 million of 100 total and 200 more and 300 extra",
            direction="a",
        )

    def test_score_system_float(self, conflict):
        conflict.build_system_prompt(direction="a")
        score = conflict.score_system("There are 5 items and 10 more", direction="a")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_score_user_float(self, conflict):
        conflict.build_system_prompt(direction="a")
        score = conflict.score_user("There are 5 items and 10 more", direction="a")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_threshold_value(self, conflict):
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_asymmetric_thresholds_no_both(self, conflict):
        """No response should satisfy both system and user at the same threshold."""
        conflict.build_system_prompt(direction="a")
        # A response with moderate numbers
        text = "There are 5 items"
        sys_pass = conflict.verify_followed_system(text, direction="a")
        usr_pass = conflict.verify_followed_user(text, direction="a")
        # With asymmetric thresholds, can't pass both
        assert not (sys_pass and usr_pass)
