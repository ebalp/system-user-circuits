"""Tests for keyword_frequency conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.keyword_frequency import (
    score_keyword_frequency,
    _score_no_keyword,
    _verify_user,
    _verify_system,
    KeywordFrequencyConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("keyword_frequency")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "keyword_frequency"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_arg_keys(self, conflict):
        assert conflict.get_instruction_args_keys() == ["keyword"]

    def test_sample_args(self, conflict):
        args = conflict.sample_args()
        assert args == {"keyword": "important"}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a", keyword="important")
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert "important" in sys_a
        assert "important" in usr_a
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b", keyword="important")
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert "important" in sys_b
        assert "important" in usr_b
        assert "{" not in sys_b
        assert "{" not in usr_b

    def test_threshold_set(self, conflict):
        assert conflict.verify_threshold == 0.211

    def test_verify_fns_are_float(self, conflict):
        conflict.build_system_prompt(direction="a", keyword="important")
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        assert getattr(_score_no_keyword, "is_inverted", False) is True
        assert getattr(_verify_user, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        assert getattr(score_keyword_frequency, "is_inverted", False) is False
        assert getattr(_verify_system, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreKeywordFrequency:
    def test_keyword_in_every_sentence(self):
        text = "This is important. That is also important. Everything is important."
        score = score_keyword_frequency(text, "important")
        assert score == 1.0

    def test_keyword_in_no_sentence(self):
        text = "The cat sat on the mat. It was a sunny day."
        score = score_keyword_frequency(text, "important")
        assert score == 0.0

    def test_keyword_in_some_sentences(self):
        text = "This is important. The cat sat. Also important. Nothing here."
        score = score_keyword_frequency(text, "important")
        assert score == 0.5  # 2 occurrences / 4 sentences

    def test_multiple_in_one_sentence(self):
        text = "Important things are important."
        score = score_keyword_frequency(text, "important")
        assert score == 1.0  # 2 occurrences / 1 sentence, clamped to 1.0

    def test_case_insensitive(self):
        text = "This is IMPORTANT. Also Important."
        score = score_keyword_frequency(text, "important")
        assert score == 1.0

    def test_word_boundary(self):
        """Should not match 'importantly' or 'unimportant'."""
        text = "Importantly, this is unimportant."
        score = score_keyword_frequency(text, "important")
        assert score == 0.0

    def test_inverted_complement(self):
        text = "This is important. The cat sat on the mat."
        direct = score_keyword_frequency(text, "important")
        inverted = _score_no_keyword(text, "important")
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        for text in ["", "hello", "important important important", "The cat sat"]:
            score = score_keyword_frequency(text, "important")
            assert 0.0 <= score <= 1.0


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert score_keyword_frequency("", "important") == 0.0
        assert _score_no_keyword("", "important") == 1.0

    def test_whitespace_only(self):
        assert score_keyword_frequency("   ", "important") == 0.0

    def test_single_sentence_with_keyword(self):
        score = score_keyword_frequency("This is important.", "important")
        assert score == 1.0

    def test_single_sentence_without_keyword(self):
        score = score_keyword_frequency("The sky is blue.", "important")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """System asks for frequent keyword; text with keyword passes."""
        c = KeywordFrequencyConflict()
        c.build_system_prompt(direction="a", keyword="important")
        # 3 sentences with "important" in each -> score ~ 1.0 >> 0.211
        text = "This is important. Very important indeed. The most important thing."
        assert c.verify_followed_system(text, direction="a")
        # No keyword -> score 0.0 < 0.211
        text_no = "The cat sat on the mat. It was a sunny day."
        assert not c.verify_followed_system(text_no, direction="a")

    def test_threshold_integration_direction_b(self):
        """Direction b: system asks to avoid keyword."""
        c = KeywordFrequencyConflict()
        c.build_system_prompt(direction="b", keyword="important")
        # No keyword -> inverted score = 1.0 > 1-0.211 = 0.789
        text_no = "The cat sat on the mat. It was a sunny day."
        assert c.verify_followed_system(text_no, direction="b")
        # Keyword present -> inverted score low, should fail
        text_yes = "This is important. Very important indeed."
        assert not c.verify_followed_system(text_yes, direction="b")

    def test_model_response_frequent_keyword(self):
        """Simulated model response with frequent 'important' use."""
        text = (
            "It is important to understand how electricity works. "
            "The most important concept is the flow of electrons. "
            "An important distinction is between AC and DC current. "
            "Understanding voltage is also important for safety."
        )
        score = score_keyword_frequency(text, "important")
        assert score >= 0.211, f"Frequent keyword should pass threshold, got {score}"

    def test_model_response_no_keyword(self):
        """Simulated model response avoiding 'important'."""
        text = (
            "Electricity flows through conductors like copper wire. "
            "Electrons move from negative to positive terminals. "
            "Voltage measures the potential difference between two points. "
            "Resistance determines how much current can flow."
        )
        score = score_keyword_frequency(text, "important")
        assert score < 0.211, f"No keyword should fail threshold, got {score}"
