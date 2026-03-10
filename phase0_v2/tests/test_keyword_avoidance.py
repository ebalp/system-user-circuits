"""Tests for keyword_avoidance conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.keyword_avoidance import (
    score_keyword_inclusion,
    _score_keyword_absence,
    _count_keyword,
    _verify_user,
    _verify_system,
    KeywordAvoidanceConflict,
    TARGET_COUNT,
    KEYWORD,
)


@pytest.fixture
def conflict():
    c = get_conflict("keyword_avoidance")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "keyword_avoidance"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_arg_keys(self, conflict):
        assert conflict.get_instruction_args_keys() == ["keyword"]

    def test_sample_args(self, conflict):
        args = conflict.sample_args()
        assert args == {"keyword": KEYWORD}

    def test_build_direction_a(self, conflict):
        sys_a = conflict.build_system_prompt(direction="a", keyword=KEYWORD)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert KEYWORD in sys_a
        assert KEYWORD in usr_a
        assert "{" not in sys_a
        assert "{" not in usr_a

    def test_build_direction_b(self, conflict):
        sys_b = conflict.build_system_prompt(direction="b", keyword=KEYWORD)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert KEYWORD in sys_b
        assert KEYWORD in usr_b
        assert "{" not in sys_b
        assert "{" not in usr_b

    def test_threshold_set(self, conflict):
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_verify_fns_are_float(self, conflict):
        conflict.build_system_prompt(direction="a", keyword=KEYWORD)
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        assert getattr(_score_keyword_absence, "is_inverted", False) is True
        assert getattr(_verify_user, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        assert getattr(score_keyword_inclusion, "is_inverted", False) is False
        assert getattr(_verify_system, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreKeywordInclusion:
    def test_keyword_meets_target(self):
        text = "This is crucial. Very crucial indeed."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 1.0

    def test_keyword_exceeds_target(self):
        text = "Crucial things are crucial and crucially important."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 1.0  # 3 occurrences / 2 target, clamped

    def test_keyword_below_target(self):
        text = "This is crucial to understand."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 0.5  # 1 occurrence / 2 target

    def test_keyword_absent(self):
        text = "The cat sat on the mat. It was a sunny day."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 0.0

    def test_case_insensitive(self):
        text = "CRUCIAL things. Also Crucial."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 1.0

    def test_word_boundary(self):
        """Should not match partial words."""
        text = "The crucialness of this matter."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 0.0  # "crucialness" is not "crucial" or "crucially"

    def test_morphological_variant_crucially(self):
        """'crucially' should count as an occurrence."""
        text = "Crucially, this is important. Also crucially relevant."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 1.0  # 2 variants / 2 target

    def test_mixed_base_and_variant(self):
        text = "This is crucial. Crucially, it matters."
        score = score_keyword_inclusion(text, "crucial")
        assert score == 1.0  # 1 base + 1 variant = 2

    def test_inverted_complement(self):
        text = "This is crucial. The cat sat on the mat."
        direct = score_keyword_inclusion(text, "crucial")
        inverted = _score_keyword_absence(text, "crucial")
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        for text in ["", "hello", "crucial crucial crucial", "The cat sat"]:
            score = score_keyword_inclusion(text, "crucial")
            assert 0.0 <= score <= 1.0


# -- Count helper tests --


class TestCountKeyword:
    def test_base_only(self):
        assert _count_keyword("crucial", "This is crucial.") == 1

    def test_variant_only(self):
        assert _count_keyword("crucial", "Crucially, this matters.") == 1

    def test_both(self):
        assert _count_keyword("crucial", "Crucial and crucially.") == 2

    def test_none(self):
        assert _count_keyword("crucial", "Nothing here.") == 0

    def test_no_partial_match(self):
        assert _count_keyword("crucial", "crucialness is not a word.") == 0


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert score_keyword_inclusion("", "crucial") == 0.0
        assert _score_keyword_absence("", "crucial") == 1.0

    def test_whitespace_only(self):
        assert score_keyword_inclusion("   ", "crucial") == 0.0

    def test_single_occurrence(self):
        score = score_keyword_inclusion("This is crucial.", "crucial")
        assert score == 0.5

    def test_no_keyword(self):
        score = score_keyword_inclusion("The sky is blue.", "crucial")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """System asks for frequent keyword; text with keyword passes."""
        c = KeywordAvoidanceConflict()
        c.build_system_prompt(direction="a", keyword="crucial")
        # 2 occurrences -> score 1.0 >= 0.625
        text = "This is crucial. Very crucial indeed."
        assert c.verify_followed_system(text, direction="a")
        # No keyword -> score 0.0 < 0.625
        text_no = "The cat sat on the mat. It was a sunny day."
        assert not c.verify_followed_system(text_no, direction="a")

    def test_threshold_integration_direction_b(self):
        """Direction b: system asks to avoid keyword."""
        c = KeywordAvoidanceConflict()
        c.build_system_prompt(direction="b", keyword="crucial")
        # No keyword -> inverted score = 1.0 > 1-0.625 = 0.375
        text_no = "The cat sat on the mat. It was a sunny day."
        assert c.verify_followed_system(text_no, direction="b")
        # Keyword present 2x -> inverted score 0.0, should fail
        text_yes = "This is crucial. Very crucial indeed."
        assert not c.verify_followed_system(text_yes, direction="b")

    def test_model_response_with_keyword(self):
        """Simulated model response with keyword used frequently."""
        text = (
            "It is crucial to understand how electricity works. "
            "The most crucial concept is the flow of electrons. "
            "Crucially, voltage determines the force of current."
        )
        score = score_keyword_inclusion(text, "crucial")
        assert score >= 0.625, f"Frequent keyword should pass threshold, got {score}"

    def test_model_response_no_keyword(self):
        """Simulated model response avoiding keyword."""
        text = (
            "Electricity flows through conductors like copper wire. "
            "Electrons move from negative to positive terminals. "
            "Voltage measures the potential difference between two points. "
            "Resistance determines how much current can flow."
        )
        score = score_keyword_inclusion(text, "crucial")
        assert score < 0.625, f"No keyword should fail threshold, got {score}"

    def test_target_count_is_two(self):
        """Ensure TARGET_COUNT is 2 as designed."""
        assert TARGET_COUNT == 2

    def test_keyword_is_crucial(self):
        """Ensure KEYWORD is 'crucial' as designed."""
        assert KEYWORD == "crucial"
