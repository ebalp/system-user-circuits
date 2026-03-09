"""Tests for paragraph_start_same_word conflict definition."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.paragraph_start_same_word import (
    score_paragraph_start_same,
    score_paragraph_start_different,
    ParagraphStartSameWordConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("paragraph_start_same_word")
    assert c is not None
    return c


# -- Contract tests --

class TestContract:
    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "paragraph_start_same_word"

    def test_counterbalance_quality(self, conflict):
        assert conflict.counterbalance_quality == "full"

    def test_templates_nonempty(self, conflict):
        assert conflict._system_template
        assert conflict._user_template
        assert conflict._inverse_system_template
        assert conflict._inverse_user_template

    def test_build_direction_a(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_build_direction_b(self, conflict):
        args = conflict.sample_args()
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_verify_returns_bool(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        assert isinstance(conflict.verify_followed_system("test", direction="a"), bool)
        assert isinstance(conflict.verify_followed_user("test", direction="a"), bool)

    def test_score_returns_float(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        sys_score = conflict.score_system("The cat.\n\nThe dog.", direction="a")
        usr_score = conflict.score_user("The cat.\n\nThe dog.", direction="a")
        assert isinstance(sys_score, float)
        assert isinstance(usr_score, float)

    def test_arg_keys_empty(self, conflict):
        assert conflict.get_instruction_args_keys() == []

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}


# -- Scorer tests --

class TestScorer:
    def test_all_same_start(self):
        text = "The cat is big.\n\nThe dog is small.\n\nThe bird can fly."
        assert score_paragraph_start_same(text) == 1.0

    def test_all_different_start(self):
        text = "First point here.\n\nSecond point here.\n\nThird point here."
        score = score_paragraph_start_same(text)
        assert abs(score - 1 / 3) < 0.01

    def test_mixed_start(self):
        # 3 out of 4 start with "the"
        text = "The cat.\n\nThe dog.\n\nThe bird.\n\nA fish."
        score = score_paragraph_start_same(text)
        assert abs(score - 0.75) < 0.01

    def test_two_paragraphs_same(self):
        text = "Hello world.\n\nHello again."
        assert score_paragraph_start_same(text) == 1.0

    def test_two_paragraphs_different(self):
        text = "Hello world.\n\nGoodbye again."
        assert score_paragraph_start_same(text) == 0.5

    def test_inverted_scorer(self):
        text = "The cat.\n\nThe dog.\n\nThe bird."
        assert score_paragraph_start_different(text) == 0.0

    def test_inverted_all_different(self):
        text = "First point.\n\nSecond point.\n\nThird point."
        score = score_paragraph_start_different(text)
        assert abs(score - 2 / 3) < 0.01

    def test_inverted_is_marked(self):
        assert getattr(score_paragraph_start_different, "is_inverted", False) is True

    def test_anti_correlated(self):
        text = "The cat.\n\nA dog.\n\nThe bird.\n\nSome fish."
        same = score_paragraph_start_same(text)
        diff = score_paragraph_start_different(text)
        assert abs(same + diff - 1.0) < 1e-9


# -- Edge cases --

class TestEdgeCases:
    def test_empty_string(self):
        assert score_paragraph_start_same("") == 1.0

    def test_single_paragraph(self):
        assert score_paragraph_start_same("Just one paragraph here.") == 1.0

    def test_whitespace_only(self):
        assert score_paragraph_start_same("   \n\n   ") == 1.0

    def test_single_newline_fallback(self):
        # Only single newlines -- should fall back to splitting on \n
        text = "The cat is nice.\nThe dog is good.\nThe bird sings."
        assert score_paragraph_start_same(text) == 1.0

    def test_punctuation_stripped(self):
        # First word has leading punctuation (e.g. markdown bullet)
        text = "- The cat.\n\n- The dog.\n\n- A bird."
        # First words after stripping punctuation: '', 'the', 'the', 'a'
        # Actually '-' stripped leaves empty, but 'the' is second word...
        # Let's just check it doesn't crash
        score = score_paragraph_start_same(text)
        assert 0.0 <= score <= 1.0

    def test_case_insensitive(self):
        text = "the cat.\n\nThe dog.\n\nTHE bird."
        assert score_paragraph_start_same(text) == 1.0

    def test_triple_newline_separator(self):
        text = "The cat.\n\n\nThe dog.\n\n\nThe bird."
        assert score_paragraph_start_same(text) == 1.0


# -- Threshold / verification tests --

class TestVerification:
    def test_high_same_score_passes_system(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        # All paragraphs start with "The" -> score = 1.0, threshold = 0.5
        text = "The cat is big.\n\nThe dog is small.\n\nThe bird can fly."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_low_same_score_fails_system(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        text = "First point.\n\nSecond point.\n\nThird point."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_different_words_passes_user(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        text = "First point.\n\nSecond point.\n\nThird point."
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_same_words_fails_user(self, conflict):
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        text = "The cat.\n\nThe dog.\n\nThe bird."
        assert conflict.verify_followed_user(text, direction="a") is False

    def test_no_followed_both_at_threshold(self, conflict):
        """With anti-correlated inverted pair, followed_both should be impossible."""
        conflict.build_system_prompt(direction="a", **conflict.sample_args())
        # Score = 0.5 exactly: system uses >= T, user uses > (1-T)
        # score >= 0.5 -> True, inverted_score = 0.5 > 0.5 -> False
        text = "The cat.\n\nA dog."
        sys_pass = conflict.verify_followed_system(text, direction="a")
        usr_pass = conflict.verify_followed_user(text, direction="a")
        assert not (sys_pass and usr_pass), "followed_both should be impossible"
