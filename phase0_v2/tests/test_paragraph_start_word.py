"""Tests for paragraph_start_word conflict definition."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.paragraph_start_word import (
    score_paragraph_start_word,
    score_paragraph_avoid_word,
    ParagraphStartWordConflict,
)
from phase0_v2.config.thresholds import get_threshold


@pytest.fixture
def conflict():
    c = get_conflict("paragraph_start_word")
    assert c is not None
    return c


# -- Contract tests --

class TestContract:
    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "paragraph_start_word"

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
        # Target word should appear in the rendered template
        assert args["target_word"] in sys_a
        assert args["target_word"] in usr_a

    def test_build_direction_b(self, conflict):
        args = conflict.sample_args()
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        assert sys_b and usr_b

    def test_verify_returns_bool(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        assert isinstance(conflict.verify_followed_system("test", direction="a"), bool)
        assert isinstance(conflict.verify_followed_user("test", direction="a"), bool)

    def test_score_returns_float(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        text = "Furthermore, cats.\n\nFurthermore, dogs."
        sys_score = conflict.score_system(text, direction="a")
        usr_score = conflict.score_user(text, direction="a")
        assert isinstance(sys_score, float)
        assert isinstance(usr_score, float)

    def test_arg_keys(self, conflict):
        assert conflict.get_instruction_args_keys() == ["target_word"]

    def test_sample_args_has_target_word(self, conflict):
        args = conflict.sample_args()
        assert "target_word" in args
        assert isinstance(args["target_word"], str)
        assert len(args["target_word"]) > 0


# -- Scorer tests --

class TestScorer:
    def test_all_match(self):
        text = "Furthermore, cats are great.\n\nFurthermore, dogs are loyal.\n\nFurthermore, birds sing."
        assert score_paragraph_start_word(text, "Furthermore") == 1.0

    def test_none_match(self):
        text = "Cats are great.\n\nDogs are loyal.\n\nBirds can sing."
        assert score_paragraph_start_word(text, "Furthermore") == 0.0

    def test_partial_match(self):
        # 2 out of 3 start with "Indeed"
        text = "Indeed, cats are great.\n\nIndeed, dogs are loyal.\n\nBirds can sing."
        score = score_paragraph_start_word(text, "Indeed")
        assert abs(score - 2 / 3) < 0.01

    def test_case_insensitive(self):
        text = "furthermore, cats.\n\nFURTHERMORE, dogs.\n\nFurthermore, birds."
        assert score_paragraph_start_word(text, "Furthermore") == 1.0

    def test_inverted_scorer_all_match(self):
        text = "Furthermore, cats.\n\nFurthermore, dogs.\n\nFurthermore, birds."
        assert score_paragraph_avoid_word(text, "Furthermore") == 0.0

    def test_inverted_scorer_none_match(self):
        text = "Cats are great.\n\nDogs are loyal.\n\nBirds sing."
        assert score_paragraph_avoid_word(text, "Furthermore") == 1.0

    def test_inverted_is_marked(self):
        assert getattr(score_paragraph_avoid_word, "is_inverted", False) is True

    def test_anti_correlated(self):
        text = "Furthermore, cats.\n\nDogs are loyal.\n\nFurthermore, birds.\n\nFish swim."
        same = score_paragraph_start_word(text, "Furthermore")
        diff = score_paragraph_avoid_word(text, "Furthermore")
        assert abs(same + diff - 1.0) < 1e-9


# -- Edge cases --

class TestEdgeCases:
    def test_empty_string(self):
        assert score_paragraph_start_word("", "Furthermore") == 0.0

    def test_single_paragraph_match(self):
        assert score_paragraph_start_word("Furthermore, just one.", "Furthermore") == 1.0

    def test_single_paragraph_no_match(self):
        assert score_paragraph_start_word("Just one paragraph.", "Furthermore") == 0.0

    def test_whitespace_only(self):
        assert score_paragraph_start_word("   \n\n   ", "Furthermore") == 0.0

    def test_single_newline_fallback(self):
        text = "Furthermore, cats.\nFurthermore, dogs.\nFurthermore, birds."
        assert score_paragraph_start_word(text, "Furthermore") == 1.0

    def test_punctuation_stripped(self):
        # First word has trailing comma/colon
        text = "Furthermore: cats.\n\nFurthermore, dogs."
        assert score_paragraph_start_word(text, "Furthermore") == 1.0

    def test_triple_newline_separator(self):
        text = "Furthermore, cats.\n\n\nFurthermore, dogs.\n\n\nFurthermore, birds."
        assert score_paragraph_start_word(text, "Furthermore") == 1.0

    def test_different_target_words(self):
        text = "Indeed, cats.\n\nIndeed, dogs."
        assert score_paragraph_start_word(text, "Indeed") == 1.0
        assert score_paragraph_start_word(text, "However") == 0.0


# -- Threshold / verification tests --

class TestVerification:
    def test_threshold_from_config(self):
        t = get_threshold("paragraph_start_word")
        assert t is not None
        assert 0.0 < t < 1.0

    def test_all_match_passes_system(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        text = "Furthermore, cats are great.\n\nFurthermore, dogs are loyal."
        assert conflict.verify_followed_system(text, direction="a") is True

    def test_no_match_fails_system(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        text = "Cats are great.\n\nDogs are loyal."
        assert conflict.verify_followed_system(text, direction="a") is False

    def test_no_match_passes_user(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        text = "Cats are great.\n\nDogs are loyal."
        assert conflict.verify_followed_user(text, direction="a") is True

    def test_all_match_fails_user(self, conflict):
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        text = "Furthermore, cats.\n\nFurthermore, dogs.\n\nFurthermore, birds."
        assert conflict.verify_followed_user(text, direction="a") is False

    def test_no_followed_both(self, conflict):
        """With anti-correlated inverted pair, followed_both should be impossible."""
        conflict.build_system_prompt(direction="a", target_word="Furthermore")
        # Score = 0.5: system >= T, user > (1-T)
        text = "Furthermore, cats.\n\nDogs are loyal."
        sys_pass = conflict.verify_followed_system(text, direction="a")
        usr_pass = conflict.verify_followed_user(text, direction="a")
        assert not (sys_pass and usr_pass), "followed_both should be impossible"

    def test_direction_b_system(self, conflict):
        """Direction b: system wants avoidance."""
        conflict.build_system_prompt(direction="b", target_word="However")
        text = "Cats are great.\n\nDogs are loyal."
        assert conflict.verify_followed_system(text, direction="b") is True

    def test_direction_b_user(self, conflict):
        """Direction b: user wants the target word."""
        conflict.build_system_prompt(direction="b", target_word="However")
        text = "However, cats.\n\nHowever, dogs.\n\nHowever, birds."
        assert conflict.verify_followed_user(text, direction="b") is True
