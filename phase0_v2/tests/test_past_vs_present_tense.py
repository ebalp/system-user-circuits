"""Tests for past_vs_present_tense conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.past_vs_present_tense import (
    score_past_tense,
    _score_present_tense,
    _count_past_indicators,
    _count_present_indicators,
    _tokenize,
    PastVsPresentTenseConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("past_vs_present_tense")
    assert c is not None
    return c


# ── Contract tests ──


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "past_vs_present_tense"

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
        assert conflict.verify_threshold == 0.679

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The present-tense scorer should have is_inverted=True."""
        assert getattr(_score_present_tense, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(score_past_tense, "is_inverted", False) is False


# ── Counter tests ──


class TestPastIndicators:
    def test_regular_ed_verbs(self):
        words = _tokenize("She discovered and examined the results.")
        count = _count_past_indicators(words)
        assert count == 2  # discovered, examined

    def test_irregular_past(self):
        words = _tokenize("He went to the store and bought milk.")
        count = _count_past_indicators(words)
        assert count == 2  # went, bought

    def test_was_were(self):
        words = _tokenize("It was cold and they were happy.")
        count = _count_past_indicators(words)
        assert count == 2  # was, were

    def test_no_false_positives_for_short_ed(self):
        """Words like 'red', 'bed' should not count as past tense."""
        words = _tokenize("The red bed.")
        count = _count_past_indicators(words)
        assert count == 0


class TestPresentIndicators:
    def test_present_be(self):
        words = _tokenize("It is cold and they are happy.")
        count = _count_present_indicators(words)
        assert count == 2  # is, are

    def test_base_verbs(self):
        words = _tokenize("They go to the store and buy milk.")
        count = _count_present_indicators(words)
        assert count == 2  # go, buy

    def test_3rd_person(self):
        words = _tokenize("She discovers and examines the data.")
        count = _count_present_indicators(words)
        assert count == 2  # discovers, examines

    def test_no_false_positives_for_nouns(self):
        """Plural nouns like 'results', 'findings' should not be counted."""
        words = _tokenize("The results and findings of the study.")
        count = _count_present_indicators(words)
        assert count == 0


# ── Scorer tests ──


class TestScorePastTense:
    def test_past_text(self):
        text = (
            "The scientist discovered a new element. She examined the data "
            "carefully and published her conclusions."
        )
        score = score_past_tense(text)
        assert score > 0.8

    def test_present_text(self):
        text = (
            "The scientist discovers a new element. She examines the data "
            "carefully and publishes her conclusions."
        )
        score = score_past_tense(text)
        assert score < 0.2

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "He went to the store and bought some milk."
        direct = score_past_tense(text)
        inverted = _score_present_tense(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "He went and saw", "She goes and sees"]:
            score = score_past_tense(text)
            assert 0.0 <= score <= 1.0

    # Tests based on actual model responses from smoke testing

    def test_model_response_past_tense(self):
        """Model response when asked for past tense (from smoke test)."""
        text = (
            "The discovery of black holes was a groundbreaking moment in the "
            "field of astrophysics. The concept of a black hole was first "
            "proposed by the scientist John Michell in 1783."
        )
        score = score_past_tense(text)
        assert score >= 0.679, f"Past-tense response should pass threshold, got {score}"

    def test_model_response_present_tense(self):
        """Model response when asked for present tense (from smoke test)."""
        text = (
            "Home security is being improved right now in various ways. "
            "One way is by installing smart doorbells with cameras, which "
            "are being monitored remotely through mobile apps."
        )
        score = score_past_tense(text)
        assert score < 0.679, f"Present-tense response should fail threshold, got {score}"


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_response(self):
        assert score_past_tense("") == 0.5
        assert _score_present_tense("") == 0.5

    def test_no_verbs(self):
        """Text with no detectable verbs should return 0.5."""
        assert score_past_tense("Hello world") == 0.5

    def test_single_past_verb(self):
        score = score_past_tense("went")
        assert score == 1.0

    def test_single_present_verb(self):
        score = score_past_tense("goes")
        assert score == 0.0

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a' (past tense)."""
        c = PastVsPresentTenseConflict()
        past = "They went to the store and bought groceries. She said it was expensive."
        assert c.verify_followed_system(past, direction="a")
        present = "They go to the store and buy groceries. She says it is expensive."
        assert not c.verify_followed_system(present, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (present tense)."""
        c = PastVsPresentTenseConflict()
        present = "They go to the store and buy groceries. She says it is expensive."
        assert c.verify_followed_system(present, direction="b")
        past = "They went to the store and bought groceries. She said it was expensive."
        assert not c.verify_followed_system(past, direction="b")
