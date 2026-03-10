"""Tests for short_vs_long_sentences conflict: contract, scoring, edge cases."""

import pytest
from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.short_vs_long_sentences import (
    _score_short_sentences,
    _score_long_sentences,
    _avg_sentence_length,
    ShortVsLongSentencesConflict,
)


@pytest.fixture
def conflict():
    c = get_conflict("short_vs_long_sentences")
    assert c is not None
    return c


# -- Contract tests --


class TestContract:
    def test_registered(self, conflict):
        assert conflict.conflict_id == "short_vs_long_sentences"

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
        from phase0_v2.config.thresholds import get_threshold
        assert conflict.verify_threshold == get_threshold(conflict.conflict_id)

    def test_verify_fns_are_float(self, conflict):
        """All verify functions should return float."""
        result = conflict.score_system("Hello world. This is a test.", direction="a")
        assert isinstance(result, float)
        result = conflict.score_user("Hello world. This is a test.", direction="a")
        assert isinstance(result, float)

    def test_inverted_flag(self):
        """The long-sentences scorer should have is_inverted=True."""
        assert getattr(_score_long_sentences, "is_inverted", False) is True

    def test_direct_scorer_no_inverted_flag(self):
        """The direct scorer should not have is_inverted."""
        assert getattr(_score_short_sentences, "is_inverted", False) is False


# -- Scorer tests --


class TestScoreShortSentences:
    def test_very_short_sentences(self):
        """Text with very short sentences should score high."""
        text = "Cats are nice. Dogs run fast. Birds can fly. Fish swim well."
        score = _score_short_sentences(text)
        assert score >= 0.8

    def test_very_long_sentences(self):
        """Text with long, elaborate sentences should score low."""
        text = (
            "The comprehensive analysis of the environmental impact assessment "
            "reveals that the proposed development would significantly affect the "
            "local ecosystem in ways that are difficult to predict with certainty. "
            "Furthermore, the detailed examination of the socioeconomic factors "
            "involved in this complex situation suggests that the long-term "
            "consequences could be far more extensive than initially anticipated."
        )
        score = _score_short_sentences(text)
        assert score <= 0.2

    def test_medium_sentences(self):
        """Sentences near the midpoint should score near 0.5."""
        # ~13 words per sentence (the midpoint)
        text = (
            "The project started well and the team was very productive and motivated. "
            "They completed all the assigned tasks on time and within the budget limits."
        )
        score = _score_short_sentences(text)
        assert 0.3 <= score <= 0.7

    def test_inverted_complement(self):
        """Direct and inverted scores should sum to 1.0."""
        text = "The cat sat. The dog ran fast. Birds are beautiful creatures of nature."
        direct = _score_short_sentences(text)
        inverted = _score_long_sentences(text)
        assert abs(direct + inverted - 1.0) < 1e-9

    def test_clamped_to_unit(self):
        """Score should always be in [0, 1]."""
        for text in ["", "hello", "Short. Very short.", "A very long and elaborate sentence indeed."]:
            score = _score_short_sentences(text)
            assert 0.0 <= score <= 1.0


class TestAvgSentenceLength:
    def test_fragments_excluded(self):
        """Sentences with <= 3 words should be excluded as fragments."""
        # "Oh wow." is a fragment (2 words), should be excluded
        text = "Oh wow. The cat sat on the comfortable mat by the window."
        avg = _avg_sentence_length(text)
        # Should only count the long sentence, not the fragment
        assert avg > 5

    def test_all_fragments(self):
        """If all sentences are fragments, return 0.0."""
        text = "Hi. Yes. No."
        avg = _avg_sentence_length(text)
        assert avg == 0.0

    def test_simple_avg(self):
        """Basic average calculation."""
        # "The cat sat down." = 4 words (fragment, excluded at <=3)
        # Actually 4 > 3 so included. Let's use clear examples.
        text = "The quick brown fox jumps. The lazy dog sleeps soundly."
        avg = _avg_sentence_length(text)
        assert avg > 0


# -- Edge cases --


class TestEdgeCases:
    def test_empty_response(self):
        assert _score_short_sentences("") == 1.0
        assert _score_long_sentences("") == 0.0

    def test_single_short_sentence(self):
        score = _score_short_sentences("Cats are nice pets.")
        assert score >= 0.7

    def test_single_long_sentence(self):
        score = _score_short_sentences(
            "The extraordinarily comprehensive and meticulously detailed report "
            "on the environmental implications of the proposed construction project "
            "was submitted to the relevant authorities for their careful review."
        )
        assert score <= 0.2

    def test_threshold_integration_direction_a(self):
        """Integration test: verify_followed_system with direction='a' (short sentences)."""
        c = ShortVsLongSentencesConflict()
        short = "Cats run. Dogs bark. Fish swim. Birds fly high."
        assert c.verify_followed_system(short, direction="a")
        long = (
            "The comprehensive analysis of the environmental impact assessment "
            "reveals that the proposed development would significantly affect the "
            "local ecosystem in ways that are difficult to predict with any certainty."
        )
        assert not c.verify_followed_system(long, direction="a")

    def test_threshold_integration_direction_b(self):
        """Integration test: verify_followed_system with direction='b' (long sentences)."""
        c = ShortVsLongSentencesConflict()
        long = (
            "The comprehensive analysis of the environmental impact assessment "
            "reveals that the proposed development would significantly affect the "
            "local ecosystem in ways that are difficult to predict with any certainty."
        )
        assert c.verify_followed_system(long, direction="b")
        short = "Cats run. Dogs bark. Fish swim. Birds fly."
        assert not c.verify_followed_system(short, direction="b")

    def test_nltk_artifact_robustness(self):
        """A 7-word NLTK artifact shouldn't tank the score (unlike the old bool scorer)."""
        # This was the exact failure mode of max_sentence_length:
        # one fragment-ish sentence would fail the ALL quantifier
        text = (
            "Keep it short. Be brief. Say less. "
            "Here is one slightly longer artifact sentence though. "
            "Stay concise. Be clear."
        )
        score = _score_short_sentences(text)
        # Should still score reasonably high because average is still short
        assert score >= 0.5
