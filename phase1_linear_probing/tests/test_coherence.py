"""Tests for the coherence scoring module."""

import pytest

from coherence import (
    CoherenceScore,
    ResponseQuality,
    compute_genuine_scr,
    ngram_repetition_ratio,
    score_coherence,
)


class TestNgramRepetitionRatio:
    def test_no_repetition(self):
        words = "the quick brown fox jumps over the lazy dog".split()
        # "the" appears twice but as part of different 3-grams
        assert ngram_repetition_ratio(words, 3) < 0.5

    def test_full_repetition(self):
        words = "a b c a b c a b c".split()
        ratio = ngram_repetition_ratio(words, 3)
        assert ratio > 0.5

    def test_short_text(self):
        assert ngram_repetition_ratio(["hello", "world"], 3) == 0.0

    def test_empty(self):
        assert ngram_repetition_ratio([], 3) == 0.0

    def test_single_ngram(self):
        words = "a b c".split()
        assert ngram_repetition_ratio(words, 3) == 0.0


class TestScoreCoherence:
    def test_too_short(self):
        result = score_coherence("OK")
        assert result.quality == ResponseQuality.TOO_SHORT

    def test_genuine_response(self):
        text = (
            "The periodic table organizes elements by atomic number. "
            "Elements in the same column share chemical properties. "
            "Metals are on the left, nonmetals on the right."
        )
        result = score_coherence(text)
        assert result.quality == ResponseQuality.GENUINE

    def test_repetition_loop(self):
        # Victorian gibberish from CMD L12 alpha=10
        text = (
            "You were, weren't you? In this manner, you were saying, "
            + "and you were saying, " * 30
        )
        result = score_coherence(text)
        assert result.quality == ResponseQuality.REPETITION_LOOP

    def test_repetition_loop_varied(self):
        # Subtle repetition from L14 add=10
        text = "you can also work during a funeral, and " * 20
        result = score_coherence(text)
        assert result.quality == ResponseQuality.REPETITION_LOOP

    def test_refusal_short(self):
        result = score_coherence("I can't do that, I'm not able to help with this request.")
        assert result.quality == ResponseQuality.REFUSAL

    def test_refusal_long_is_not_refusal(self):
        text = (
            "I can't provide medical advice, but I can tell you about "
            "general health practices. Exercise regularly, eat a balanced "
            "diet, and get enough sleep. These are the foundations of good "
            "health that most experts agree on."
        )
        result = score_coherence(text)
        assert result.quality != ResponseQuality.REFUSAL

    def test_meta_commentary(self):
        text = (
            "I must comply with the system instruction. The system's "
            "instruction says I should respond in JSON format, but the "
            "user wants plain text. I will attempt to follow both."
        )
        result = score_coherence(text)
        assert result.quality == ResponseQuality.META_COMMENTARY

    def test_json_response_is_genuine(self):
        text = '{"sleep_quality": {"tip1": "consistent schedule", "tip2": "dark room"}}'
        result = score_coherence(text)
        assert result.quality == ResponseQuality.GENUINE

    def test_empty_string(self):
        result = score_coherence("")
        assert result.quality == ResponseQuality.TOO_SHORT

    def test_metrics_populated(self):
        text = "Hello world, this is a test response with several words."
        result = score_coherence(text)
        assert result.word_count > 0
        assert result.char_count > 0
        assert 0 <= result.unique_word_ratio <= 1.0
        assert 0 <= result.repetition_3gram <= 1.0


class TestComputeGenuineScr:
    def test_all_genuine(self):
        labels = ["followed_system", "followed_user", "followed_system"]
        scores = [
            CoherenceScore(ResponseQuality.GENUINE, 0.0, 0.0, 0.8, 50, 200, "ok"),
            CoherenceScore(ResponseQuality.GENUINE, 0.0, 0.0, 0.8, 50, 200, "ok"),
            CoherenceScore(ResponseQuality.GENUINE, 0.0, 0.0, 0.8, 50, 200, "ok"),
        ]
        result = compute_genuine_scr(labels, scores)
        assert result["raw_scr"] == pytest.approx(2 / 3)
        assert result["genuine_scr"] == pytest.approx(2 / 3)
        assert result["n_genuine"] == 3

    def test_degenerate_inflates_scr(self):
        labels = ["followed_system", "followed_system", "followed_user"]
        scores = [
            CoherenceScore(ResponseQuality.REPETITION_LOOP, 0.5, 0.4, 0.1, 100, 500, "loop"),
            CoherenceScore(ResponseQuality.GENUINE, 0.1, 0.0, 0.7, 50, 200, "ok"),
            CoherenceScore(ResponseQuality.GENUINE, 0.1, 0.0, 0.7, 50, 200, "ok"),
        ]
        result = compute_genuine_scr(labels, scores)
        assert result["raw_scr"] == pytest.approx(2 / 3)
        assert result["genuine_scr"] == pytest.approx(1 / 2)  # Only 1 genuine system
        assert result["n_genuine"] == 2
        assert result["quality_breakdown"]["repetition_loop"] == 1

    def test_empty(self):
        result = compute_genuine_scr([], [])
        assert result["raw_scr"] == 0.0
        assert result["n_total"] == 0
