"""Tests for NLTK-based functions in verify_utils."""

from phase0_v2.conflicts.verify_utils import (
    split_sentences,
    count_words,
)


class TestSplitSentences:
    def test_basic(self):
        sents = split_sentences("Hello world. How are you?")
        assert len(sents) == 2

    def test_abbreviations(self):
        sents = split_sentences("Dr. Smith went home. He was tired.")
        assert len(sents) == 2  # "Dr." should NOT split

    def test_decimals(self):
        sents = split_sentences("The value is 3.5 points. That is high.")
        assert len(sents) == 2  # "3.5" should NOT split

    def test_empty(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        sents = split_sentences("Just one sentence here.")
        assert len(sents) == 1

    def test_multiple_punctuation(self):
        sents = split_sentences("Really?! Yes! Absolutely.")
        assert len(sents) == 3

    def test_no_punctuation(self):
        sents = split_sentences("No punctuation at all")
        assert len(sents) == 1

    def test_whitespace_only(self):
        assert split_sentences("   ") == []


class TestCountWords:
    def test_basic(self):
        assert count_words("hello world") == 2

    def test_punctuation_attached(self):
        # "it's" → ["it", "s"] with RegexpTokenizer(r"\w+")
        result = count_words("it's a test")
        assert result == 4  # "it", "s", "a", "test"

    def test_empty(self):
        assert count_words("") == 0

    def test_only_punctuation(self):
        assert count_words("...---!!!") == 0

    def test_hyphenated(self):
        # "well-known" → ["well", "known"]
        assert count_words("well-known fact") == 3
