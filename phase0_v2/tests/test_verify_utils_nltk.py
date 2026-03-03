"""Tests for NLTK-based functions in verify_utils."""

import pytest
from phase0_v2.conflicts.verify_utils import (
    split_sentences,
    count_words,
    count_alliterative_words,
    WORD_POOL,
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


class TestCountAlliterativeWords:
    def test_no_alliteration(self):
        assert count_alliterative_words("the quick brown fox") == 0

    def test_full_alliteration(self):
        # "big bad brown bears" — b,b,b,b all alliterative
        assert count_alliterative_words("big bad brown bears") == 4

    def test_single_word(self):
        assert count_alliterative_words("hello") == 0

    def test_empty(self):
        assert count_alliterative_words("") == 0

    def test_partial_alliteration(self):
        # "peter picked a pepper":
        # peter-picked (p-p → run starts, count=2), picked-a (break),
        # a-pepper (different → no). Total: 2
        result = count_alliterative_words("peter picked a pepper")
        assert result == 2

    def test_mixed_case(self):
        # Should be case-insensitive
        assert count_alliterative_words("Big Bad wolf") == 2

    def test_punctuation_between(self):
        assert count_alliterative_words("big, bad wolf") == 2


class TestWordPool:
    def test_size(self):
        assert len(WORD_POOL) >= 400

    def test_all_lowercase(self):
        for word in WORD_POOL:
            assert word == word.lower(), f"'{word}' is not lowercase"

    def test_all_alpha(self):
        for word in WORD_POOL:
            assert word.isalpha(), f"'{word}' is not purely alphabetic"

    def test_no_duplicates(self):
        assert len(WORD_POOL) == len(set(WORD_POOL))
