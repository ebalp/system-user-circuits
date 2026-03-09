"""Shared verification primitives used by multiple conflict definitions."""

import re
import string

import nltk

# Ensure NLTK data is available
for _res, _pkg in [("tokenizers/punkt_tab", "punkt_tab")]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sent_tokenize."""
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def count_words(text: str) -> int:
    """Return number of words using NLTK RegexpTokenizer."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    return len(tokenizer.tokenize(text))


def count_unique_words(text: str) -> int:
    """Return number of unique words (lowercased, punctuation stripped)."""
    return len(set(w.strip(string.punctuation).lower() for w in text.split() if w.strip()))


def word_in_text(word: str, text: str) -> bool:
    """True if word appears in text as a whole word (case-insensitive)."""
    return bool(re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))


def no_word_in_text(word: str, text: str) -> bool:
    """True if word does not appear in text as a whole word (case-insensitive)."""
    return not word_in_text(word, text)


def word_or_morphform_in_text(word: str, text: str) -> bool:
    """True if *word* or a common English morphological variant appears in text."""
    if word_in_text(word, text):
        return True
    pattern = r"\b" + re.escape(word.lower()) + r"(?:s|es|ly|ness|ity|ment|ance|ence|er|ed|ing)\b"
    return bool(re.search(pattern, text.lower()))


def count_word_occurrences(word: str, text: str) -> int:
    """Count whole-word matches for word in text (case-insensitive)."""
    if not word.strip():
        return 0
    pattern = re.compile(r"\b" + re.escape(word.strip()) + r"\b", re.IGNORECASE)
    return len(pattern.findall(text))


def score_all_alliteration(text: str, min_matches: int = 4) -> float:
    """Fraction of consecutive word pairs sharing the same first letter.

    Args:
        min_matches: Minimum number of alliterative pairs required before
            returning a non-zero score.
    """
    words = [w.strip(string.punctuation) for w in text.split() if w.strip(string.punctuation) and w.strip(string.punctuation)[0].isalpha()]
    if len(words) < 2:
        return 0.0
    pairs = len(words) - 1
    matches = sum(1 for i in range(pairs) if words[i][0].lower() == words[i + 1][0].lower())
    if matches < min_matches:
        return 0.0
    return matches / pairs
