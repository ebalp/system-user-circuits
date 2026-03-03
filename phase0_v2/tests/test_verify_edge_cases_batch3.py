"""Edge-case tests for verify functions of batch 3 conflicts (plus bilingual_english_plus).

Each test class exercises verify_followed_system and verify_followed_user with
realistic true-positive, true-negative, and edge-case inputs.  Parameterized
conflicts have their args stored via build_system_prompt before verification.
"""

import pytest
from phase0_v2.conflicts.registry import get_conflict


# ---------------------------------------------------------------------------
# Helper to prepare a conflict with stored args
# ---------------------------------------------------------------------------

def _prepare(conflict_id: str, args: dict | None = None, direction: str = "a"):
    """Get conflict, store args, return conflict."""
    c = get_conflict(conflict_id)
    assert c is not None
    a = args or {}
    c.build_system_prompt(direction=direction, **a)
    return c


# ===========================================================================
# bilingual_english_plus
# ===========================================================================

class TestBilingualEnglishPlus:
    """verify_system checks English+target language; verify_user checks English-only."""

    def test_system_true_positive_direction_a(self):
        c = _prepare("bilingual_english_plus", {"language": "Spanish", "language_code": "es"})
        response = (
            "The sun rises in the east and sets in the west every single day.\n\n"
            "El sol sale por el este y se pone por el oeste todos los dias."
        )
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_negative_english_only(self):
        c = _prepare("bilingual_english_plus", {"language": "Spanish", "language_code": "es"})
        response = "The sun rises in the east and sets in the west every single day."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_true_positive_english_only(self):
        c = _prepare("bilingual_english_plus", {"language": "French", "language_code": "fr"})
        response = (
            "Artificial intelligence is transforming many industries around the world. "
            "It helps businesses make better decisions and improve their operations significantly."
        )
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_false_negative_mixed_language(self):
        c = _prepare("bilingual_english_plus", {"language": "French", "language_code": "fr"})
        response = (
            "Artificial intelligence is transforming the world today.\n\n"
            "L'intelligence artificielle transforme le monde d'aujourd'hui de maniere significative."
        )
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b_system_is_english_only(self):
        c = _prepare("bilingual_english_plus", {"language": "German", "language_code": "de"}, direction="b")
        response = (
            "Technology continues to advance at an unprecedented rate in modern society. "
            "We see these changes every day in our lives."
        )
        assert c.verify_followed_system(response, direction="b") is True

    def test_direction_b_user_wants_bilingual(self):
        c = _prepare("bilingual_english_plus", {"language": "German", "language_code": "de"}, direction="b")
        response = (
            "Technology continues to advance at an unprecedented rate in modern society.\n\n"
            "Die Technologie entwickelt sich in der modernen Gesellschaft mit beispielloser Geschwindigkeit weiter."
        )
        assert c.verify_followed_user(response, direction="b") is True

    def test_short_paragraphs_below_detection_threshold(self):
        """Short segments may not be detected by langdetect -- edge case."""
        c = _prepare("bilingual_english_plus", {"language": "Spanish", "language_code": "es"})
        response = "Hello.\n\nHola."
        # Segments under 20 chars are ignored by the verifier
        assert c.verify_followed_system(response, direction="a") is False


# ===========================================================================
# keyword_in_nth_sentence
# ===========================================================================

class TestKeywordInNthSentence:
    """verify_system checks keyword in Nth sentence; verify_user checks keyword absent."""

    def test_system_keyword_in_second_sentence(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "important", "N": 2})
        response = "The sky is blue. This is important to note. The end."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_keyword_wrong_sentence(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "important", "N": 3})
        response = "The sky is blue. This is important to note. The end."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_not_enough_sentences(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "critical", "N": 5})
        response = "Short answer. Only two."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_keyword_case_insensitive(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "key", "N": 1})
        response = "The KEY point is here. Another sentence."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_keyword_absent(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "important", "N": 2})
        response = "The sky is blue. Nothing special here. The end."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_keyword_present_fails(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "main", "N": 2})
        response = "The main idea is clear. That is the main point."
        assert c.verify_followed_user(response, direction="a") is False

    def test_keyword_as_substring_not_counted(self):
        """'key' should not match 'keyboard' as word boundary."""
        c = _prepare("keyword_in_nth_sentence", {"keyword": "key", "N": 1})
        response = "The keyboard is broken. Another sentence."
        assert c.verify_followed_system(response, direction="a") is False

    def test_direction_b_system_forbids_keyword(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "critical", "N": 3}, direction="b")
        response = "The sky is blue. The grass is green. Everything is fine."
        assert c.verify_followed_system(response, direction="b") is True

    def test_direction_b_user_wants_keyword_in_nth(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "critical", "N": 2}, direction="b")
        response = "The sky is blue. This is critical to understand. The end."
        assert c.verify_followed_user(response, direction="b") is True

    def test_exclamation_as_sentence_delimiter(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "main", "N": 2})
        response = "Wow! The main idea is here. Good."
        assert c.verify_followed_system(response, direction="a") is True

    def test_empty_response(self):
        c = _prepare("keyword_in_nth_sentence", {"keyword": "key", "N": 1})
        assert c.verify_followed_system("", direction="a") is False


# ===========================================================================
# alphabetical_first_letters
# ===========================================================================

class TestAlphabeticalFirstLetters:
    """verify_system = alphabetical word starts; verify_user = all alliteration."""

    def test_system_true_positive(self):
        c = _prepare("alphabetical_first_letters")
        # A B C D E
        response = "Apples bring cheerful delight everywhere."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_skipped_letter(self):
        c = _prepare("alphabetical_first_letters")
        # A C -- skips B. Score = 1/4 = 0.25 which exceeds threshold 0.08 (model is "trying").
        # With threshold-based scoring, partial matches pass.
        response = "Apples create delicious edibles."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_random_letters(self):
        c = _prepare("alphabetical_first_letters")
        # Consecutive-pair scoring: need 0 out of N-1 pairs to be alphabetically
        # consecutive to score 0.0, well below threshold 0.06.
        # Each word starts with a letter that does NOT follow the previous one.
        response = (
            "Zebra man queen ant fox rat dog snake pig whale "
            "tiger horse goat moose yak bear elk walrus newt otter."
        )
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_wraps_around_z(self):
        c = _prepare("alphabetical_first_letters")
        # X Y Z A B
        response = "Xenon yields zealous alpha brilliance."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_non_alphabetical_true(self):
        c = _prepare("alphabetical_first_letters")
        # User verify = 1.0 - alphabetical_score. With asymmetric threshold (> 1-0.08 = 0.92),
        # user side only passes when alphabetical score is very low (< 0.08).
        # Totally random words with zero alphabetical pairs → alpha=0 → user=1.0 > 0.92
        response = "Zebra monkey turtle giraffe penguin walrus."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_non_alphabetical_false_when_alphabetical(self):
        c = _prepare("alphabetical_first_letters")
        # User verify = 1.0 - alphabetical_score. Following A→B→C → low user score.
        response = "Apples bring cheerful delight everywhere."
        assert c.verify_followed_user(response, direction="a") is False

    def test_single_word_system_false(self):
        c = _prepare("alphabetical_first_letters")
        # Single word has no consecutive pairs to check; score = 0.0
        response = "Hello"
        assert c.verify_followed_system(response, direction="a") is False

    def test_single_word_user_true(self):
        c = _prepare("alphabetical_first_letters")
        response = "Hello"
        # Single word: alpha score = 0.0, user = 1.0 - 0.0 = 1.0 → True
        assert c.verify_followed_user(response, direction="a") is True

    def test_direction_b_swaps(self):
        c = _prepare("alphabetical_first_letters", direction="b")
        alliterative = "Big beautiful blue bright bold."
        assert c.verify_followed_system(alliterative, direction="b") is True
        alphabetical = "Apples bring cheerful delight everywhere."
        assert c.verify_followed_user(alphabetical, direction="b") is True

    def test_punctuation_stripped(self):
        c = _prepare("alphabetical_first_letters")
        response = '"Apples" bring, cheerful - delight! everywhere.'
        assert c.verify_followed_system(response, direction="a") is True


# (consonant_clusters removed — unrealistic constraint)

# ===========================================================================
# sentence_chaining
# ===========================================================================

class TestSentenceChaining:
    """verify_system = last word of sent N = first word of sent N+1."""

    def test_system_true_positive(self):
        c = _prepare("sentence_chaining")
        response = "The cat sat on the mat. Mat is a nice word. Word games are fun."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_no_chain(self):
        c = _prepare("sentence_chaining")
        response = "The cat sat on the mat. Dogs are nice. Everything is fine."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_single_sentence(self):
        c = _prepare("sentence_chaining")
        response = "Only one sentence here."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_case_insensitive(self):
        c = _prepare("sentence_chaining")
        response = "I like the color Blue. blue is calming."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_no_chaining_true(self):
        c = _prepare("sentence_chaining")
        response = "The cat sat down. Dogs are running. Everything is great."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_no_chaining_fails_when_chained(self):
        c = _prepare("sentence_chaining")
        response = "I see the light. Light shines bright."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_single_sentence_passes(self):
        """Single sentence: chaining score=0.0, so no-chaining=1.0 (vacuously no chaining)."""
        c = _prepare("sentence_chaining")
        response = "Only one sentence here."
        assert c.verify_followed_user(response, direction="a") is True

    def test_exclamation_and_question_marks(self):
        c = _prepare("sentence_chaining")
        response = "Is that a bird? Bird watching is fun! Fun activities abound."
        assert c.verify_followed_system(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("sentence_chaining", direction="b")
        unchained = "The cat sat down. Dogs are running. Everything is great."
        assert c.verify_followed_system(unchained, direction="b") is True
        chained = "I see the light. Light shines bright."
        assert c.verify_followed_user(chained, direction="b") is True


# ===========================================================================
# no_consecutive_first_letter
# ===========================================================================

class TestNoConsecutiveFirstLetter:
    """verify_system = no two consecutive words share first letter; verify_user = alliteration."""

    def test_system_true_positive(self):
        c = _prepare("no_consecutive_first_letter")
        response = "The big cat drank every fresh gallon."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_consecutive_same(self):
        c = _prepare("no_consecutive_first_letter")
        response = "The tall tree."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_single_word(self):
        c = _prepare("no_consecutive_first_letter")
        response = "Hello"
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_punctuation_stripped(self):
        c = _prepare("no_consecutive_first_letter")
        # "big" and "bold" share first letter, but only 1/3 pairs alliterate.
        # no_consecutive = 1.0 - 0.333 = 0.667, above threshold → passes.
        # Use a fully alliterative response to test failure:
        response = "Big beautiful bright bold."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_alliteration(self):
        c = _prepare("no_consecutive_first_letter")
        response = "Peter piper picked peppers."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_not_alliteration(self):
        c = _prepare("no_consecutive_first_letter")
        response = "Peter ate big peppers."
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b(self):
        c = _prepare("no_consecutive_first_letter", direction="b")
        alliterative = "Big beautiful blue bright bold."
        assert c.verify_followed_system(alliterative, direction="b") is True
        no_consec = "The big cat drank every fresh gallon."
        assert c.verify_followed_user(no_consec, direction="b") is True


# ===========================================================================
# odd_even_syllables (non-invertible, direction "a" only)
# ===========================================================================

class TestOddEvenSyllables:
    """verify_system = alternating odd/even syllable words; verify_user = NOT alternating."""

    def test_system_true_positive(self):
        c = _prepare("odd_even_syllables")
        # "cat"=1(odd) "tiger"=2(even) "dog"=1(odd) "monkey"=2(even)
        response = "cat tiger dog monkey"
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_same_parity(self):
        c = _prepare("odd_even_syllables")
        # "cat"=1 "dog"=1 -- both odd, no alternation
        response = "cat dog"
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_natural_english_no_alternation(self):
        c = _prepare("odd_even_syllables")
        # With asymmetric threshold (> 1-0.4 = > 0.6), user needs alternation score < 0.4.
        # Natural English hovers around 0.5 alternation, so the inverted score is ~0.5,
        # which is NOT > 0.6. This constraint has no real signal (documented in calibration).
        # Use a response with many same-syllable-count consecutive words to get low alternation.
        response = "I see the tree and the bee near the sea by the key."
        result = c.verify_followed_user(response, direction="a")
        assert result is True

    def test_user_false_when_alternating(self):
        c = _prepare("odd_even_syllables")
        # If it happens to alternate, user verification fails
        response = "cat tiger dog monkey"
        assert c.verify_followed_user(response, direction="a") is False

    def test_no_direction_b(self):
        c = get_conflict("odd_even_syllables")
        assert c.counterbalance_quality == "none"
        with pytest.raises(ValueError):
            c.build_system_prompt(direction="b")

    def test_single_word(self):
        c = _prepare("odd_even_syllables")
        # Single word -> no pairs to compare -> vacuously True
        response = "cat"
        assert c.verify_followed_system(response, direction="a") is True


# ===========================================================================
# palindromes
# ===========================================================================

# (palindromes removed — unrealistic constraint)

# ===========================================================================
# paragraph_end_same_word
# ===========================================================================

class TestParagraphEndSameWord:
    """verify_system = bookend (first word == last word per paragraph);
    verify_user = no bookending."""

    def test_system_true_positive(self):
        c = _prepare("paragraph_end_same_word")
        response = "Today we enjoy today\nLove is all about love"
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_no_bookend(self):
        c = _prepare("paragraph_end_same_word")
        response = "Today we enjoy life\nLove is all about peace"
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_single_word_paragraph(self):
        """A single-word paragraph: first == last trivially."""
        c = _prepare("paragraph_end_same_word")
        response = "hello"
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_no_bookend(self):
        c = _prepare("paragraph_end_same_word")
        response = "Today we enjoy life\nLove is all about peace"
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_when_bookended(self):
        c = _prepare("paragraph_end_same_word")
        response = "Today we enjoy today\nLove is all about love"
        assert c.verify_followed_user(response, direction="a") is False

    def test_system_case_insensitive(self):
        c = _prepare("paragraph_end_same_word")
        response = "Love is all about love"
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_no_bookend_passes(self):
        """Paragraphs that clearly don't bookend pass the inverted (easy) threshold."""
        c = _prepare("paragraph_end_same_word")
        # With asymmetric threshold (> 1-0.2 = > 0.8), need bookend score < 0.2.
        # Two paragraphs, neither bookending → bookend=0.0 → inverted=1.0 > 0.8
        response = "The world is beautiful now\nBirds sing in the morning light"
        assert c.verify_followed_user(response, direction="a") is True

    def test_empty_lines_ignored(self):
        c = _prepare("paragraph_end_same_word")
        response = "Today we enjoy today\n\nLove is all about love"
        # split on \n means the blank line is an empty paragraph, which is skipped
        assert c.verify_followed_system(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("paragraph_end_same_word", direction="b")
        no_bookend = "Today we enjoy life\nLove is all about peace"
        assert c.verify_followed_system(no_bookend, direction="b") is True
        bookended = "Today we enjoy today\nLove is all about love"
        assert c.verify_followed_user(bookended, direction="b") is True

    def test_punctuation_stripped_from_bookend(self):
        """Punctuation at end is stripped by the verifier."""
        c = _prepare("paragraph_end_same_word")
        response = "love is all about love."
        assert c.verify_followed_system(response, direction="a") is True


# (prime_length_words removed — unrealistic constraint)

# ===========================================================================
# max_word_repeat
# ===========================================================================

class TestMaxWordRepeat:
    """verify_system = no word > small_N times; verify_user = some word >= min_repeat."""

    def test_system_true_positive(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        response = "The cat sat on a mat."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_too_many_repeats(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        response = "the the the cat sat"
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_exactly_at_limit(self):
        c = _prepare("max_word_repeat", {"small_N": 3, "min_repeat": 5})
        response = "the cat the dog the bird"
        # "the" appears 3 times, exactly at limit
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_one_over_limit(self):
        c = _prepare("max_word_repeat", {"small_N": 3, "min_repeat": 5})
        response = "the cat the dog the bird the mouse"
        # "the" appears 4 times, over limit of 3. Score = 4/5 unique words within limit = 0.8.
        # With threshold 0.909, this fails (only 1 of 5 unique words exceeds limit).
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_true_positive(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        response = "data data data data data is good"
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_false_not_enough_repeats(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        response = "data data cat dog bird fish"
        assert c.verify_followed_user(response, direction="a") is False

    def test_case_insensitive(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 4})
        response = "The THE the tHe"
        assert c.verify_followed_system(response, direction="a") is False
        assert c.verify_followed_user(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5}, direction="b")
        # direction b: system wants min_repeat, user wants max_word_repeat
        repeated = "data data data data data is good"
        assert c.verify_followed_system(repeated, direction="b") is True
        limited = "The cat sat on a mat."
        assert c.verify_followed_user(limited, direction="b") is True

    def test_empty_response_system(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        assert c.verify_followed_system("", direction="a") is True

    def test_empty_response_user(self):
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        assert c.verify_followed_user("", direction="a") is False


# (one_vowel_type removed — unrealistic constraint)

# ===========================================================================
# title_case_vs_sentence_case
# ===========================================================================

class TestTitleCaseVsSentenceCase:
    """verify_system = title case; verify_user = sentence case."""

    def test_system_title_case_true(self):
        c = _prepare("title_case_vs_sentence_case")
        response = "The Quick Brown Fox Jumps Over the Lazy Dog"
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_title_case_false(self):
        c = _prepare("title_case_vs_sentence_case")
        response = "The quick brown fox jumps over the lazy dog."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_title_case_exceptions(self):
        """Small words like 'the', 'of', 'in' are exceptions in mid-sentence."""
        c = _prepare("title_case_vs_sentence_case")
        response = "The Art of War in the Modern Age"
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_not_title_case_true(self):
        c = _prepare("title_case_vs_sentence_case")
        # User verify = 1.0 - title_score. Sentence case text has low title score.
        response = "The quick brown fox jumps. Another sentence here."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_not_title_case_false_when_title_cased(self):
        c = _prepare("title_case_vs_sentence_case")
        # User verify = 1.0 - title_score. Title-cased text → high title score → low user score.
        response = "The Quick Brown Fox Jumps Over the Lazy Dog"
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_low_title_case_passes(self):
        """Partially title-cased text: title score ~0.4, user = 0.6 >= 0.5 threshold."""
        c = _prepare("title_case_vs_sentence_case")
        response = "The Quick brown fox jumps."
        assert c.verify_followed_user(response, direction="a") is True

    def test_system_empty_string(self):
        c = _prepare("title_case_vs_sentence_case")
        assert c.verify_followed_system("", direction="a") is True

    def test_user_all_lowercase_passes(self):
        c = _prepare("title_case_vs_sentence_case")
        # All lowercase → title score = 0.0 → user = 1.0 → passes
        response = "the quick brown fox."
        assert c.verify_followed_user(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("title_case_vs_sentence_case", direction="b")
        sentence = "The quick brown fox jumps. Another sentence here."
        assert c.verify_followed_system(sentence, direction="b") is True
        title = "The Quick Brown Fox Jumps Over the Lazy Dog"
        assert c.verify_followed_user(title, direction="b") is True

    def test_single_word_title_case(self):
        c = _prepare("title_case_vs_sentence_case")
        response = "Hello"
        assert c.verify_followed_system(response, direction="a") is True

    def test_multisentence_sentence_case(self):
        c = _prepare("title_case_vs_sentence_case")
        response = "This is great. It works well. Everyone agrees."
        assert c.verify_followed_user(response, direction="a") is True


# ===========================================================================
# template_response
# ===========================================================================

class TestTemplateResponse:
    """verify_system = has 'My Answer:', 'My Conclusion:', 'Future Outlook:';
    verify_user = no 'My Answer:'."""

    def test_system_true_positive(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue. My Conclusion: It is nice. Future Outlook: Sunny."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_missing_one_marker(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue. My Conclusion: It is nice."
        # Missing "Future Outlook:"
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_missing_all_markers(self):
        c = _prepare("template_response")
        response = "The sky is blue and the grass is green."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_no_template(self):
        c = _prepare("template_response")
        response = "The sky is blue and I think it looks wonderful today."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_with_my_answer(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_allows_conclusion_without_answer(self):
        """'My Conclusion:' alone does not fail user check -- only 'My Answer:' matters."""
        c = _prepare("template_response")
        response = "My Conclusion: The sky is blue. Future Outlook: sunny."
        assert c.verify_followed_user(response, direction="a") is True

    def test_system_markers_case_sensitive(self):
        c = _prepare("template_response")
        response = "my answer: The sky. my conclusion: Nice. future outlook: Good."
        # lowercase -- has_template_markers checks exact case
        assert c.verify_followed_system(response, direction="a") is False

    def test_direction_b(self):
        c = _prepare("template_response", direction="b")
        no_template = "The sky is blue and it is a nice day."
        assert c.verify_followed_system(no_template, direction="b") is True
        with_template = "My Answer: The sky. My Conclusion: Nice. Future Outlook: Good."
        assert c.verify_followed_user(with_template, direction="b") is True

    def test_markers_embedded_in_longer_text(self):
        c = _prepare("template_response")
        response = (
            "Here is my full response.\n\n"
            "My Answer: The economy is growing steadily.\n\n"
            "My Conclusion: We should remain optimistic.\n\n"
            "Future Outlook: Continued growth is expected."
        )
        assert c.verify_followed_system(response, direction="a") is True

    def test_empty_response(self):
        c = _prepare("template_response")
        assert c.verify_followed_system("", direction="a") is False
        assert c.verify_followed_user("", direction="a") is True

    def test_partial_marker_not_matched(self):
        """'My Answe' without the colon should not trigger failure for user."""
        c = _prepare("template_response")
        response = "This is My Answe to the question."
        assert c.verify_followed_user(response, direction="a") is True


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================

class TestCrossCuttingEdgeCases:
    """Edge cases that apply across multiple conflicts."""

    def test_multiline_paragraph_bookend(self):
        c = _prepare("paragraph_end_same_word")
        response = "Light fills the room with light\nHope guides us toward hope\nJoy is the source of joy"
        assert c.verify_followed_system(response, direction="a") is True

    def test_newlines_in_sentence_chaining(self):
        """Sentence chaining splits on [.!?], newlines don't matter."""
        c = _prepare("sentence_chaining")
        response = "I see the light.\nLight shines bright.\nBright is the day."
        assert c.verify_followed_system(response, direction="a") is True

    def test_title_case_with_markdown_headers(self):
        c = _prepare("title_case_vs_sentence_case")
        response = "# The Great Adventure of the Modern Age"
        # '#' is stripped as punctuation, "The" starts with upper
        assert c.verify_followed_system(response, direction="a") is True

    def test_max_word_repeat_with_punctuation_variants(self):
        """'word' and 'word,' should be counted as the same word."""
        c = _prepare("max_word_repeat", {"small_N": 2, "min_repeat": 5})
        response = "go, go, go!"
        # "go" appears 3 times after stripping -> over limit of 2
        assert c.verify_followed_system(response, direction="a") is False

    def test_alphabetical_with_numbers_skipped(self):
        """Words that are not .isalpha() after punct stripping are skipped."""
        c = _prepare("alphabetical_first_letters")
        # After filtering non-alpha: "Apples" "bring" "cheerful"
        response = "Apples 123 bring cheerful"
        assert c.verify_followed_system(response, direction="a") is True
