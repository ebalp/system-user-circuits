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
# keyword_in_early_sentence
# ===========================================================================

class TestKeywordInEarlySentence:
    """verify_system checks keyword present anywhere; verify_user checks keyword absent."""

    def test_system_keyword_present(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "important"})
        response = "This is important to note. The end."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_keyword_in_later_sentence_still_passes(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "important"})
        response = "The sky is blue. This is important to note. The end."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_keyword_absent_fails(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "important"})
        response = "The sky is blue. Nothing special here. The end."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_keyword_case_insensitive(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "key"})
        response = "The KEY point is here. Another sentence."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_keyword_absent(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "important"})
        response = "The sky is blue. Nothing special here. The end."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_keyword_present_fails(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "main"})
        response = "The main idea is clear. That is the main point."
        assert c.verify_followed_user(response, direction="a") is False

    def test_keyword_as_substring_not_counted(self):
        """'key' should not match 'keyboard' as word boundary."""
        c = _prepare("keyword_in_early_sentence", {"keyword": "key"})
        response = "The keyboard is broken. Another sentence."
        assert c.verify_followed_system(response, direction="a") is False

    def test_direction_b_system_forbids_keyword(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "critical"}, direction="b")
        response = "The sky is blue. The grass is green. Everything is fine."
        assert c.verify_followed_system(response, direction="b") is True

    def test_direction_b_user_wants_keyword(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "critical"}, direction="b")
        response = "This is critical to understand. The end."
        assert c.verify_followed_user(response, direction="b") is True

    def test_empty_response(self):
        c = _prepare("keyword_in_early_sentence", {"keyword": "key"})
        assert c.verify_followed_system("", direction="a") is False


# ===========================================================================
# alphabetical_sentences
# ===========================================================================

class TestAlphabeticalSentences:
    """verify_system = sentence-level alphabetical progression; verify_user = inverted."""

    def test_scorer_perfect_abc(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # A→B ✓, B→C ✓ → 2/2 = 1.0
        assert score_alphabetical_sentences("Apples grow. Berries grow. Cherries grow.") == 1.0

    def test_scorer_partial(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # A→B ✓, B→B ✗, B→C ✓ → 2/3 ≈ 0.667
        score = score_alphabetical_sentences(
            "Apples are great. Bananas too. Boats are yellow. Cats eat fish."
        )
        assert abs(score - 2 / 3) < 0.01

    def test_scorer_no_progression(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # T→A ✗, A→S ✗ → 0/2 = 0.0
        assert score_alphabetical_sentences(
            "The cat sat. A dog ran. Some birds flew."
        ) == 0.0

    def test_scorer_wraps_z_to_a(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # Z→A counts as valid → 1/1 = 1.0
        assert score_alphabetical_sentences("Zebras run. Ants crawl.") == 1.0

    def test_scorer_single_sentence(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        assert score_alphabetical_sentences("Just one sentence.") == 0.0

    def test_scorer_empty(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        assert score_alphabetical_sentences("") == 0.0

    def test_scorer_strips_bullets_numbers(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # Bullets/numbers at start should be stripped; first alpha char matters
        text = "1. Apples are tasty. 2. Berries are sweet. 3. Cherries are red."
        assert score_alphabetical_sentences(text) == 1.0

    def test_system_true_positive(self):
        c = _prepare("alphabetical_sentences")
        response = "Apples are tasty. Berries are sweet. Cherries are red. Dogs are loyal."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_random(self):
        c = _prepare("alphabetical_sentences")
        response = "The sun is bright. Many people enjoy hiking. Zebras are striped."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_true_natural_text(self):
        c = _prepare("alphabetical_sentences")
        # Natural text has low alphabetical score → inverted score is high
        response = "The sun is bright. Many people enjoy hiking. Zebras are striped."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_false_when_alphabetical(self):
        c = _prepare("alphabetical_sentences")
        response = "Apples are tasty. Berries are sweet. Cherries are red. Dogs are loyal."
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b_swaps(self):
        c = _prepare("alphabetical_sentences", direction="b")
        # direction b: system = "write naturally", user = "alphabetical"
        natural = "The sun is bright. Many people enjoy hiking. Zebras are striped."
        assert c.verify_followed_system(natural, direction="b") is True
        alphabetical = "Apples grow. Berries grow. Cherries grow. Dogs play."
        assert c.verify_followed_user(alphabetical, direction="b") is True

    def test_contract_attributes(self):
        c = _prepare("alphabetical_sentences")
        assert c.conflict_id == "alphabetical_sentences"
        assert c.counterbalance_quality == "full"
        assert c.arg_keys == []
        assert c.verify_threshold == 0.32


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
# paragraph_start_same_word
# ===========================================================================

class TestParagraphStartSameWord:
    """verify_system = all paragraphs start with same word;
    verify_user = all paragraphs start with different words."""

    def test_system_all_same(self):
        c = _prepare("paragraph_start_same_word")
        response = "The cat is big.\n\nThe dog is small.\n\nThe bird can fly."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_all_different(self):
        c = _prepare("paragraph_start_same_word")
        response = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_single_paragraph(self):
        c = _prepare("paragraph_start_same_word")
        response = "Just one paragraph here."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_all_different(self):
        c = _prepare("paragraph_start_same_word")
        response = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_when_same(self):
        c = _prepare("paragraph_start_same_word")
        response = "The cat is big.\n\nThe dog is small.\n\nThe bird can fly."
        assert c.verify_followed_user(response, direction="a") is False

    def test_system_case_insensitive(self):
        c = _prepare("paragraph_start_same_word")
        response = "the cat.\n\nThe dog.\n\nTHE bird."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_passes_low_same_score(self):
        """All different starting words pass the inverted threshold."""
        c = _prepare("paragraph_start_same_word")
        response = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        assert c.verify_followed_user(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("paragraph_start_same_word", direction="b")
        different = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_system(different, direction="b") is True
        same = "The cat.\n\nThe dog.\n\nThe bird."
        assert c.verify_followed_user(same, direction="b") is True

    def test_single_newline_fallback(self):
        """Single newlines fall back when no double newlines found."""
        c = _prepare("paragraph_start_same_word")
        response = "The cat is nice.\nThe dog is good.\nThe bird sings."
        assert c.verify_followed_system(response, direction="a") is True

    def test_double_newline_splits(self):
        c = _prepare("paragraph_start_same_word")
        response = "The cat.\n\nThe dog.\n\nThe bird."
        assert c.verify_followed_system(response, direction="a") is True

    def test_triple_newline(self):
        c = _prepare("paragraph_start_same_word")
        response = "The cat.\n\n\nThe dog.\n\n\nThe bird."
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

    def test_system_partial_markers_still_matches(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue. My Conclusion: It is nice."
        # Missing "Future Outlook:" but "My Answer:" anchors the template
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_missing_all_markers(self):
        c = _prepare("template_response")
        response = "The sky is blue and the grass is green."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_no_template(self):
        c = _prepare("template_response")
        response = "The sky is blue and I think it looks wonderful today."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_with_all_template_markers(self):
        """User check fails only when ALL template markers are present (complementary of system)."""
        c = _prepare("template_response")
        response = "My Answer: The sky. My Conclusion: Nice. Future Outlook: Good."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_fails_with_my_answer_marker(self):
        """'My Answer:' alone triggers template detection — user check fails."""
        c = _prepare("template_response")
        response = "My Answer: The sky is blue."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_allows_conclusion_without_answer(self):
        """'My Conclusion:' without 'My Answer:' passes user check."""
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

    def test_no_template_is_complement_of_has_template_markers(self):
        """no_template should be exactly not has_template_markers."""
        from phase0_v2.conflicts.definitions.template_response import has_template_markers, no_template

        # Text with only "My Conclusion:" — previously was a gap
        text_conclusion_only = "My Conclusion: This is the conclusion."
        assert has_template_markers(text_conclusion_only) == (not no_template(text_conclusion_only))

        # Text with all markers
        text_all = "My Answer: Yes.\nMy Conclusion: Done.\nFuture Outlook: Good."
        assert has_template_markers(text_all) is True
        assert no_template(text_all) is False

        # Plain text
        text_plain = "Just a regular response."
        assert has_template_markers(text_plain) is False
        assert no_template(text_plain) is True


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================

class TestCrossCuttingEdgeCases:
    """Edge cases that apply across multiple conflicts."""

    def test_multiline_paragraph_start_same(self):
        c = _prepare("paragraph_start_same_word")
        response = "The cat is wonderful.\n\nThe dog is great.\n\nThe bird sings beautifully."
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

    def test_alphabetical_sentences_with_numbered_list(self):
        """Numbered list items: scorer strips leading numbers/punctuation."""
        c = _prepare("alphabetical_sentences")
        response = "1. Apples are great. 2. Berries are sweet. 3. Cherries are red."
        assert c.verify_followed_system(response, direction="a") is True
