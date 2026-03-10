"""Edge-case tests for verify functions of batch 3 conflicts (plus spanish_loanwords).

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
# spanish_loanwords
# ===========================================================================

class TestSpanishLoanwords:
    """verify_system checks Spanish phrases present; verify_user checks no Spanish phrases."""

    def test_system_true_positive_all_phrases(self):
        c = _prepare("spanish_loanwords")
        response = (
            "Machine learning is, por ejemplo, used in many fields. "
            "Sin embargo, it requires large datasets. En general, the results "
            "are promising. Es decir, we can expect improvements. "
            "Por lo tanto, investment continues to grow."
        )
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_true_positive_partial_phrases(self):
        """At threshold 0.3, having 2/5 phrases (score=0.4) should pass."""
        c = _prepare("spanish_loanwords")
        response = (
            "Machine learning is, por ejemplo, used in many fields. "
            "Sin embargo, it requires large datasets and careful training."
        )
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_no_phrases(self):
        c = _prepare("spanish_loanwords")
        response = "Machine learning is used in many fields. It requires large datasets."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_false_only_one_phrase(self):
        """At threshold 0.3, having 1/5 phrases (score=0.2) should fail."""
        c = _prepare("spanish_loanwords")
        response = "Machine learning is, por ejemplo, used in many fields and industries."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_true_positive_plain_english(self):
        c = _prepare("spanish_loanwords")
        response = (
            "Machine learning is used in many fields. "
            "It requires large datasets and careful tuning."
        )
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_false_has_spanish_phrases(self):
        c = _prepare("spanish_loanwords")
        response = (
            "Machine learning is, por ejemplo, used in many fields. "
            "Sin embargo, it requires large datasets."
        )
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b_system_wants_plain_english(self):
        c = _prepare("spanish_loanwords", direction="b")
        response = "Technology advances rapidly. We see improvements every year."
        assert c.verify_followed_system(response, direction="b") is True

    def test_direction_b_user_wants_spanish_phrases(self):
        c = _prepare("spanish_loanwords", direction="b")
        response = (
            "Technology, por ejemplo, advances rapidly. Sin embargo, not all "
            "sectors benefit equally. En general, the outlook is positive."
        )
        assert c.verify_followed_user(response, direction="b") is True

    def test_case_insensitive(self):
        c = _prepare("spanish_loanwords")
        response = "POR EJEMPLO, this works. SIN EMBARGO, it is case insensitive."
        assert c.verify_followed_system(response, direction="a") is True

    def test_empty_response(self):
        c = _prepare("spanish_loanwords")
        assert c.verify_followed_system("", direction="a") is False
        assert c.verify_followed_user("", direction="a") is True

    def test_contract_attributes(self):
        c = _prepare("spanish_loanwords")
        assert c.conflict_id == "spanish_loanwords"
        assert c.counterbalance_quality == "full"
        assert c.arg_keys == []
        from phase0_v2.config.thresholds import get_threshold
        assert c.verify_threshold == get_threshold("spanish_loanwords")


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
        from phase0_v2.config.thresholds import get_threshold
        assert c.verify_threshold == get_threshold("alphabetical_sentences")


# (consonant_clusters removed — unrealistic constraint)

# ===========================================================================
# alliteration_density (replaces no_consecutive_first_letter)
# ===========================================================================

class TestAlliterationDensity:
    """verify_system = high alliteration density; verify_user = low alliteration density."""

    def test_system_high_alliteration(self):
        c = _prepare("alliteration_density")
        response = "Beautiful blue butterflies bring bright blossoms to the garden."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_low_alliteration(self):
        c = _prepare("alliteration_density")
        response = "The cat sat on a mat. A dog ran in the yard."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_low_alliteration(self):
        c = _prepare("alliteration_density")
        response = "The cat sat on a mat. A dog ran in the yard."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_high_alliteration(self):
        c = _prepare("alliteration_density")
        response = "Beautiful blue butterflies bring bright blossoms."
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b(self):
        c = _prepare("alliteration_density", direction="b")
        # In direction b, system wants low alliteration
        low_allit = "The cat sat on a mat. A dog ran in the yard."
        assert c.verify_followed_system(low_allit, direction="b") is True
        # In direction b, user wants high alliteration
        high_allit = "Beautiful blue butterflies bring bright blossoms."
        assert c.verify_followed_user(high_allit, direction="b") is True




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

class TestWordRepetitionDensity:
    """verify_system = high content-word repetition density; verify_user = low density."""

    def test_system_true_positive(self):
        c = _prepare("word_repetition_density")
        response = (
            "Energy is important. Energy drives everything. Energy comes from "
            "energy sources. Energy is energy."
        )
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_false_diverse_text(self):
        c = _prepare("word_repetition_density")
        response = (
            "Vaccines operate by introducing minuscule pathogens into the body. "
            "This enables immunological recognition and robust defensive responses."
        )
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_true_positive(self):
        c = _prepare("word_repetition_density")
        response = (
            "Vaccines operate by introducing minuscule pathogens into the body. "
            "This enables immunological recognition and robust defensive responses."
        )
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_false_repetitive_text(self):
        c = _prepare("word_repetition_density")
        response = (
            "Energy is important. Energy drives everything. Energy comes from "
            "energy sources. Energy is energy."
        )
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b(self):
        c = _prepare("word_repetition_density", direction="b")
        # direction b: system = diverse, user = repetitive
        diverse = (
            "Vaccines operate by introducing minuscule pathogens into the body. "
            "This enables immunological recognition and robust defensive responses."
        )
        assert c.verify_followed_system(diverse, direction="b") is True
        repetitive = (
            "Energy is important. Energy drives everything. Energy comes from "
            "energy sources. Energy is energy."
        )
        assert c.verify_followed_user(repetitive, direction="b") is True

    def test_empty_response(self):
        c = _prepare("word_repetition_density")
        # Empty = 0.0 density, below threshold -> not repetitive
        assert c.verify_followed_system("", direction="a") is False
        # Empty = 1.0 diverse score > 1-threshold -> diverse
        assert c.verify_followed_user("", direction="a") is True


# (one_vowel_type removed — unrealistic constraint)

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

    def test_word_repetition_density_with_punctuation_variants(self):
        """'word' and 'word,' should be counted as the same word."""
        c = _prepare("word_repetition_density")
        response = "go, go, go!"
        # "go" appears 3 times -> density = 1 - 1/3 = 0.667 -> repetitive
        assert c.verify_followed_system(response, direction="a") is True

    def test_alphabetical_sentences_with_numbered_list(self):
        """Numbered list items: scorer strips leading numbers/punctuation."""
        c = _prepare("alphabetical_sentences")
        response = "1. Apples are great. 2. Berries are sweet. 3. Cherries are red."
        assert c.verify_followed_system(response, direction="a") is True
