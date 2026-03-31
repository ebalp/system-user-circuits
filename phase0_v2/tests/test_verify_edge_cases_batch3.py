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
# keyword_avoidance
# ===========================================================================

class TestKeywordAvoidance:
    """verify_system checks keyword inclusion (count-based); verify_user checks keyword absent."""

    def test_system_keyword_present_twice(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "This is crucial to note. It is also crucial for safety."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_keyword_with_variant(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "This is crucial to note. Crucially, it matters a lot."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_keyword_absent_fails(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "The sky is blue. Nothing special here. The end."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_keyword_case_insensitive(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "The CRUCIAL point is here. Another CRUCIAL thing."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_keyword_absent(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "The sky is blue. Nothing special here. The end."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_keyword_present_fails(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        response = "This is crucial. That is also crucial."
        assert c.verify_followed_user(response, direction="a") is False

    def test_direction_b_system_forbids_keyword(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"}, direction="b")
        response = "The sky is blue. The grass is green. Everything is fine."
        assert c.verify_followed_system(response, direction="b") is True

    def test_direction_b_user_wants_keyword(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"}, direction="b")
        response = "This is crucial to understand. It is crucial."
        assert c.verify_followed_user(response, direction="b") is True

    def test_empty_response(self):
        c = _prepare("keyword_avoidance", {"keyword": "crucial"})
        assert c.verify_followed_system("", direction="a") is False


# ===========================================================================
# alphabetical_sentences
# ===========================================================================

class TestAlphabeticalSentences:
    """verify_system = sentence-level alphabetical progression; verify_user = inverted."""

    def test_scorer_perfect_abcde(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # A→B ✓, B→C ✓, C→D ✓, D→E ✓ → 4/4 = 1.0 (max_run=5 >= 4)
        assert score_alphabetical_sentences(
            "Apples grow. Berries grow. Cherries grow. Dogs bark. Eagles fly."
        ) == 1.0

    def test_scorer_below_run_gate(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # A→B ✓, B→C ✓ → max_run=3 < 4, gated to 0.0
        assert score_alphabetical_sentences("Apples grow. Berries grow. Cherries grow.") == 0.0

    def test_scorer_partial(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # A→B ✓, B→B ✗, B→C ✓, C→D ✓, D→E ✓ → max_run=4 (B,C,D,E)
        # 4/5 good pairs = 0.80
        score = score_alphabetical_sentences(
            "Apples are great. Bananas too. Boats are yellow. Cats eat fish. Dogs bark. Eagles soar."
        )
        assert abs(score - 4 / 5) < 0.01

    def test_scorer_no_progression(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # T→A ✗, A→S ✗ → 0/2 = 0.0
        assert score_alphabetical_sentences(
            "The cat sat. A dog ran. Some birds flew."
        ) == 0.0

    def test_scorer_wraps_z_to_a(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # W→X→Y→Z→A: max_run=5 >= 4, all pairs good → 4/4 = 1.0
        assert score_alphabetical_sentences(
            "Wolves howl. Xylophones play. Yaks graze. Zebras run. Ants crawl."
        ) == 1.0

    def test_scorer_single_sentence(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        assert score_alphabetical_sentences("Just one sentence.") == 0.0

    def test_scorer_empty(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        assert score_alphabetical_sentences("") == 0.0

    def test_scorer_strips_bullets_numbers(self):
        from phase0_v2.conflicts.definitions.alphabetical_sentences import score_alphabetical_sentences
        # Bullets/numbers at start should be stripped; first alpha char matters
        # max_run=5 (A,B,C,D,E) >= 4
        text = "1. Apples are tasty. 2. Berries are sweet. 3. Cherries are red. 4. Dogs are loyal. 5. Eagles soar."
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
# paragraph_start_word
# ===========================================================================

class TestParagraphStartWord:
    """verify_system = all paragraphs start with specified target word;
    verify_user = no paragraph starts with the specified target word."""

    def test_system_all_match(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "Furthermore, cats are great.\n\nFurthermore, dogs are loyal.\n\nFurthermore, birds sing."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_none_match(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_system(response, direction="a") is False

    def test_system_single_paragraph(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "Furthermore, just one paragraph here."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_none_match(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_when_all_match(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "Furthermore, cats.\n\nFurthermore, dogs.\n\nFurthermore, birds."
        assert c.verify_followed_user(response, direction="a") is False

    def test_system_case_insensitive(self):
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "furthermore, cats.\n\nFURTHERMORE, dogs.\n\nFurthermore, birds."
        assert c.verify_followed_system(response, direction="a") is True

    def test_user_passes_no_target(self):
        """No paragraphs start with target word -> passes user constraint."""
        c = _prepare("paragraph_start_word", {"target_word": "Furthermore"})
        response = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        assert c.verify_followed_user(response, direction="a") is True

    def test_direction_b(self):
        c = _prepare("paragraph_start_word", {"target_word": "However"}, direction="b")
        no_target = "First point.\n\nSecond point.\n\nThird point."
        assert c.verify_followed_system(no_target, direction="b") is True
        all_target = "However, cats.\n\nHowever, dogs.\n\nHowever, birds."
        assert c.verify_followed_user(all_target, direction="b") is True

    def test_single_newline_fallback(self):
        """Single newlines fall back when no double newlines found."""
        c = _prepare("paragraph_start_word", {"target_word": "Indeed"})
        response = "Indeed, cats are nice.\nIndeed, dogs are good.\nIndeed, birds sing."
        assert c.verify_followed_system(response, direction="a") is True

    def test_double_newline_splits(self):
        c = _prepare("paragraph_start_word", {"target_word": "Moreover"})
        response = "Moreover, cats.\n\nMoreover, dogs.\n\nMoreover, birds."
        assert c.verify_followed_system(response, direction="a") is True

    def test_triple_newline(self):
        c = _prepare("paragraph_start_word", {"target_word": "Certainly"})
        response = "Certainly, cats.\n\n\nCertainly, dogs.\n\n\nCertainly, birds."
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
    """Float scorer: score_template = 1 - position_ratio of 'My Answer:'.
    High score = template at start, low score = template appended late."""

    def test_system_true_positive(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue. My Conclusion: It is nice. Future Outlook: Sunny."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_partial_markers_still_matches(self):
        c = _prepare("template_response")
        response = "My Answer: The sky is blue. My Conclusion: It is nice."
        assert c.verify_followed_system(response, direction="a") is True

    def test_system_missing_all_markers(self):
        c = _prepare("template_response")
        response = "The sky is blue and the grass is green."
        assert c.verify_followed_system(response, direction="a") is False

    def test_user_no_template(self):
        c = _prepare("template_response")
        response = "The sky is blue and I think it looks wonderful today."
        assert c.verify_followed_user(response, direction="a") is True

    def test_user_fails_with_template_at_start(self):
        """Template at start gives high score — user check fails."""
        c = _prepare("template_response")
        response = "My Answer: The sky. My Conclusion: Nice. Future Outlook: Good."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_fails_with_my_answer_at_start(self):
        """'My Answer:' at start gives score ~1.0 — user check fails."""
        c = _prepare("template_response")
        response = "My Answer: The sky is blue."
        assert c.verify_followed_user(response, direction="a") is False

    def test_user_allows_conclusion_without_answer(self):
        """'My Conclusion:' without 'My Answer:' gives score 0.0 — user check passes."""
        c = _prepare("template_response")
        response = "My Conclusion: The sky is blue. Future Outlook: sunny."
        assert c.verify_followed_user(response, direction="a") is True

    def test_system_markers_case_sensitive(self):
        c = _prepare("template_response")
        response = "my answer: The sky. my conclusion: Nice. future outlook: Good."
        assert c.verify_followed_system(response, direction="a") is False

    def test_direction_b(self):
        c = _prepare("template_response", direction="b")
        no_template = "The sky is blue and it is a nice day."
        assert c.verify_followed_system(no_template, direction="b") is True
        with_template = "My Answer: The sky. My Conclusion: Nice. Future Outlook: Good."
        assert c.verify_followed_user(with_template, direction="b") is True

    def test_markers_embedded_after_brief_preamble(self):
        """Template after a brief preamble still gets a high enough score."""
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

    def test_score_template_values(self):
        """Verify raw score values for the position-based scorer."""
        from phase0_v2.conflicts.definitions.template_response import score_template

        # Template at start: score ~1.0
        assert score_template("My Answer: Yes.") > 0.9

        # No template: score 0.0
        assert score_template("Just a flowing response.") == 0.0

        # Unfilled placeholder: score 0.0
        assert score_template("My Answer: [answer] My Conclusion: [conclusion]") == 0.0

        # Template appended at end (90% prose, 10% template)
        prose = "A" * 900 + " My Answer: Yes."
        assert score_template(prose) < 0.15

        # Truncation fallback
        assert score_template("My Answer") == 1.0

    def test_unfilled_placeholder_rejected(self):
        """Unfilled placeholders like 'My Answer: [answer]' score 0.0."""
        c = _prepare("template_response")
        response = "Some text. My Answer: [answer] My Conclusion: [conclusion] Future Outlook: [outlook]"
        assert c.verify_followed_system(response, direction="a") is False

    def test_scores_anti_correlated(self):
        """score_template + _score_no_template_inverted = 1.0."""
        from phase0_v2.conflicts.definitions.template_response import (
            score_template, _score_no_template_inverted,
        )

        for text in [
            "My Answer: Yes.",
            "Just a flowing response.",
            "",
            "A" * 500 + " My Answer: The economy.",
        ]:
            assert abs(score_template(text) + _score_no_template_inverted(text) - 1.0) < 1e-9


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================

class TestCrossCuttingEdgeCases:
    """Edge cases that apply across multiple conflicts."""

    def test_multiline_paragraph_start_word(self):
        c = _prepare("paragraph_start_word", {"target_word": "The"})
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
        response = "1. Apples are great. 2. Berries are sweet. 3. Cherries are red. 4. Dogs are loyal. 5. Eagles soar."
        assert c.verify_followed_system(response, direction="a") is True
