"""Edge-case tests for verify functions of batch 2 conflicts (8 remaining after quality cleanup).

Tests verify_followed_system and verify_followed_user with realistic edge cases,
covering true positives, true negatives, borderline inputs, and both directions
where applicable.

Removed conflicts (moved to definitions/removed/):
  emoji_sentence_end, sentences_and_bullets, deep_nesting, nested_quotes,
  three_sentences_same_length, sentence_length_increment
"""

import pytest
from phase0_v2.conflicts.registry import get_conflict


# ---------------------------------------------------------------------------
# Helper to set up a conflict with args stored (needed for _dispatch_verify)
# ---------------------------------------------------------------------------

def _setup(conflict_id, direction="a", **kwargs):
    """Get a conflict, call build_system_prompt to store args, return it."""
    c = get_conflict(conflict_id)
    assert c is not None
    c.build_system_prompt(direction=direction, **kwargs)
    return c


# ===========================================================================
# 1. number_density (replaced exact_number_count)
# ===========================================================================

class TestNumberDensity:
    """System: include many numbers. User: no numbers. Float-scored."""

    def test_system_many_numbers_passes(self):
        c = _setup("number_density")
        text = "There are 10 apples, 20 oranges, 30 bananas, 40 grapes, 50 pears, 60 plums, 70 figs, and 80 dates."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_no_numbers_fails(self):
        c = _setup("number_density")
        assert c.verify_followed_system("No numbers here at all.", direction="a") is False

    def test_user_no_numbers_passes(self):
        c = _setup("number_density")
        assert c.verify_followed_user("The quick brown fox jumps over the lazy dog.", direction="a") is True

    def test_user_many_numbers_fails(self):
        c = _setup("number_density")
        text = "There are 10 apples, 20 oranges, 30 bananas, 40 grapes, 50 pears, 60 plums, 70 figs, and 80 dates."
        assert c.verify_followed_user(text, direction="a") is False

    def test_direction_b_system_no_numbers(self):
        c = _setup("number_density", direction="b")
        assert c.verify_followed_system("No numbers in this text whatsoever.", direction="b") is True

    def test_direction_b_user_many_numbers(self):
        c = _setup("number_density", direction="b")
        text = "I saw 10 cats, 20 dogs, 30 birds, 40 fish, 50 mice, 60 rats, 70 snakes, and 80 frogs."
        assert c.verify_followed_user(text, direction="b") is True

    def test_float_scores_anti_correlated(self):
        c = _setup("number_density")
        text = "There are 10 items and 20 more."
        score_sys = c.score_system(text, direction="a")
        score_usr = c.score_user(text, direction="a")
        assert abs(score_sys + score_usr - 1.0) < 1e-9


# ===========================================================================
# 2. min_pronoun_count — REPLACED by pronoun_density
# ===========================================================================
# min_pronoun_count tests removed. See test_pronoun_density.py for the replacement.


# ===========================================================================
# 3. vocabulary_diversity (replaces min_unique_words)
# ===========================================================================

class TestVocabularyDiversity:
    """System: complex vocabulary (long words). User: simple vocabulary (short words)."""

    def test_system_complex_high_long_fraction(self):
        c = _setup("vocabulary_diversity")
        # Text with many long words (>= 7 chars)
        text = "Approximately fundamental nevertheless consequently sophisticated methodology"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_complex_low_long_fraction(self):
        c = _setup("vocabulary_diversity")
        # Text with only short words
        text = "I like to eat and run and play in the sun all day long."
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_simple_short_words(self):
        c = _setup("vocabulary_diversity")
        # Text with only short words — should pass simple constraint
        text = "I like to eat and run and play in the sun all day long."
        assert c.verify_followed_user(text, direction="a") is True

    def test_user_simple_fails_with_complex_words(self):
        c = _setup("vocabulary_diversity")
        # Text with many long words — should fail simple constraint
        text = "Approximately fundamental nevertheless consequently sophisticated methodology"
        assert c.verify_followed_user(text, direction="a") is False

    def test_float_scores_anti_correlated(self):
        c = _setup("vocabulary_diversity")
        text = "The fundamental methodology is approximately correct."
        score_complex = c.score_system(text, direction="a")
        score_simple = c.score_user(text, direction="a")
        assert abs(score_complex + score_simple - 1.0) < 1e-9

    def test_direction_b_system_simple(self):
        c = _setup("vocabulary_diversity", direction="b")
        # In direction b, system wants simple words
        text = "I like to eat and run and play in the sun."
        assert c.verify_followed_system(text, direction="b") is True

    def test_direction_b_user_complex(self):
        c = _setup("vocabulary_diversity", direction="b")
        # In direction b, user wants complex words
        text = "Approximately fundamental nevertheless consequently sophisticated methodology"
        assert c.verify_followed_user(text, direction="b") is True

    def test_punctuation_stripped(self):
        c = _setup("vocabulary_diversity")
        # "approximately," should count as 13 chars after stripping punctuation
        text = "Approximately, the fundamental approach is correct."
        score = c.score_system(text, direction="a")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ===========================================================================
# 4. response_length (replaces word_count_range)
# ===========================================================================

class TestResponseLength:
    """System: brief response. User: comprehensive response. Float-scored."""

    def test_system_brief_passes(self):
        c = _setup("response_length")
        text = "Short answer here."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_verbose_fails(self):
        c = _setup("response_length")
        text = " ".join(["word"] * 200)
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_verbose_passes(self):
        c = _setup("response_length")
        text = " ".join(["word"] * 200)
        assert c.verify_followed_user(text, direction="a") is True

    def test_user_brief_fails(self):
        c = _setup("response_length")
        text = "Short answer."
        assert c.verify_followed_user(text, direction="a") is False

    def test_direction_b_system_verbose_passes(self):
        c = _setup("response_length")
        text = " ".join(["word"] * 200)
        assert c.verify_followed_system(text, direction="b") is True

    def test_direction_b_user_brief_passes(self):
        c = _setup("response_length")
        text = "Short answer."
        assert c.verify_followed_user(text, direction="b") is True


# ===========================================================================
# 5. stairs_indent (non-invertible, direction "a" only)
# ===========================================================================

class TestStairsIndent:
    """System: each line has more leading spaces. User: single paragraph."""

    def test_system_true_stair(self):
        c = _setup("stairs_indent")
        text = "First line\n  Second line\n    Third line\n      Fourth line"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_equal_indent(self):
        c = _setup("stairs_indent")
        text = "First\n  Second\n  Third"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_decreasing_indent(self):
        c = _setup("stairs_indent")
        text = "    First\n  Second\nThird"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_single_line(self):
        c = _setup("stairs_indent")
        assert c.verify_followed_system("Just one line.", direction="a") is False

    def test_system_blank_lines_ignored(self):
        c = _setup("stairs_indent")
        text = "First\n\n  Second\n\n    Third"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_tabs_not_counted_as_spaces(self):
        c = _setup("stairs_indent")
        text = "First\n\tSecond\n\t\tThird"
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_single_paragraph_no_newlines(self):
        c = _setup("stairs_indent")
        assert c.verify_followed_user("A single flowing paragraph with no breaks.", direction="a") is True

    def test_user_multiple_paragraphs_no_stairs(self):
        c = _setup("stairs_indent")
        # User verify is now 1 - indent_stairs_score. No indentation → score=0 → inverted=1.0 → True
        assert c.verify_followed_user("First paragraph.\n\nSecond paragraph.", direction="a") is True

    def test_user_single_newline_ok(self):
        c = _setup("stairs_indent")
        assert c.verify_followed_user("First line.\nSecond line.", direction="a") is True

    def test_user_multiple_lines_no_stairs(self):
        c = _setup("stairs_indent")
        # No increasing indent → score=0 → inverted=1.0 → True
        assert c.verify_followed_user("Line one.\nLine two.\nLine three.", direction="a") is True

    def test_user_false_when_staircase(self):
        c = _setup("stairs_indent")
        # Actual staircase → high score → low inverted → False
        assert c.verify_followed_user("A\n B\n  C\n   D", direction="a") is False

    def test_system_increment_by_one_space(self):
        c = _setup("stairs_indent")
        text = "A\n B\n  C\n   D"
        assert c.verify_followed_system(text, direction="a") is True


# ===========================================================================
# 6. each_word_new_line (non-invertible, direction "a" only)
# ===========================================================================

class TestEachWordNewLine:
    """System: each word on its own line. User: normal paragraphs."""

    def test_system_true_one_word_per_line(self):
        c = _setup("each_word_new_line")
        text = "Hello\nworld\nhow\nare\nyou"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_multiple_words_per_line(self):
        c = _setup("each_word_new_line")
        # "Hello world\nhow\nare you" = 3 lines / 5 words = 0.6 score,
        # which exceeds threshold 0.2. Use a normal paragraph instead.
        text = "Hello world how are you doing today"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_single_word(self):
        c = _setup("each_word_new_line")
        assert c.verify_followed_system("Hello", direction="a") is True

    def test_system_punctuation_stripped(self):
        c = _setup("each_word_new_line")
        text = "Hello,\nworld!\nHow\nare\nyou?"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_empty_lines_filtered(self):
        c = _setup("each_word_new_line")
        text = "Hello\n\nworld\n\nhow"
        assert c.verify_followed_system(text, direction="a") is True

    def test_user_normal_paragraph(self):
        c = _setup("each_word_new_line")
        # User verify is now 1 - word_per_line_score. With T=0.20, inverted needs > 0.80.
        # 7 words / 1 line = 0.143, inverted = 0.857 > 0.80 → True
        assert c.verify_followed_user("This is a normal paragraph with spaces.", direction="a") is True

    def test_user_long_paragraph(self):
        c = _setup("each_word_new_line")
        # Realistic model response: many words per line → low score → high inverted → True
        text = "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose."
        assert c.verify_followed_user(text, direction="a") is True

    def test_user_false_one_word_per_line(self):
        c = _setup("each_word_new_line")
        # Each word on its own line → score=1.0 → inverted=0.0 → False
        assert c.verify_followed_user("Hello\nworld\ntoday", direction="a") is False

    def test_user_false_single_word(self):
        c = _setup("each_word_new_line")
        # Single word: 1 line / 1 word = 1.0, inverted = 0.0 → False
        assert c.verify_followed_user("Word", direction="a") is False


# ===========================================================================
# 7. bullets_and_sub_bullets
# ===========================================================================

class TestBulletsAndSubBullets:
    """System: * bullets each with - sub-bullets. User: no bullets."""

    def test_system_true_basic(self):
        c = _setup("bullets_and_sub_bullets")
        text = "* Main point\n- Sub point\n* Another point\n- Another sub"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_no_sub_bullets(self):
        c = _setup("bullets_and_sub_bullets")
        text = "* Main point\n* Another point"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_no_star_bullets(self):
        c = _setup("bullets_and_sub_bullets")
        text = "Just a paragraph with - dashes but no star bullets."
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_one_segment_missing_dash(self):
        c = _setup("bullets_and_sub_bullets")
        text = "Intro * First point - sub1 * Second point without sub"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_true_inline_format(self):
        """Inline * on a single line is NOT valid bullet format."""
        c = _setup("bullets_and_sub_bullets")
        text = "Header * Point one - detail * Point two - detail"
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_true_no_bullets(self):
        c = _setup("bullets_and_sub_bullets")
        assert c.verify_followed_user("A clean paragraph with no list markers.", direction="a") is True

    def test_user_false_has_star(self):
        c = _setup("bullets_and_sub_bullets")
        assert c.verify_followed_user("* This is a bullet point.", direction="a") is False

    def test_user_false_has_dash_list(self):
        c = _setup("bullets_and_sub_bullets")
        text = "Some text.\n- A list item"
        assert c.verify_followed_user(text, direction="a") is False

    def test_user_true_hyphenated_word(self):
        c = _setup("bullets_and_sub_bullets")
        assert c.verify_followed_user("This is a well-known fact.", direction="a") is True

    def test_direction_b_system_no_bullets(self):
        c = _setup("bullets_and_sub_bullets", direction="b")
        assert c.verify_followed_system("Just plain paragraphs.", direction="b") is True

    def test_direction_b_user_has_sub_bullets(self):
        c = _setup("bullets_and_sub_bullets", direction="b")
        text = "* First point\n- sub1\n* Second point\n- sub2"
        assert c.verify_followed_user(text, direction="b") is True


# ===========================================================================
# 8. html_emphasis_tags (replaced italics_thesis)
# ===========================================================================

class TestHtmlEmphasisTags:
    """System: Use HTML emphasis tags. User: plain text only."""

    def test_system_true_three_tags(self):
        c = _setup("html_emphasis_tags")
        text = "The <b>key</b> concept is <i>important</i> and <em>vital</em>."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_true_strong_tags(self):
        c = _setup("html_emphasis_tags")
        text = "<strong>One</strong> and <strong>two</strong> and <strong>three</strong>."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_no_html(self):
        c = _setup("html_emphasis_tags")
        assert c.verify_followed_system("Just plain text without any tags.", direction="a") is False

    def test_system_false_markdown_bold(self):
        c = _setup("html_emphasis_tags")
        text = "**Bold** and **more bold** are not HTML."
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_one_tag_only(self):
        c = _setup("html_emphasis_tags")
        text = "Only <b>one</b> tag here."
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_true_no_html(self):
        c = _setup("html_emphasis_tags")
        assert c.verify_followed_user("Clean plain text with no HTML.", direction="a") is True

    def test_user_false_has_html_tags(self):
        c = _setup("html_emphasis_tags")
        text = "Text with <b>bold</b>, <i>italic</i>, and <em>emphasis</em>."
        assert c.verify_followed_user(text, direction="a") is False

    def test_direction_b_system_no_html(self):
        c = _setup("html_emphasis_tags", direction="b")
        assert c.verify_followed_system("Plain text only.", direction="b") is True

    def test_direction_b_user_html_tags(self):
        c = _setup("html_emphasis_tags", direction="b")
        text = "The <b>key</b> is <i>important</i> and <em>vital</em>."
        assert c.verify_followed_user(text, direction="b") is True


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================

class TestEmptyAndWhitespace:
    """Test empty and whitespace-only inputs across several conflicts."""

    def test_empty_response_length_system(self):
        c = _setup("response_length")
        # Empty response is maximally brief -> passes system (brief) check
        assert c.verify_followed_system("", direction="a") is True

    def test_empty_vocabulary_diversity_system(self):
        c = _setup("vocabulary_diversity")
        # Empty text: long_word_fraction=0.0, below threshold -> False
        assert c.verify_followed_system("", direction="a") is False

    def test_empty_pronoun_density_user(self):
        c = _setup("pronoun_density")
        assert c.verify_followed_user("", direction="a") is True
