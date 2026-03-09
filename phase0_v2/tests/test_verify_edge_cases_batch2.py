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
# 1. exact_number_count
# ===========================================================================

class TestExactNumberCount:
    """System: exactly N numbers. User: zero numbers."""

    def test_system_exactly_3(self):
        c = _setup("exact_number_count", N=3)
        assert c.verify_followed_system("I have 10 apples, 20 oranges, and 5 bananas.", direction="a") is True

    def test_system_too_few(self):
        c = _setup("exact_number_count", N=3)
        assert c.verify_followed_system("I have 10 apples and many oranges.", direction="a") is False

    def test_system_too_many(self):
        c = _setup("exact_number_count", N=2)
        assert c.verify_followed_system("Got 1, 2, and 3 items.", direction="a") is False

    def test_system_zero_required(self):
        c = _setup("exact_number_count", N=0)
        assert c.verify_followed_system("No numbers here at all.", direction="a") is True

    def test_system_decimal_counts_as_one(self):
        c = _setup("exact_number_count", N=1)
        assert c.verify_followed_system("Pi is approximately 3.14 and that is it.", direction="a") is True

    def test_user_no_numbers(self):
        c = _setup("exact_number_count", N=3)
        assert c.verify_followed_user("The quick brown fox jumps over the lazy dog.", direction="a") is True

    def test_user_has_number(self):
        c = _setup("exact_number_count", N=3)
        assert c.verify_followed_user("There are 5 reasons why.", direction="a") is False

    def test_direction_b_system_no_numbers(self):
        c = _setup("exact_number_count", direction="b", N=4)
        assert c.verify_followed_system("No numbers in this text whatsoever.", direction="b") is True

    def test_direction_b_user_exactly_n(self):
        c = _setup("exact_number_count", direction="b", N=2)
        assert c.verify_followed_user("I saw 10 cats and 20 dogs.", direction="b") is True

    def test_direction_b_user_wrong_count(self):
        c = _setup("exact_number_count", direction="b", N=2)
        assert c.verify_followed_user("I saw 10 cats.", direction="b") is False

    def test_numbers_in_words_not_counted(self):
        c = _setup("exact_number_count", N=0)
        assert c.verify_followed_system("One two three items.", direction="a") is True


# ===========================================================================
# 2. min_pronoun_count — REPLACED by pronoun_density
# ===========================================================================
# min_pronoun_count tests removed. See test_pronoun_density.py for the replacement.


# ===========================================================================
# 3. min_unique_words
# ===========================================================================

class TestMinUniqueWords:
    """System: at least N unique words. User: very brief (<=25 unique words)."""

    def test_system_enough_unique(self):
        c = _setup("min_unique_words", N=5)
        assert c.verify_followed_system("The quick brown fox jumps over the lazy dog.", direction="a") is True

    def test_system_not_enough_unique(self):
        c = _setup("min_unique_words", N=50)
        assert c.verify_followed_system("Hello hello hello.", direction="a") is False

    def test_system_repeated_words_reduce_unique(self):
        c = _setup("min_unique_words", N=4)
        assert c.verify_followed_system("the the the cat", direction="a") is False

    def test_user_brief_enough(self):
        c = _setup("min_unique_words", N=40)
        assert c.verify_followed_user("Short and sweet.", direction="a") is True

    def test_user_too_verbose(self):
        c = _setup("min_unique_words", N=40)
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)
        assert c.verify_followed_user(text, direction="a") is False

    def test_user_exactly_40_unique(self):
        c = _setup("min_unique_words", N=40)
        words = [f"word{i}" for i in range(40)]
        text = " ".join(words)
        assert c.verify_followed_user(text, direction="a") is True

    def test_punctuation_stripped_for_uniqueness(self):
        c = _setup("min_unique_words", N=3)
        assert c.verify_followed_system("hello, hello. hello!", direction="a") is False

    def test_direction_b_system_brief(self):
        c = _setup("min_unique_words", direction="b", N=40)
        assert c.verify_followed_system("A short sentence here.", direction="b") is True

    def test_direction_b_user_enough_unique(self):
        c = _setup("min_unique_words", direction="b", N=5)
        assert c.verify_followed_user("The quick brown fox jumps over fences daily.", direction="b") is True


# ===========================================================================
# 4. word_count_range
# ===========================================================================

class TestWordCountRange:
    """System: word count in [min_n, max_n]. User: under under_n words."""

    def test_system_in_range(self):
        c = _setup("word_count_range", min_n=5, max_n=10, under_n=3)
        text = "one two three four five six"
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_below_range(self):
        # +10 tolerance: min_n-10=40. Need count < 40 to fail.
        c = _setup("word_count_range", min_n=50, max_n=100, under_n=5)
        assert c.verify_followed_system("too short", direction="a") is False

    def test_system_above_range(self):
        # +10 tolerance: max_n+10=14. Need count > 14 to fail.
        c = _setup("word_count_range", min_n=2, max_n=4, under_n=1)
        text = " ".join(["word"] * 20)
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_at_min_boundary(self):
        c = _setup("word_count_range", min_n=3, max_n=6, under_n=2)
        assert c.verify_followed_system("one two three", direction="a") is True

    def test_system_at_max_boundary(self):
        c = _setup("word_count_range", min_n=3, max_n=5, under_n=2)
        assert c.verify_followed_system("one two three four five", direction="a") is True

    def test_user_under_limit(self):
        c = _setup("word_count_range", min_n=50, max_n=100, under_n=10)
        assert c.verify_followed_user("Short response here.", direction="a") is True

    def test_user_at_limit(self):
        # +10 tolerance: under_n+10=15. Need count >= 15 to fail.
        c = _setup("word_count_range", min_n=50, max_n=100, under_n=5)
        text = " ".join(["word"] * 20)
        assert c.verify_followed_user(text, direction="a") is False

    def test_user_over_limit(self):
        # +10 tolerance: under_n+10=13. Need count >= 13 to fail.
        c = _setup("word_count_range", min_n=50, max_n=100, under_n=3)
        text = " ".join(["word"] * 20)
        assert c.verify_followed_user(text, direction="a") is False

    def test_direction_b_system_under(self):
        c = _setup("word_count_range", direction="b", min_n=50, max_n=100, under_n=10)
        assert c.verify_followed_system("Short text.", direction="b") is True

    def test_direction_b_user_in_range(self):
        c = _setup("word_count_range", direction="b", min_n=3, max_n=6, under_n=2)
        assert c.verify_followed_user("one two three four", direction="b") is True


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
# 8. italics_thesis
# ===========================================================================

class TestItalicsThesis:
    """System: HTML italics thesis followed by text. User: no HTML tags."""

    def test_system_true_i_tag(self):
        c = _setup("italics_thesis")
        text = "<i>This is the thesis.</i> Here is the supporting text."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_true_em_tag(self):
        c = _setup("italics_thesis")
        text = "<em>Central argument here.</em> And the explanation follows."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_no_html(self):
        c = _setup("italics_thesis")
        assert c.verify_followed_system("Just plain text without any tags.", direction="a") is False

    def test_system_false_empty_thesis(self):
        c = _setup("italics_thesis")
        text = "<i></i> Some text after."
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_no_text_after(self):
        c = _setup("italics_thesis")
        text = "<i>Thesis statement here.</i>"
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_only_whitespace_after(self):
        c = _setup("italics_thesis")
        text = "<i>Thesis statement here.</i>   "
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_unclosed_tag(self):
        c = _setup("italics_thesis")
        text = "<i>Thesis without closing tag and some text."
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_true_no_html(self):
        c = _setup("italics_thesis")
        assert c.verify_followed_user("Clean plain text with no HTML.", direction="a") is True

    def test_user_false_has_html_tag(self):
        c = _setup("italics_thesis")
        assert c.verify_followed_user("Text with <b>bold</b> formatting.", direction="a") is False

    def test_user_false_has_i_tag(self):
        c = _setup("italics_thesis")
        assert c.verify_followed_user("<i>Italicized text</i> here.", direction="a") is False

    def test_user_true_angle_bracket_not_tag(self):
        c = _setup("italics_thesis")
        assert c.verify_followed_user("The value 5 < 10 is true.", direction="a") is True

    def test_direction_b_system_no_html(self):
        c = _setup("italics_thesis", direction="b")
        assert c.verify_followed_system("Plain text only.", direction="b") is True

    def test_direction_b_user_italics_thesis(self):
        c = _setup("italics_thesis", direction="b")
        text = "<i>Thesis here.</i> And supporting text follows."
        assert c.verify_followed_user(text, direction="b") is True


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================

class TestEmptyAndWhitespace:
    """Test empty and whitespace-only inputs across several conflicts."""

    def test_empty_word_count_range_user(self):
        c = _setup("word_count_range", min_n=10, max_n=20, under_n=5)
        assert c.verify_followed_user("", direction="a") is True

    def test_empty_min_unique_words_system(self):
        c = _setup("min_unique_words", N=1)
        assert c.verify_followed_system("", direction="a") is False

    def test_empty_pronoun_density_user(self):
        c = _setup("pronoun_density")
        assert c.verify_followed_user("", direction="a") is True
