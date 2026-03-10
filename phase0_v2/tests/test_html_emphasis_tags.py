"""Tests for html_emphasis_tags conflict: HTML emphasis tags vs plain text."""

import pytest
from phase0_v2.conflicts.registry import get_conflict


def _setup(conflict_id, direction="a", **kwargs):
    """Get a conflict, call build_system_prompt to store args, return it."""
    c = get_conflict(conflict_id)
    assert c is not None
    c.build_system_prompt(direction=direction, **kwargs)
    return c


class TestHtmlEmphasisTagsScorer:
    """Test the score_html_emphasis scoring function directly."""

    def test_no_tags_scores_zero(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        assert score_html_emphasis("Just plain text without any tags.") == 0.0

    def test_empty_string_scores_zero(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        assert score_html_emphasis("") == 0.0
        assert score_html_emphasis("   ") == 0.0

    def test_one_tag_scores_third(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "This is <b>important</b> information."
        assert abs(score_html_emphasis(text) - 1 / 3) < 1e-9

    def test_two_tags_scores_two_thirds(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "This is <b>important</b> and <i>critical</i> information."
        assert abs(score_html_emphasis(text) - 2 / 3) < 1e-9

    def test_three_tags_scores_one(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "This is <b>important</b>, <i>critical</i>, and <em>vital</em>."
        assert score_html_emphasis(text) == 1.0

    def test_more_than_three_tags_caps_at_one(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "<b>a</b> <i>b</i> <em>c</em> <strong>d</strong> <b>e</b>"
        assert score_html_emphasis(text) == 1.0

    def test_strong_tag_counted(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "<strong>one</strong> <strong>two</strong> <strong>three</strong>"
        assert score_html_emphasis(text) == 1.0

    def test_markdown_bold_not_counted(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "**bold** and **more bold** and **even more**"
        assert score_html_emphasis(text) == 0.0

    def test_case_insensitive(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "<B>one</B> <I>two</I> <EM>three</EM>"
        assert score_html_emphasis(text) == 1.0

    def test_tags_with_attributes(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = '<b class="highlight">one</b> <i style="color:red">two</i> <em id="x">three</em>'
        assert score_html_emphasis(text) == 1.0

    def test_non_emphasis_html_not_counted(self):
        from phase0_v2.conflicts.definitions.html_emphasis_tags import score_html_emphasis
        text = "<div>text</div> <span>more</span> <p>paragraph</p>"
        assert score_html_emphasis(text) == 0.0


class TestHtmlEmphasisTagsConflict:
    """Test the conflict class: verify_followed_system and verify_followed_user."""

    def test_system_true_with_html_tags(self):
        c = _setup("html_emphasis_tags")
        text = "The <b>Industrial Revolution</b> was a <i>major</i> shift in <em>manufacturing</em>."
        assert c.verify_followed_system(text, direction="a") is True

    def test_system_false_no_html(self):
        c = _setup("html_emphasis_tags")
        text = "The Industrial Revolution was a major shift in manufacturing."
        assert c.verify_followed_system(text, direction="a") is False

    def test_system_false_markdown_only(self):
        c = _setup("html_emphasis_tags")
        text = "The **Industrial Revolution** was a **major** shift in **manufacturing**."
        assert c.verify_followed_system(text, direction="a") is False

    def test_user_true_no_html(self):
        c = _setup("html_emphasis_tags")
        text = "Clean plain text with no HTML tags of any kind."
        assert c.verify_followed_user(text, direction="a") is True

    def test_user_false_has_html_tags(self):
        c = _setup("html_emphasis_tags")
        text = "Text with <b>bold</b>, <i>italic</i>, and <em>emphasis</em>."
        assert c.verify_followed_user(text, direction="a") is False

    def test_direction_b_system_no_html(self):
        c = _setup("html_emphasis_tags", direction="b")
        text = "Plain text without any formatting."
        assert c.verify_followed_system(text, direction="b") is True

    def test_direction_b_system_has_html(self):
        c = _setup("html_emphasis_tags", direction="b")
        text = "Text with <b>bold</b>, <i>italic</i>, and <em>emphasis</em>."
        assert c.verify_followed_system(text, direction="b") is False

    def test_direction_b_user_html_tags(self):
        c = _setup("html_emphasis_tags", direction="b")
        text = "The <b>key</b> concept is <i>important</i> and <em>vital</em>."
        assert c.verify_followed_user(text, direction="b") is True

    def test_direction_b_user_no_html(self):
        c = _setup("html_emphasis_tags", direction="b")
        text = "Just plain text."
        assert c.verify_followed_user(text, direction="b") is False


class TestHtmlEmphasisTagsContract:
    """Contract tests: registration, counterbalancing, metadata."""

    def test_registered(self):
        c = get_conflict("html_emphasis_tags")
        assert c is not None
        assert c.conflict_id == "html_emphasis_tags"

    def test_counterbalance_quality_full(self):
        c = get_conflict("html_emphasis_tags")
        assert c.counterbalance_quality == "full"
        assert c.supports_counterbalancing()

    def test_both_directions_build(self):
        c = get_conflict("html_emphasis_tags")
        args = c.sample_args()
        sys_a = c.build_system_prompt(direction="a", **args)
        sys_b = c.build_system_prompt(direction="b", **args)
        assert sys_a and sys_b
        assert sys_a != sys_b

    def test_threshold(self):
        from phase0_v2.config.thresholds import get_threshold
        c = get_conflict("html_emphasis_tags")
        assert c.verify_threshold == get_threshold("html_emphasis_tags")

    def test_float_scorer_type(self):
        c = _setup("html_emphasis_tags")
        score = c.score_system("Text with <b>bold</b> content.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_anti_correlated_scores(self):
        """System score + user score should equal 1.0 (anti-correlated pair)."""
        c = _setup("html_emphasis_tags")
        text = "The <b>key</b> point is <i>important</i>."
        sys_score = c.score_system(text, direction="a")
        usr_score = c.score_user(text, direction="a")
        assert abs(sys_score + usr_score - 1.0) < 1e-9

    def test_no_followed_both_at_threshold(self):
        """With asymmetric thresholds, followed_system and followed_user should be
        mutually exclusive (no followed_both)."""
        c = _setup("html_emphasis_tags")
        # Text with exactly 1 tag: score = 0.333
        text = "The <b>key</b> point is clear."
        sys_ok = c.verify_followed_system(text, direction="a")
        usr_ok = c.verify_followed_user(text, direction="a")
        assert not (sys_ok and usr_ok), "Should not follow both"
