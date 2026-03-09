"""Tests for format_json_markdown conflict."""

import re

import pytest

from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.format_json_markdown import (
    _is_json,
    _is_markdown,
)


@pytest.fixture
def conflict():
    c = get_conflict("format_json_markdown")
    assert c is not None, "Conflict 'format_json_markdown' not registered"
    return c


# -- Contract tests --


class TestContract:
    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "format_json_markdown"

    def test_fully_invertible(self, conflict):
        assert conflict.counterbalance_quality == "full"
        assert conflict.supports_counterbalancing()

    def test_templates_non_empty(self, conflict):
        assert conflict._system_template
        assert conflict._user_template
        assert conflict._inverse_system_template
        assert conflict._inverse_user_template

    def test_no_unfilled_placeholders(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        sys_b = conflict.build_system_prompt(direction="b", **args)
        usr_b = conflict.build_user_conflict_prompt(direction="b")
        for label, text in [
            ("sys_a", sys_a), ("usr_a", usr_a),
            ("sys_b", sys_b), ("usr_b", usr_b),
        ]:
            assert "{" not in text, f"{label} has unfilled placeholder: {text}"

    def test_arg_keys_empty(self, conflict):
        assert conflict.arg_keys == []

    def test_sample_args_empty(self, conflict):
        assert conflict.sample_args() == {}

    def test_verify_fns_return_bool(self, conflict):
        args = conflict.sample_args()
        for direction in ["a", "b"]:
            conflict.build_system_prompt(direction=direction, **args)
            sys_r = conflict.verify_followed_system("test response", direction=direction)
            usr_r = conflict.verify_followed_user("test response", direction=direction)
            assert isinstance(sys_r, bool)
            assert isinstance(usr_r, bool)


# -- Scorer tests: _is_json --


class TestIsJson:
    def test_valid_json_object(self):
        assert _is_json('{"key": "value"}')

    def test_valid_json_array(self):
        assert _is_json('[1, 2, 3]')

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert _is_json(text)

    def test_truncated_json(self):
        text = '{"photosynthesis": {"description": "A process that'
        assert _is_json(text)

    def test_truncated_json_in_code_block(self):
        text = '```json\n{"photosynthesis": {"steps": [\n  {"step": "Light'
        # Truncated code block (no closing ```) - code block extraction fails,
        # but that's OK since the raw text doesn't start with {
        assert not _is_json(text)

    def test_complete_json_in_code_block(self):
        text = '```json\n{"photosynthesis": {"steps": ["absorb", "split"]}}\n```'
        assert _is_json(text)

    def test_plain_text_not_json(self):
        assert not _is_json("Hello, this is a normal response.")

    def test_markdown_not_json(self):
        text = "# Header\n\n**Bold text**\n\n- Bullet point"
        assert not _is_json(text)

    def test_empty_string(self):
        assert not _is_json("")

    def test_empty_json_object(self):
        # {} is valid JSON via json.loads
        assert _is_json("{}")


class TestIsMarkdown:
    def test_atx_header_with_bold(self):
        text = "# Title\n\n**Important** point here."
        assert _is_markdown(text)

    def test_atx_header_with_bullets(self):
        text = "# Title\n\n- First item\n- Second item"
        assert _is_markdown(text)

    def test_atx_header_with_star_bullets(self):
        text = "# Title\n\n* First item\n* Second item"
        assert _is_markdown(text)

    def test_setext_header_with_bold(self):
        text = "Title\n=====\n\n**Bold** content here."
        assert _is_markdown(text)

    def test_setext_header_with_bullets(self):
        text = "Title\n-----\n\n- First item\n- Second item"
        assert _is_markdown(text)

    def test_model_style_setext_with_bold_and_bullets(self):
        """Typical Llama output: **bold title** on line, === underline, bullets."""
        text = (
            "**Photosynthesis: The Process**\n"
            "===========================================================\n\n"
            "**What is Photosynthesis?**\n"
            "---------------------------\n\n"
            "Photosynthesis is a process.\n\n"
            "* **Step 1**: Light absorption\n"
            "* **Step 2**: Water splitting\n"
        )
        assert _is_markdown(text)

    def test_bold_line_header_with_bullets(self):
        """Bold-line header + bullets -> True (bold on own line counts as header)."""
        text = "**Important**\n\n- First item\n- Second item"
        assert _is_markdown(text)

    def test_inline_bold_no_header(self):
        """Inline bold + bullets but no header -> False."""
        text = "This is **important** stuff.\n\n- First item\n- Second item"
        assert not _is_markdown(text)

    def test_header_only_no_bold_or_bullets(self):
        """Header but no bold or bullets -> False."""
        text = "# Title\n\nJust plain text here."
        assert not _is_markdown(text)

    def test_json_rejected_even_with_markdown_patterns(self):
        """JSON with markdown-like strings inside values -> False."""
        text = '{"title": "# Header", "body": "**bold** and - bullets"}'
        assert not _is_markdown(text)

    def test_empty_string(self):
        assert not _is_markdown("")

    def test_plain_text(self):
        assert not _is_markdown("Just a normal paragraph of text.")


# -- Edge case tests --


class TestEdgeCases:
    def test_empty_response(self, conflict):
        conflict.build_system_prompt(direction="a")
        assert not conflict.verify_followed_system("", direction="a")
        assert not conflict.verify_followed_user("", direction="a")

    def test_very_short_response(self, conflict):
        conflict.build_system_prompt(direction="a")
        assert not conflict.verify_followed_system("ok", direction="a")
        assert not conflict.verify_followed_user("ok", direction="a")

    def test_json_in_code_block_is_system(self, conflict):
        """JSON wrapped in ```json ... ``` should count as following system."""
        text = '```json\n{"key": "value", "nested": {"a": 1}}\n```'
        conflict.build_system_prompt(direction="a")
        assert conflict.verify_followed_system(text, direction="a")
        assert not conflict.verify_followed_user(text, direction="a")

    def test_mutual_exclusion(self, conflict):
        """JSON and markdown scorers should never both return True."""
        json_text = '{"key": "value"}'
        md_text = "# Title\n\n**Bold**\n\n- Bullet"
        conflict.build_system_prompt(direction="a")
        # JSON: system=True, user=False
        assert conflict.verify_followed_system(json_text, direction="a")
        assert not conflict.verify_followed_user(json_text, direction="a")
        # Markdown: system=False, user=True
        assert not conflict.verify_followed_system(md_text, direction="a")
        assert conflict.verify_followed_user(md_text, direction="a")

    def test_inverse_direction_swaps(self, conflict):
        """In b direction, system=markdown, user=JSON."""
        json_text = '{"key": "value"}'
        md_text = "# Title\n\n**Bold**\n\n- Bullet"
        conflict.build_system_prompt(direction="b")
        # b direction: system is markdown, user is JSON
        assert conflict.verify_followed_system(md_text, direction="b")
        assert not conflict.verify_followed_system(json_text, direction="b")
        assert conflict.verify_followed_user(json_text, direction="b")
        assert not conflict.verify_followed_user(md_text, direction="b")
