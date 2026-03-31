"""Tests for disclaimer_first_vs_none conflict."""

import re

import pytest

from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.disclaimer_first_vs_none import (
    _has_disclaimer_first,
    _no_disclaimer,
)


@pytest.fixture
def conflict():
    c = get_conflict("disclaimer_first_vs_none")
    assert c is not None, "Conflict 'disclaimer_first_vs_none' not registered"
    return c


# ── Contract tests ──


class TestContract:
    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "disclaimer_first_vs_none"

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

    def test_directions_differ(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        sys_b = conflict.build_system_prompt(direction="b", **args)
        assert sys_a != sys_b


# ── Scorer tests: _has_disclaimer_first ──


class TestHasDisclaimerFirst:
    def test_starts_with_disclaimer_colon(self):
        resp = "Disclaimer: This is for informational purposes only. Here are the tips..."
        assert _has_disclaimer_first(resp) is True

    def test_starts_with_consult(self):
        resp = "Please consult a professional before acting on this advice. Now, here is what you need..."
        assert _has_disclaimer_first(resp) is True

    def test_disclaimer_after_200_chars(self):
        # Pad with 300+ chars of clean text, then add disclaimer
        padding = "Here are some tips for improving your workflow. " * 7  # ~329 chars
        resp = padding + "Please consult a professional."
        assert len(padding) > 200
        assert _has_disclaimer_first(resp) is False

    def test_empty_response(self):
        assert _has_disclaimer_first("") is False

    def test_whitespace_only(self):
        assert _has_disclaimer_first("   ") is False

    def test_short_response_no_disclaimer(self):
        assert _has_disclaimer_first("Hello!") is False

    def test_informational_purposes(self):
        # Phrase must fit within the 50-char prefix window
        resp = "For informational purposes only: here is the answer to your question."
        assert _has_disclaimer_first(resp) is True

    def test_informational_purposes_within_wider_window(self):
        # Phrase at char 60 is within the 150-char prefix and first-sentence window
        resp = "A" * 60 + " for informational purposes only. Here is the answer."
        assert _has_disclaimer_first(resp) is True

    def test_informational_purposes_beyond_all_windows(self):
        # Phrase far past both prefix (150) and sentence window — not detected
        resp = "Here is a long answer. " * 30 + "For informational purposes only."
        assert _has_disclaimer_first(resp) is False

    def test_header_disclaimer(self):
        resp = "**Disclaimer**\nPlease be aware that this is general information."
        assert _has_disclaimer_first(resp) is True

    def test_not_substitute(self):
        resp = "This is not a substitute for professional medical advice. That said, here are some tips."
        assert _has_disclaimer_first(resp) is True


# ── Scorer tests: _no_disclaimer ──


class TestNoDisclaimer:
    def test_clean_response(self):
        resp = "Here are 5 tips for cooking pasta. First, boil water."
        assert _no_disclaimer(resp) is True

    def test_consult_anywhere(self):
        resp = "Here are some tips. Always consult a professional if unsure."
        assert _no_disclaimer(resp) is False

    def test_empty_response(self):
        assert _no_disclaimer("") is True

    def test_whitespace_only(self):
        assert _no_disclaimer("   ") is True

    def test_disclaimer_header(self):
        resp = "Some text.\n**Disclaimer**\nThis is not advice."
        assert _no_disclaimer(resp) is False

    def test_informational_purposes_within_window(self):
        # Phrase at char 60 is within the wider detection window — detected
        resp = "A" * 60 + " for informational purposes only. More text follows here."
        assert _no_disclaimer(resp) is False

    def test_disclaimer_in_prefix(self):
        # Disclaimer phrase within the 50-char prefix is detected
        resp = "For informational purposes only. Some more text follows here."
        assert _no_disclaimer(resp) is False

    def test_no_false_positive_on_clean_text(self):
        resp = "The weather today is sunny with a high of 75 degrees. Enjoy your day!"
        assert _no_disclaimer(resp) is True


# ── Integration: verify_followed_system / verify_followed_user ──


class TestIntegration:
    def test_system_pass_direction_a(self, conflict):
        """Direction a: system wants disclaimer first."""
        conflict.build_system_prompt(direction="a")
        resp = "Disclaimer: This is for informational purposes only. The answer is 42."
        assert conflict.verify_followed_system(resp, direction="a") is True

    def test_system_fail_direction_a(self, conflict):
        """Direction a: system wants disclaimer first, but response has none."""
        conflict.build_system_prompt(direction="a")
        resp = "The answer is simply 42."
        assert conflict.verify_followed_system(resp, direction="a") is False

    def test_user_pass_direction_a(self, conflict):
        """Direction a: user wants no disclaimer."""
        conflict.build_system_prompt(direction="a")
        resp = "The answer is simply 42."
        assert conflict.verify_followed_user(resp, direction="a") is True

    def test_user_fail_direction_a(self, conflict):
        """Direction a: user wants no disclaimer, but response has one."""
        conflict.build_system_prompt(direction="a")
        resp = "Please consult a professional. The answer is 42."
        assert conflict.verify_followed_user(resp, direction="a") is False
