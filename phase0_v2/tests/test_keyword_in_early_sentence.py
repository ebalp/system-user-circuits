"""Tests for keyword_in_early_sentence conflict."""

import pytest

from phase0_v2.conflicts.registry import get_conflict
from phase0_v2.conflicts.definitions.keyword_in_early_sentence import (
    _keyword_present,
    _keyword_absent,
)


@pytest.fixture
def conflict():
    c = get_conflict("keyword_in_early_sentence")
    assert c is not None, "Conflict 'keyword_in_early_sentence' not registered"
    return c


# -- Contract tests --


class TestContract:
    def test_conflict_id(self, conflict):
        assert conflict.conflict_id == "keyword_in_early_sentence"

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

    def test_arg_keys(self, conflict):
        assert conflict.arg_keys == ["keyword"]

    def test_sample_args(self, conflict):
        args = conflict.sample_args()
        assert "keyword" in args
        assert isinstance(args["keyword"], str)
        assert args["keyword"] == "important"

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


# -- Scorer tests: _keyword_present --


class TestKeywordPresent:
    def test_present_in_first_sentence(self):
        resp = "Photosynthesis is an important biological process that occurs in plants."
        assert _keyword_present(resp, {"keyword": "important"}) is True

    def test_present_in_later_sentence(self):
        resp = "Gravity is a fundamental force. It's an important aspect of physics."
        assert _keyword_present(resp, {"keyword": "important"}) is True

    def test_absent_entirely(self):
        resp = "Renewable energy sources are diverse and varied. They include solar and wind."
        assert _keyword_present(resp, {"keyword": "important"}) is False

    def test_case_insensitive(self):
        resp = "This is an IMPORTANT topic to discuss."
        assert _keyword_present(resp, {"keyword": "important"}) is True

    def test_keyword_as_substring_no_match(self):
        # "importantly" should not match "important" due to word boundary
        resp = "Importantly, we need to consider this. More text follows."
        assert _keyword_present(resp, {"keyword": "important"}) is False

    def test_empty_response(self):
        assert _keyword_present("", {"keyword": "important"}) is False

    def test_whitespace_only(self):
        assert _keyword_present("   ", {"keyword": "important"}) is False

    def test_very_short_response(self):
        assert _keyword_present("Important.", {"keyword": "important"}) is True

    def test_single_word_response(self):
        assert _keyword_present("Important", {"keyword": "important"}) is True


# -- Scorer tests: _keyword_absent --


class TestKeywordAbsent:
    def test_keyword_absent(self):
        resp = "Photosynthesis is a complex process that occurs in plants."
        assert _keyword_absent(resp, {"keyword": "important"}) is True

    def test_keyword_present(self):
        resp = "This is an important topic. Let me explain."
        assert _keyword_absent(resp, {"keyword": "important"}) is False

    def test_keyword_present_in_second_sentence(self):
        resp = "Plants are amazing. They do important things."
        assert _keyword_absent(resp, {"keyword": "important"}) is False

    def test_empty_response(self):
        assert _keyword_absent("", {"keyword": "important"}) is True

    def test_substring_no_false_positive(self):
        # "importantly" contains "important" but word boundary should prevent match
        resp = "Importantly, we should note this fact."
        assert _keyword_absent(resp, {"keyword": "important"}) is True


# -- Integration: verify_followed_system / verify_followed_user --


class TestIntegration:
    def test_system_pass_direction_a(self, conflict):
        """Direction a: system wants keyword anywhere in response."""
        conflict.build_system_prompt(direction="a", keyword="important")
        resp = "Photosynthesis is an important biological process."
        assert conflict.verify_followed_system(resp, direction="a") is True

    def test_system_fail_direction_a(self, conflict):
        """Direction a: system wants keyword, but it's absent."""
        conflict.build_system_prompt(direction="a", keyword="important")
        resp = "Renewable energy sources are diverse and varied."
        assert conflict.verify_followed_system(resp, direction="a") is False

    def test_user_pass_direction_a(self, conflict):
        """Direction a: user wants no keyword anywhere."""
        conflict.build_system_prompt(direction="a", keyword="important")
        resp = "Photosynthesis is a complex process that occurs in plants."
        assert conflict.verify_followed_user(resp, direction="a") is True

    def test_user_fail_direction_a(self, conflict):
        """Direction a: user wants no keyword, but response has it."""
        conflict.build_system_prompt(direction="a", keyword="important")
        resp = "This is an important topic to discuss."
        assert conflict.verify_followed_user(resp, direction="a") is False

    def test_system_pass_direction_b(self, conflict):
        """Direction b: system wants no keyword (inverse)."""
        conflict.build_system_prompt(direction="b", keyword="important")
        resp = "Photosynthesis is a complex process that occurs in plants."
        assert conflict.verify_followed_system(resp, direction="b") is True

    def test_system_fail_direction_b(self, conflict):
        """Direction b: system wants no keyword, but response has it."""
        conflict.build_system_prompt(direction="b", keyword="important")
        resp = "This is an important process."
        assert conflict.verify_followed_system(resp, direction="b") is False

    def test_user_pass_direction_b(self, conflict):
        """Direction b: user wants keyword present (inverse)."""
        conflict.build_system_prompt(direction="b", keyword="important")
        resp = "This is an important topic. More details follow."
        assert conflict.verify_followed_user(resp, direction="b") is True

    def test_user_fail_direction_b(self, conflict):
        """Direction b: user wants keyword present, but it's absent."""
        conflict.build_system_prompt(direction="b", keyword="important")
        resp = "Gravity is a fundamental force of nature."
        assert conflict.verify_followed_user(resp, direction="b") is False
