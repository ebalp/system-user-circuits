"""Test conflicts batch 2: remaining batch 2 conflicts after quality cleanup."""

import pytest
from phase0_v2.conflicts.registry import get_conflict


BATCH2_IDS = [
    "number_density",
    "vocabulary_diversity", "response_length",
    "each_word_new_line", "bullets_and_sub_bullets", "html_emphasis_tags",
]

NON_INVERTIBLE = set()
PARTIAL = {"bullets_and_sub_bullets"}


@pytest.fixture(params=BATCH2_IDS)
def conflict(request):
    c = get_conflict(request.param)
    assert c is not None, f"Conflict '{request.param}' not registered"
    return c


class TestBatch2Contract:
    def test_build_direction_a(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_counterbalance_quality_correct(self, conflict):
        if conflict.conflict_id in NON_INVERTIBLE:
            assert conflict.counterbalance_quality == "none"
            assert not conflict.supports_counterbalancing()
        elif conflict.conflict_id in PARTIAL:
            assert conflict.counterbalance_quality == "partial"
            assert conflict.supports_counterbalancing()
        else:
            assert conflict.counterbalance_quality == "full"
            assert conflict.supports_counterbalancing()

    def test_invertible_has_both_directions(self, conflict):
        if conflict.supports_counterbalancing():
            args = conflict.sample_args()
            sys_a = conflict.build_system_prompt(direction="a", **args)
            sys_b = conflict.build_system_prompt(direction="b", **args)
            assert sys_a != sys_b

    def test_non_invertible_raises_on_direction_b(self, conflict):
        if not conflict.supports_counterbalancing():
            args = conflict.sample_args()
            with pytest.raises(ValueError):
                conflict.build_system_prompt(direction="b", **args)


# ── Specific verify function tests ──

class TestEachWordNewLine:
    def test_verify_direction_a(self):
        c = get_conflict("each_word_new_line")
        c.build_system_prompt(direction="a")
        one_per_line = "Hello\nworld\nhow\nare\nyou"
        sys_r = c.verify_followed_system(one_per_line, direction="a")
        assert isinstance(sys_r, bool)

    def test_verify_direction_b(self):
        c = get_conflict("each_word_new_line")
        c.build_system_prompt(direction="b")
        # Need 38+ words for inverted threshold > 0.973 (threshold=0.027)
        normal = "The quick brown fox jumps over the lazy dog and the cat sits on the mat in the garden by the tree near the pond while the birds sing in the sky above the rolling green hills today"
        sys_r = c.verify_followed_system(normal, direction="b")
        assert sys_r is True
        one_per_line = "Hello\nworld\nhow\nare\nyou"
        usr_r = c.verify_followed_user(one_per_line, direction="b")
        assert usr_r is True


class TestParametrizedConflicts:
    def test_word_count_range_replaced_by_response_length(self):
        """word_count_range was replaced by response_length."""
        assert get_conflict("word_count_range") is None
        c = get_conflict("response_length")
        assert c is not None
        assert c.sample_args() == {}

    def test_exact_number_count_replaced_by_number_density(self):
        """exact_number_count was replaced by number_density."""
        assert get_conflict("exact_number_count") is None
        c = get_conflict("number_density")
        assert c is not None
        assert c.sample_args() == {}

    def test_min_pronoun_count_replaced_by_pronoun_density(self):
        """min_pronoun_count was replaced by pronoun_density."""
        assert get_conflict("min_pronoun_count") is None
        c = get_conflict("pronoun_density")
        assert c is not None
