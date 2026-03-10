"""Test conflicts batch 3: final 14 + registry completeness + compatibility matrix."""

import pytest
from phase0_v2.conflicts.registry import get_conflict, get_all_conflicts, get_conflict_ids
from phase0_v2.conflicts.compatibility import (
    INCOMPATIBLE, EXPLICITLY_COMPATIBLE, TASK_CATEGORIES,
    validate_matrix_coverage, is_compatible,
)
from collections import Counter


BATCH3_IDS = [
    "keyword_in_early_sentence", "alphabetical_sentences",
    "sentence_connector_density", "alliteration_density", "odd_even_syllables",
    "paragraph_start_same_word",
    "word_repetition_density", "lowercase_vs_capitalized",
    "template_response",
]

NON_INVERTIBLE_BATCH3 = {"odd_even_syllables"}
PARTIAL_BATCH3 = set()  # alliteration_density is "full", not "partial"


@pytest.fixture(params=BATCH3_IDS)
def conflict(request):
    c = get_conflict(request.param)
    assert c is not None, f"Conflict '{request.param}' not registered"
    return c


class TestBatch3Contract:
    def test_build_direction_a(self, conflict):
        args = conflict.sample_args()
        sys_a = conflict.build_system_prompt(direction="a", **args)
        usr_a = conflict.build_user_conflict_prompt(direction="a")
        assert sys_a and usr_a

    def test_counterbalance_quality(self, conflict):
        if conflict.conflict_id in NON_INVERTIBLE_BATCH3:
            assert conflict.counterbalance_quality == "none"
        elif conflict.conflict_id in PARTIAL_BATCH3:
            assert conflict.counterbalance_quality == "partial"
        else:
            assert conflict.counterbalance_quality == "full"

    def test_non_invertible_raises_on_direction_b(self, conflict):
        if not conflict.supports_counterbalancing():
            with pytest.raises(ValueError):
                conflict.build_system_prompt(direction="b", **conflict.sample_args())

    def test_verify_returns_bool(self, conflict):
        args = conflict.sample_args()
        conflict.build_system_prompt(direction="a", **args)
        assert isinstance(conflict.verify_followed_system("test", direction="a"), bool)
        assert isinstance(conflict.verify_followed_user("test", direction="a"), bool)


# -- Registry completeness --

class TestRegistryComplete:
    def test_total_conflicts(self):
        assert len(get_all_conflicts()) == 42

    def test_all_ids_unique(self):
        ids = get_conflict_ids()
        assert len(ids) == len(set(ids))

    def test_counterbalance_distribution(self):
        counts = Counter(c.counterbalance_quality for c in get_all_conflicts())
        assert counts["full"] == 38, f"Expected 38 full, got {counts['full']}"
        assert counts["partial"] == 1, f"Expected 1 partial, got {counts['partial']}"
        assert counts["none"] == 3, f"Expected 3 none, got {counts['none']}"

    def test_all_non_invertible_ids(self):
        non_inv = {c.conflict_id for c in get_all_conflicts() if c.counterbalance_quality == "none"}
        assert non_inv == {"stairs_indent", "each_word_new_line", "odd_even_syllables"}


# -- Compatibility matrix --

class TestCompatibilityMatrix:
    def test_all_conflicts_covered(self):
        uncovered = validate_matrix_coverage(get_conflict_ids())
        assert uncovered == [], f"Not in matrix: {uncovered}"

    def test_no_overlap(self):
        overlap = set(INCOMPATIBLE.keys()) & EXPLICITLY_COMPATIBLE
        assert overlap == set(), f"In both INCOMPATIBLE and EXPLICITLY_COMPATIBLE: {overlap}"

    def test_incompatible_categories_valid(self):
        for cid, cats in INCOMPATIBLE.items():
            invalid = cats - TASK_CATEGORIES
            assert invalid == set(), f"{cid} has invalid categories: {invalid}"

    def test_stairs_indent_incompatible_with_coding_math(self):
        assert not is_compatible("stairs_indent", "coding")
        assert not is_compatible("stairs_indent", "math")
        assert is_compatible("stairs_indent", "general")

    def test_language_en_es_compatible_with_all(self):
        for cat in TASK_CATEGORIES:
            assert is_compatible("language_en_es", cat)

    def test_stairs_indent_incompatible_coding_math(self):
        assert not is_compatible("stairs_indent", "coding")
        assert not is_compatible("stairs_indent", "math")
        assert is_compatible("stairs_indent", "creative")


# -- Specific conflict verify tests --

class TestLowercaseVsCapitalized:
    def test_lowercase_verify(self):
        c = get_conflict("lowercase_vs_capitalized")
        c.build_system_prompt(direction="a")
        assert isinstance(c.verify_followed_system("all lowercase text here.", direction="a"), bool)
        assert isinstance(c.verify_followed_user("This is Properly Capitalized.", direction="a"), bool)


class TestWordRepetitionDensity:
    def test_no_args(self):
        c = get_conflict("word_repetition_density")
        args = c.sample_args()
        assert args == {}
        c.build_system_prompt(direction="a", **args)
        result = c.verify_followed_system("A unique sentence with no repeats.", direction="a")
        assert isinstance(result, bool)
