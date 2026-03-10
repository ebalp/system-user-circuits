"""Tests for cross-model threshold intersection logic."""

import pytest

from phase0_v2.calibration.cross_model_thresholds import (
    IntersectionResult,
    ModelRange,
    _intersect_range_lists,
    _intersect_two,
    _midpoint,
    _model_tier,
    compute_intersections,
)


class TestIntersectTwo:
    def test_overlapping(self):
        assert _intersect_two((0.1, 0.5), (0.3, 0.7)) == (0.3, 0.5)

    def test_contained(self):
        assert _intersect_two((0.1, 0.9), (0.3, 0.5)) == (0.3, 0.5)

    def test_identical(self):
        assert _intersect_two((0.2, 0.8), (0.2, 0.8)) == (0.2, 0.8)

    def test_touching(self):
        assert _intersect_two((0.1, 0.5), (0.5, 0.9)) == (0.5, 0.5)

    def test_disjoint(self):
        assert _intersect_two((0.1, 0.3), (0.5, 0.7)) is None

    def test_disjoint_reversed(self):
        assert _intersect_two((0.5, 0.7), (0.1, 0.3)) is None


class TestIntersectRangeLists:
    def test_single_model_single_range(self):
        result = _intersect_range_lists([[(0.1, 0.5)]])
        assert result == (0.1, 0.5)

    def test_two_models_overlap(self):
        result = _intersect_range_lists([
            [(0.1, 0.5)],
            [(0.3, 0.7)],
        ])
        assert result == (0.3, 0.5)

    def test_two_models_disjoint(self):
        result = _intersect_range_lists([
            [(0.1, 0.3)],
            [(0.5, 0.7)],
        ])
        assert result is None

    def test_three_models_overlap(self):
        result = _intersect_range_lists([
            [(0.1, 0.6)],
            [(0.2, 0.5)],
            [(0.3, 0.7)],
        ])
        assert result == (0.3, 0.5)

    def test_disjoint_regions_cartesian(self):
        """One model has disjoint regions; intersection uses Cartesian product."""
        result = _intersect_range_lists([
            [(0.1, 0.3), (0.6, 0.9)],  # model A has two regions
            [(0.2, 0.7)],  # model B overlaps both
        ])
        # Possible: (0.1,0.3)∩(0.2,0.7) = (0.2,0.3) width 0.1
        #           (0.6,0.9)∩(0.2,0.7) = (0.6,0.7) width 0.1
        # Tied width — takes first found (0.2, 0.3) or (0.6, 0.7)
        assert result is not None
        low, high = result
        assert high - low == pytest.approx(0.1)

    def test_disjoint_regions_picks_widest(self):
        """When Cartesian product yields multiple intersections, pick widest."""
        result = _intersect_range_lists([
            [(0.1, 0.2), (0.5, 0.9)],  # model A
            [(0.15, 0.85)],  # model B
        ])
        # (0.1,0.2)∩(0.15,0.85) = (0.15,0.2) width 0.05
        # (0.5,0.9)∩(0.15,0.85) = (0.5,0.85) width 0.35  ← widest
        assert result == (0.5, 0.85)

    def test_empty_input(self):
        assert _intersect_range_lists([]) is None


class TestMidpoint:
    def test_simple(self):
        assert _midpoint((0.1, 0.5)) == 0.3

    def test_rounding(self):
        assert _midpoint((0.1, 0.2)) == 0.15

    def test_point(self):
        assert _midpoint((0.5, 0.5)) == 0.5


class TestModelTier:
    def test_70b(self):
        assert _model_tier("meta-llama_Llama-3.3-70B-Instruct") == "strong"

    def test_27b(self):
        assert _model_tier("google_gemma-3-27b-it") == "strong"

    def test_8b(self):
        assert _model_tier("meta-llama_Llama-3.1-8B-Instruct") == "mid"

    def test_1b(self):
        assert _model_tier("meta-llama_Llama-3.2-1B-Instruct") == "weak"

    def test_no_match(self):
        assert _model_tier("some-unknown-model") == "mid"

    def test_no_false_match_170b(self):
        """170B should not match '70b' — the '70b' is preceded by a digit."""
        assert _model_tier("hypothetical-170B-model") == "mid"


class TestComputeIntersections:
    def _make_ranges(self, model_id: str, conflicts: dict) -> dict[str, ModelRange]:
        """Helper: {conflict_id: (low, high, ba)} → {conflict_id: ModelRange}"""
        result = {}
        for cid, (low, high, ba) in conflicts.items():
            result[cid] = ModelRange(
                model_id=model_id,
                ranges=[(low, high)],
                ba=ba,
                tier=_model_tier(model_id),
            )
        return result

    def test_full_intersection(self):
        all_ranges = {
            "modelA": self._make_ranges("modelA", {"c1": (0.1, 0.5, 1.0)}),
            "modelB": self._make_ranges("modelB", {"c1": (0.3, 0.7, 1.0)}),
        }
        results = compute_intersections(all_ranges, {"c1": 0.4})
        assert len(results) == 1
        r = results[0]
        assert r.full_intersection == (0.3, 0.5)
        assert r.proposed_t == 0.4
        assert r.status == "OK"

    def test_subset_fallback(self):
        all_ranges = {
            "modelA": self._make_ranges("modelA", {"c1": (0.1, 0.3, 1.0)}),
            "modelB": self._make_ranges("modelB", {"c1": (0.2, 0.4, 1.0)}),
            "modelC": self._make_ranges("modelC", {"c1": (0.6, 0.8, 0.9)}),
        }
        results = compute_intersections(all_ranges, {"c1": 0.25})
        r = results[0]
        assert r.full_intersection is None
        assert r.subset_intersection is not None
        assert len(r.subset_models) == 2
        assert len(r.excluded_models) == 1
        assert "SUBSET" in r.status

    def test_no_intersection(self):
        all_ranges = {
            "modelA": self._make_ranges("modelA", {"c1": (0.1, 0.2, 1.0)}),
            "modelB": self._make_ranges("modelB", {"c1": (0.4, 0.5, 1.0)}),
            "modelC": self._make_ranges("modelC", {"c1": (0.7, 0.8, 1.0)}),
        }
        results = compute_intersections(all_ranges, {"c1": 0.3})
        r = results[0]
        assert r.status == "NO INTERSECTION"

    def test_single_model(self):
        all_ranges = {
            "modelA": self._make_ranges("modelA", {"c1": (0.1, 0.5, 1.0)}),
        }
        results = compute_intersections(all_ranges, {})
        r = results[0]
        assert r.full_intersection == (0.1, 0.5)
        assert r.proposed_t == 0.3

    def test_outside_status(self):
        all_ranges = {
            "modelA": self._make_ranges("modelA", {"c1": (0.3, 0.5, 1.0)}),
            "modelB": self._make_ranges("modelB", {"c1": (0.4, 0.6, 1.0)}),
        }
        results = compute_intersections(all_ranges, {"c1": 0.2})
        r = results[0]
        assert r.status == "OUTSIDE"
