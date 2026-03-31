"""Tests for per-model threshold optimizer (Pareto frontier)."""

import numpy as np
import pytest

from phase0_v2.calibration.per_model_thresholds import (
    classify_distribution,
    compute_three_metrics,
    find_pareto_frontier,
    intersect_ranges,
    select_threshold,
)


@pytest.fixture(autouse=True)
def _seed_rng():
    np.random.seed(42)


# ── Distribution classification ─────────────────────────────────────────


class TestClassifyDistribution:
    def test_bimodal(self):
        scores = np.concatenate([
            np.random.normal(0.2, 0.05, 100),
            np.random.normal(0.8, 0.05, 100),
        ])
        scores = np.clip(scores, 0, 1)
        dist, peaks, *_ = classify_distribution(scores)
        assert dist == "bimodal"
        assert len(peaks) == 2

    def test_unimodal(self):
        scores = np.random.normal(0.5, 0.1, 200)
        scores = np.clip(scores, 0, 1)
        dist, peaks, *_ = classify_distribution(scores)
        assert dist == "unimodal"
        assert len(peaks) <= 1

    def test_trimodal_classified_as_bimodal(self):
        """Three-peaked distribution keeps only the two tallest peaks."""
        scores = np.concatenate([
            np.random.normal(0.15, 0.04, 100),
            np.random.normal(0.50, 0.04, 100),
            np.random.normal(0.85, 0.04, 100),
        ])
        scores = np.clip(scores, 0, 1)
        dist, peaks, *_ = classify_distribution(scores)
        assert dist == "bimodal"
        assert len(peaks) == 2


# ── Range intersection ───────────────────────────────────────────────────


class TestIntersectRanges:
    def test_overlapping(self):
        assert intersect_ranges((0.2, 0.6), (0.4, 0.8)) == (0.4, 0.6)

    def test_non_overlapping(self):
        assert intersect_ranges((0.1, 0.3), (0.5, 0.7)) is None

    def test_one_none(self):
        assert intersect_ranges((0.2, 0.5), None) is None
        assert intersect_ranges(None, (0.2, 0.5)) is None

    def test_both_none(self):
        assert intersect_ranges(None, None) is None

    def test_touching(self):
        assert intersect_ranges((0.2, 0.5), (0.5, 0.8)) == (0.5, 0.5)

    def test_subset(self):
        assert intersect_ranges((0.2, 0.8), (0.4, 0.6)) == (0.4, 0.6)


# ── Pareto frontier ─────────────────────────────────────────────────────


class TestComputeThreeMetrics:
    def test_bimodal_returns_metrics(self):
        scores = np.concatenate([
            np.random.normal(0.2, 0.05, 100),
            np.random.normal(0.8, 0.05, 100),
        ])
        scores = np.clip(scores, 0, 1)
        candidates = [t / 100 for t in range(1, 100)]
        result = compute_three_metrics(scores, [], candidates)
        assert result is not None
        assert result["distribution"] == "bimodal"
        assert len(result["d_norm"]) == len(candidates)
        assert len(result["c_norm"]) == len(candidates)
        assert len(result["ba"]) == len(candidates)

    def test_zero_variance_returns_none(self):
        scores = np.array([0.5] * 100)
        candidates = [t / 100 for t in range(1, 100)]
        assert compute_three_metrics(scores, [], candidates) is None

    def test_unimodal_d_norm_mass_fraction(self):
        """Unimodal d_norm = min(valley=1.0, mass): high in the middle, low at extremes."""
        scores = np.random.normal(0.5, 0.1, 200)
        scores = np.clip(scores, 0, 1)
        candidates = [t / 100 for t in range(1, 100)]
        result = compute_three_metrics(scores, [], candidates)
        assert result is not None
        assert result["distribution"] == "unimodal"
        # d_norm near the median should be ~1.0 (mass split evenly, valley=1.0)
        mid_idx = 49  # T=0.50
        assert result["d_norm"][mid_idx] > 0.8
        # d_norm at extremes should be ~0.0 (almost all mass on one side)
        assert result["d_norm"][0] < 0.05
        assert result["d_norm"][-1] < 0.05

    def test_bimodal_d_norm_min_of_valley_and_mass(self):
        """Bimodal d_norm = min(valley, mass): low at valley between peaks."""
        scores = np.concatenate([
            np.random.normal(0.2, 0.05, 200),
            np.random.normal(0.8, 0.05, 200),
        ])
        scores = np.clip(scores, 0, 1)
        candidates = [t / 100 for t in range(1, 100)]
        result = compute_three_metrics(scores, [], candidates)
        assert result is not None
        assert result["distribution"] == "bimodal"
        # d_norm at the valley (~T=0.50) should be low (valley drives min)
        mid_idx = 49  # T=0.50
        assert result["d_norm"][mid_idx] < 0.05
        # d_norm on a peak should be high
        assert result["d_norm"][19] >= 0.5  # T=0.20 (on left peak)


class TestFindParetoFrontier:
    def test_basic_frontier(self):
        metrics = {
            "d_norm": np.array([0.0, 0.01, 0.02, 0.5]),
            "c_norm": np.array([0.0, 0.01, 0.02, 0.5]),
            "ba": np.array([1.0, 1.0, 1.0, 1.0]),
        }
        pareto = find_pareto_frontier(metrics, max_d=0.05, max_c=0.05, min_ba=0.9)
        # Point 0 dominates all others
        assert pareto == [0]

    def test_tradeoff_frontier(self):
        metrics = {
            "d_norm": np.array([0.01, 0.02]),
            "c_norm": np.array([0.02, 0.01]),
            "ba": np.array([1.0, 1.0]),
        }
        pareto = find_pareto_frontier(metrics, max_d=0.05, max_c=0.05, min_ba=0.9)
        # Neither dominates the other
        assert sorted(pareto) == [0, 1]

    def test_infeasible_ba(self):
        metrics = {
            "d_norm": np.array([0.0, 0.0]),
            "c_norm": np.array([0.0, 0.0]),
            "ba": np.array([0.8, 0.85]),
        }
        pareto = find_pareto_frontier(metrics, max_d=0.05, max_c=0.05, min_ba=0.9)
        assert pareto == []

    def test_infeasible_costs(self):
        metrics = {
            "d_norm": np.array([0.1, 0.2]),
            "c_norm": np.array([0.1, 0.2]),
            "ba": np.array([1.0, 1.0]),
        }
        pareto = find_pareto_frontier(metrics, max_d=0.05, max_c=0.05, min_ba=0.9)
        assert pareto == []


# ── Threshold selection ──────────────────────────────────────────────────


class TestSelectThreshold:
    def test_bimodal_finds_threshold(self):
        scores = np.concatenate([
            np.random.normal(0.2, 0.05, 100),
            np.random.normal(0.8, 0.05, 100),
        ])
        scores = np.clip(scores, 0, 1)
        result = select_threshold(scores, [])
        assert result["feasible"]
        assert 0.2 < result["threshold"] < 0.8
        assert result["distribution"] == "bimodal"
        assert result["d_norm"] is not None
        assert result["c_norm"] is not None
        assert result["ba"] is not None

    def test_insufficient_data(self):
        scores = np.array([0.1, 0.2, 0.3])
        result = select_threshold(scores, [])
        assert not result["feasible"]

    def test_zero_variance(self):
        scores = np.array([0.5] * 50)
        result = select_threshold(scores, [])
        assert not result["feasible"]
