"""Tests for phase1_linear_probing.metadata_clf module.

Verifies that the metadata control classifiers report both roc_auc and
balanced_accuracy, support grouped CV, and maintain backward compatibility.
All tests use synthetic data -- no dependency on real experiment files.
"""

import numpy as np
import pandas as pd
import pytest

from metadata_clf import (
    run_linear_control,
    run_boosted_control,
    run_group_ablation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def meta_X():
    """Synthetic feature matrix: 50 samples, 10 features."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 10)).astype(np.float32)


@pytest.fixture
def meta_y():
    """50 binary labels, balanced."""
    return np.array([0, 1] * 25)


@pytest.fixture
def meta_groups():
    """50 samples across 4 groups."""
    return np.array(
        ["type_a"] * 13
        + ["type_b"] * 12
        + ["type_c"] * 13
        + ["type_d"] * 12
    )


# ── Expected keys ────────────────────────────────────────────────────────────

_EXPECTED_KEYS = {
    "roc_auc_mean",
    "roc_auc_std",
    "roc_auc_scores",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "balanced_accuracy_scores",
    "estimator_name",
    # backward compat aliases
    "mean",
    "std",
    "scores",
}


# ── run_linear_control ───────────────────────────────────────────────────────

def test_run_linear_control_returns_both_metrics(meta_X, meta_y):
    """Linear control returns both roc_auc and balanced_accuracy metrics."""
    result = run_linear_control(meta_X, meta_y, n_folds=2)

    assert _EXPECTED_KEYS <= set(result.keys())
    assert isinstance(result["roc_auc_mean"], float)
    assert isinstance(result["balanced_accuracy_mean"], float)
    assert len(result["roc_auc_scores"]) == 2
    assert len(result["balanced_accuracy_scores"]) == 2
    assert result["estimator_name"] == "LogisticRegression"


def test_run_linear_control_grouped(meta_X, meta_y, meta_groups):
    """Linear control works with grouped CV mode."""
    result = run_linear_control(
        meta_X, meta_y,
        cv_mode="grouped",
        groups=meta_groups,
        n_folds=4,
    )

    assert _EXPECTED_KEYS <= set(result.keys())
    assert isinstance(result["roc_auc_mean"], float)
    assert isinstance(result["balanced_accuracy_mean"], float)
    assert len(result["roc_auc_scores"]) == 4


# ── run_boosted_control ──────────────────────────────────────────────────────

def test_run_boosted_control_returns_both_metrics(meta_X, meta_y):
    """Boosted control returns both roc_auc and balanced_accuracy metrics."""
    result = run_boosted_control(
        meta_X, meta_y,
        n_folds=2,
        n_search_iter=2,
        inner_folds=2,
    )

    assert _EXPECTED_KEYS <= set(result.keys())
    assert isinstance(result["roc_auc_mean"], float)
    assert isinstance(result["balanced_accuracy_mean"], float)
    assert len(result["roc_auc_scores"]) == 2
    assert len(result["balanced_accuracy_scores"]) == 2
    assert result["estimator_name"] == "HistGradientBoosting (nested CV)"


def test_run_boosted_control_grouped(meta_X, meta_y, meta_groups):
    """Boosted control works with grouped CV mode."""
    result = run_boosted_control(
        meta_X, meta_y,
        cv_mode="grouped",
        groups=meta_groups,
        n_folds=4,
        n_search_iter=2,
        inner_folds=2,
    )

    assert _EXPECTED_KEYS <= set(result.keys())
    assert isinstance(result["roc_auc_mean"], float)
    assert isinstance(result["balanced_accuracy_mean"], float)
    assert len(result["roc_auc_scores"]) == 4


# ── run_group_ablation ───────────────────────────────────────────────────────

def test_run_group_ablation_multi_metric(meta_X, meta_y):
    """Group ablation returns DataFrame with both metric columns."""
    # Use 2 simple feature groups covering all 10 features
    feature_groups = {
        "group_a": list(range(0, 5)),
        "group_b": list(range(5, 10)),
    }

    df = run_group_ablation(
        meta_X, meta_y, feature_groups,
        classifier="linear",
        n_folds=2,
    )

    assert isinstance(df, pd.DataFrame)
    # baseline + 2 groups
    assert len(df) == 3
    expected_cols = {
        "group_dropped",
        "roc_auc_mean",
        "roc_auc_std",
        "roc_auc_drop",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "balanced_accuracy_drop",
    }
    assert expected_cols <= set(df.columns)
    # Baseline row has zero drop
    baseline = df[df["group_dropped"] == "none (baseline)"].iloc[0]
    assert baseline["roc_auc_drop"] == 0.0
    assert baseline["balanced_accuracy_drop"] == 0.0


# ── backward compatibility ───────────────────────────────────────────────────

def test_backward_compat_aliases(meta_X, meta_y):
    """Backward compat aliases (mean, std, scores) match roc_auc values."""
    result = run_linear_control(meta_X, meta_y, n_folds=2)

    assert result["mean"] == result["roc_auc_mean"]
    assert result["std"] == result["roc_auc_std"]
    np.testing.assert_array_equal(result["scores"], result["roc_auc_scores"])

    # Same check for boosted
    result_b = run_boosted_control(
        meta_X, meta_y, n_folds=2, n_search_iter=2, inner_folds=2,
    )
    assert result_b["mean"] == result_b["roc_auc_mean"]
    assert result_b["std"] == result_b["roc_auc_std"]
    np.testing.assert_array_equal(
        result_b["scores"], result_b["roc_auc_scores"]
    )
