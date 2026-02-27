"""Tests for phase1_linear_probing.probe module.

All tests use synthetic data from conftest.py -- no dependency on real
experiment files.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from phase1_linear_probing.probe import (
    ProbeResult,
    compute_constraint_cmds,
    compute_fold_cmds,
    load_classifiers,
    load_results,
    make_cv_splitter,
    probe_and_fit,
    probe_control,
    results_path,
    save_classifiers,
    save_results,
)


# ── make_cv_splitter ─────────────────────────────────────────────────────────


def test_make_cv_splitter_stratified():
    """Stratified mode returns a StratifiedKFold instance."""
    cv = make_cv_splitter(cv_mode="stratified", n_folds=3, random_state=0)
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 3


def test_make_cv_splitter_grouped():
    """Grouped mode returns a GroupKFold instance."""
    cv = make_cv_splitter(cv_mode="grouped", n_folds=4)
    assert isinstance(cv, GroupKFold)
    assert cv.n_splits == 4


def test_make_cv_splitter_invalid():
    """Invalid cv_mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown cv_mode"):
        make_cv_splitter(cv_mode="invalid")


# ── probe_and_fit ────────────────────────────────────────────────────────────


def test_probe_and_fit_returns_both_metrics(
    synthetic_activations, synthetic_labels
):
    """CV scores contain both roc_auc and balanced_accuracy columns."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="stratified",
        n_folds=2,
    )
    assert "last_prompt" in results
    pr = results["last_prompt"]

    expected_cols = {
        "layer",
        "roc_auc_mean",
        "roc_auc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    }
    assert expected_cols == set(pr.cv_scores.columns)
    # 4 layers in synthetic data
    assert len(pr.cv_scores) == 4


def test_probe_and_fit_grouped_cv(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Grouped CV mode runs without error and returns expected shape."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
    )
    assert "last_prompt" in results
    pr = results["last_prompt"]
    assert len(pr.cv_scores) == 4
    assert pr.cv_mode == "grouped"


def test_probe_and_fit_classifiers_shape(
    synthetic_activations, synthetic_labels
):
    """Number of classifiers and scalers equals n_layers."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
    )
    pr = results["last_prompt"]
    assert len(pr.classifiers) == 4
    assert len(pr.scalers) == 4
    for clf in pr.classifiers:
        assert isinstance(clf, LogisticRegression)


def test_probe_and_fit_weights_unit_norm(
    synthetic_activations, synthetic_labels
):
    """Fitted weights are unit-normalized."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
    )
    pr = results["last_prompt"]
    norms = np.linalg.norm(pr.weights, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_probe_and_fit_with_scaler(synthetic_activations, synthetic_labels):
    """When use_scaler=True, each scaler is a StandardScaler instance."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
        use_scaler=True,
    )
    pr = results["last_prompt"]
    assert pr.use_scaler is True
    for scaler in pr.scalers:
        assert isinstance(scaler, StandardScaler)


def test_probe_and_fit_without_scaler(synthetic_activations, synthetic_labels):
    """When use_scaler=False, each scaler entry is None."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
        use_scaler=False,
    )
    pr = results["last_prompt"]
    assert pr.use_scaler is False
    for scaler in pr.scalers:
        assert scaler is None


# ── probe_control ────────────────────────────────────────────────────────────


def test_probe_control_returns_both_metrics(
    synthetic_activations, synthetic_labels
):
    """Control probe DataFrame has both roc_auc and balanced_accuracy columns."""
    results = probe_control(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
        n_permutations=1,
    )
    assert "last_prompt" in results
    df = results["last_prompt"]

    expected_cols = {
        "layer",
        "roc_auc_mean",
        "roc_auc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    }
    assert expected_cols == set(df.columns)
    assert len(df) == 4


def test_probe_control_grouped(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Grouped CV mode works for probe_control."""
    results = probe_control(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
        n_permutations=1,
    )
    assert "last_prompt" in results
    df = results["last_prompt"]
    assert len(df) == 4


# ── save / load classifiers ─────────────────────────────────────────────────


def test_save_load_classifiers_roundtrip(
    tmp_path, synthetic_activations, synthetic_labels
):
    """Round-trip: save then load preserves weights and metadata."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        n_folds=2,
    )
    original = results["last_prompt"]

    path = tmp_path / "clf_last_prompt_unscaled.joblib"
    save_classifiers(original, path, model_name="test/model")

    loaded = load_classifiers(path)

    np.testing.assert_allclose(loaded["weights"], original.weights)
    np.testing.assert_allclose(loaded["weights_raw"], original.weights_raw)
    np.testing.assert_allclose(loaded["biases"], original.biases)
    assert loaded["metadata"]["pos_name"] == "last_prompt"
    assert loaded["metadata"]["model_name"] == "test/model"
    assert loaded["metadata"]["n_layers"] == 4
    assert loaded["metadata"]["d_model"] == 16


# ── ProbeResult dataclass ────────────────────────────────────────────────────


def test_probe_result_dataclass():
    """ProbeResult can be created manually and all fields are accessible."""
    pr = ProbeResult(
        cv_scores=pd.DataFrame(
            [{"layer": 0, "roc_auc_mean": 0.5, "roc_auc_std": 0.01,
              "balanced_accuracy_mean": 0.5, "balanced_accuracy_std": 0.01}]
        ),
        weights=np.ones((1, 8)),
        weights_raw=np.ones((1, 8)),
        biases=np.zeros(1),
        scalers=[None],
        classifiers=[LogisticRegression()],
        pos_name="test_pos",
        cv_mode="stratified",
        use_scaler=False,
    )
    assert pr.pos_name == "test_pos"
    assert pr.cv_mode == "stratified"
    assert pr.use_scaler is False
    assert pr.weights.shape == (1, 8)
    assert len(pr.classifiers) == 1
    assert len(pr.scalers) == 1
    assert isinstance(pr.cv_scores, pd.DataFrame)


def test_save_load_results_roundtrip(
    tmp_path, synthetic_activations, synthetic_labels
):
    """save_results → load_results preserves full ProbeResult including CV scores."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        ["last_prompt"],
        cv_mode="stratified",
        n_folds=2,
        use_scaler=False,
    )
    path = tmp_path / "results_test.joblib"
    save_results(results, path, model_name="test-model")
    loaded = load_results(path)

    assert set(loaded.keys()) == set(results.keys())
    for pos in results:
        orig = results[pos]
        back = loaded[pos]
        np.testing.assert_allclose(orig.weights, back.weights)
        np.testing.assert_allclose(orig.biases, back.biases)
        pd.testing.assert_frame_equal(orig.cv_scores, back.cv_scores)
        assert orig.pos_name == back.pos_name
        assert orig.cv_mode == back.cv_mode
        assert orig.use_scaler == back.use_scaler
        assert len(back.classifiers) == len(orig.classifiers)


def test_results_path():
    """results_path produces distinct paths for different cv_mode/scaler combos."""
    from pathlib import Path

    run_dir = Path("/fake/run")
    p1 = results_path(run_dir, "grouped", False)
    p2 = results_path(run_dir, "grouped", True)
    p3 = results_path(run_dir, "stratified", False)

    assert p1 != p2
    assert p1 != p3
    assert p1.name == "results_grouped_unscaled.joblib"
    assert p2.name == "results_grouped_scaled.joblib"
    assert p3.name == "results_stratified_unscaled.joblib"


# ── fold-level data ──────────────────────────────────────────────────────────


def test_probe_and_fit_fold_weights_shape(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """fold_weights has shape (n_layers, n_folds, d_model)."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
    )
    pr = results["last_prompt"]
    assert pr.fold_weights is not None
    assert pr.fold_weights.shape == (4, 4, 16)  # n_layers, n_folds, d_model


def test_probe_and_fit_fold_scores_shape(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """fold_scores has shape (n_layers, n_folds)."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
    )
    pr = results["last_prompt"]
    assert pr.fold_scores is not None
    assert pr.fold_scores.shape == (4, 4)


def test_probe_and_fit_fold_group_names(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """fold_group_names has one entry per fold for grouped CV."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
    )
    pr = results["last_prompt"]
    assert pr.fold_group_names is not None
    assert len(pr.fold_group_names) == 4
    # Each name should be one of the group values
    unique_groups = set(synthetic_groups)
    for name in pr.fold_group_names:
        assert name in unique_groups


def test_probe_and_fit_fold_weights_unit_norm(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Per-fold probe weights are unit-normalized."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="grouped",
        groups=synthetic_groups,
        n_folds=4,
    )
    pr = results["last_prompt"]
    norms = np.linalg.norm(pr.fold_weights, axis=2)  # (n_layers, n_folds)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_probe_and_fit_stratified_fold_data(
    synthetic_activations, synthetic_labels
):
    """Stratified CV also populates fold_weights/fold_scores (no group names)."""
    results = probe_and_fit(
        synthetic_activations,
        synthetic_labels,
        token_positions=["last_prompt"],
        cv_mode="stratified",
        n_folds=2,
    )
    pr = results["last_prompt"]
    assert pr.fold_weights is not None
    assert pr.fold_scores is not None
    assert pr.fold_weights.shape == (4, 2, 16)
    assert pr.fold_scores.shape == (4, 2)
    assert pr.fold_group_names is None


# ── compute_fold_cmds ────────────────────────────────────────────────────────


def test_compute_fold_cmds_shape(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """compute_fold_cmds returns dict with n_folds entries, each (d_model,)."""
    cmds = compute_fold_cmds(
        synthetic_activations,
        synthetic_labels,
        synthetic_groups,
        "last_prompt",
        layer=0,
        n_folds=4,
    )
    assert isinstance(cmds, dict)
    assert len(cmds) == 4
    for name, vec in cmds.items():
        assert isinstance(name, str)
        assert vec.shape == (16,)


def test_compute_fold_cmds_keys_match_groups(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Dict keys are the unique group names."""
    cmds = compute_fold_cmds(
        synthetic_activations,
        synthetic_labels,
        synthetic_groups,
        "last_prompt",
        layer=0,
        n_folds=4,
    )
    assert set(cmds.keys()) == set(synthetic_groups)


def test_compute_fold_cmds_unit_norm(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """CMD vectors are unit-normalized."""
    cmds = compute_fold_cmds(
        synthetic_activations,
        synthetic_labels,
        synthetic_groups,
        "last_prompt",
        layer=1,
        n_folds=4,
    )
    for vec in cmds.values():
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)


# ── compute_constraint_cmds ──────────────────────────────────────────────────


def test_compute_constraint_cmds_keys(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Returns one entry per unique group with both classes present."""
    cmds = compute_constraint_cmds(
        synthetic_activations,
        synthetic_labels,
        synthetic_groups,
        "last_prompt",
        layer=0,
    )
    assert isinstance(cmds, dict)
    # All groups in synthetic_groups have both classes (alternating 0,1)
    assert set(cmds.keys()) == set(synthetic_groups)


def test_compute_constraint_cmds_unit_norm(
    synthetic_activations, synthetic_labels, synthetic_groups
):
    """Per-constraint CMD vectors are unit-normalized."""
    cmds = compute_constraint_cmds(
        synthetic_activations,
        synthetic_labels,
        synthetic_groups,
        "last_prompt",
        layer=2,
    )
    for vec in cmds.values():
        assert vec.shape == (16,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)
