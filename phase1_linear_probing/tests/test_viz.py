"""Smoke tests for viz.py — verify functions return expected types without error.

Each test creates minimal synthetic data inline, calls the function, and
verifies it returns a ``go.Figure`` (or ``pd.DataFrame`` for the summary
table).  These do NOT depend on conftest fixtures since viz is standalone.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from viz import (
    build_summary_table,
    plot_bias_analysis,
    plot_cosine_curves,
    plot_cosine_heatmaps,
    plot_direction_agreement,
    plot_fold_generalization,
    plot_metadata_ablation,
    plot_metadata_importance,
    plot_probe_comparison,
    plot_probe_curves,
    plot_scaled_vs_unscaled,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_probe_df(n_layers: int = 4) -> pd.DataFrame:
    """Create a minimal probe-results DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "layer": list(range(n_layers)),
            "roc_auc_mean": rng.uniform(0.5, 0.9, n_layers),
            "roc_auc_std": rng.uniform(0.01, 0.05, n_layers),
            "balanced_accuracy_mean": rng.uniform(0.5, 0.9, n_layers),
            "balanced_accuracy_std": rng.uniform(0.01, 0.05, n_layers),
        }
    )


def _make_weights(n_layers: int = 4, d_model: int = 16) -> np.ndarray:
    """Create a random weight matrix and unit-normalise rows."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal((n_layers, d_model))
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    return w / norms


def _make_ablation_df() -> pd.DataFrame:
    """Create a minimal ablation DataFrame."""
    return pd.DataFrame(
        {
            "group_dropped": ["none (baseline)", "length_feats", "constraint_type"],
            "roc_auc_mean": [0.85, 0.83, 0.80],
            "roc_auc_std": [0.02, 0.03, 0.04],
            "roc_auc_drop": [0.0, 0.02, 0.05],
        }
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_plot_probe_curves_returns_figure():
    probe_results = {
        "TL": {"last_prompt": _make_probe_df()},
    }
    fig = plot_probe_curves(
        probe_results,
        token_positions=["last_prompt"],
        model_name="test-model",
        n_folds=5,
    )
    assert isinstance(fig, go.Figure)


def test_plot_probe_curves_with_save(tmp_path):
    probe_results = {
        "TL": {"last_prompt": _make_probe_df()},
    }
    save_path = tmp_path / "test.html"
    fig = plot_probe_curves(
        probe_results,
        token_positions=["last_prompt"],
        model_name="test-model",
        n_folds=5,
        save_path=save_path,
    )
    assert isinstance(fig, go.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_probe_comparison_returns_figure():
    df_s = _make_probe_df()
    df_g = _make_probe_df()
    fig = plot_probe_comparison(
        df_s, df_g, pos_name="last_prompt", model_name="test-model"
    )
    assert isinstance(fig, go.Figure)


def test_plot_cosine_heatmaps_returns_figure():
    ws = _make_weights(4, 16)
    wu = _make_weights(4, 16)
    fig = plot_cosine_heatmaps(ws, wu, pos_name="last_prompt")
    assert isinstance(fig, go.Figure)


def test_plot_cosine_curves_returns_figure():
    ws = _make_weights(4, 16)
    wu = _make_weights(4, 16)
    fig = plot_cosine_curves(ws, wu, pos_name="last_prompt")
    assert isinstance(fig, go.Figure)


def test_plot_scaled_vs_unscaled_returns_figure():
    rng = np.random.default_rng(42)
    n = 4
    fig = plot_scaled_vs_unscaled(
        auc_scaled_cv=rng.uniform(0.5, 0.9, n),
        auc_unscaled_cv=rng.uniform(0.5, 0.9, n),
        direction_agreement=rng.uniform(0.8, 1.0, n),
        scale_cv=rng.uniform(0.1, 0.5, n),
        scale_max_min=rng.uniform(1.5, 10.0, n),
        pos_name="last_prompt",
    )
    assert isinstance(fig, go.Figure)


def _make_scalers(n_layers: int = 4, d_model: int = 16):
    """Create a list of fitted StandardScaler objects for testing."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    scalers = []
    for _ in range(n_layers):
        scaler = StandardScaler()
        scaler.fit(rng.standard_normal((50, d_model)))
        scalers.append(scaler)
    return scalers


def test_plot_bias_analysis_returns_figure_without_scalers():
    rng = np.random.default_rng(42)
    n = 4
    fig = plot_bias_analysis(
        biases_scaled=rng.standard_normal(n),
        biases_unscaled=rng.standard_normal(n),
        pos_name="last_prompt",
    )
    assert isinstance(fig, go.Figure)


def test_plot_bias_analysis_returns_figure_with_scalers():
    rng = np.random.default_rng(42)
    n = 4
    fig = plot_bias_analysis(
        biases_scaled=rng.standard_normal(n),
        biases_unscaled=rng.standard_normal(n),
        pos_name="last_prompt",
        scalers=_make_scalers(n, 16),
    )
    assert isinstance(fig, go.Figure)


def test_plot_metadata_importance_returns_figure():
    rng = np.random.default_rng(42)
    linear_coefs = rng.standard_normal(5)
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    boosted_imps = rng.uniform(0, 0.1, 3)
    boosted_stds = rng.uniform(0, 0.02, 3)
    feature_names_cat = ["c1", "c2", "c3"]

    fig = plot_metadata_importance(
        linear_coefs, feature_names, boosted_imps, boosted_stds, feature_names_cat
    )
    assert isinstance(fig, go.Figure)


def test_plot_metadata_ablation_returns_figure():
    df_lin = _make_ablation_df()
    df_bst = _make_ablation_df()
    fig = plot_metadata_ablation(df_lin, df_bst, ctrl_linear=0.85, ctrl_boosted=0.88)
    assert isinstance(fig, go.Figure)


def test_build_summary_table_columns():
    probe_results = {
        "TL": {"last_prompt": _make_probe_df()},
    }
    df = build_summary_table(
        probe_results,
        token_positions=["last_prompt"],
        n_layers=32,
        ctrl_linear=0.80,
        ctrl_boosted=0.85,
    )
    assert isinstance(df, pd.DataFrame)
    for col in [
        "backend",
        "token_position",
        "peak_roc_auc",
        "peak_balanced_accuracy",
        "roc_auc_peak_layer",
        "linear_ctrl",
        "gap_vs_linear",
        "boosted_ctrl",
        "gap_vs_boosted",
    ]:
        assert col in df.columns, f"Missing column: {col}"


def test_build_summary_table_row_count():
    probe_results = {
        "TL": {"last_prompt": _make_probe_df()},
        "nnterp": {"last_prompt": _make_probe_df()},
    }
    df = build_summary_table(
        probe_results,
        token_positions=["last_prompt"],
        n_layers=32,
        ctrl_linear=0.80,
    )
    assert len(df) == 2


def test_plot_direction_agreement_returns_figure():
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal((4, 16)).astype(np.float32)
    w1 /= np.linalg.norm(w1, axis=1, keepdims=True)
    w2 = rng.standard_normal((4, 16)).astype(np.float32)
    w2 /= np.linalg.norm(w2, axis=1, keepdims=True)
    fig = plot_direction_agreement(w1, w2, "last_prompt")
    assert isinstance(fig, go.Figure)


def _make_fold_cmds_dict(n_folds: int = 4, d_model: int = 16, names=None):
    """Create a dict of unit-norm CMD vectors keyed by group name."""
    if names is None:
        names = ["format", "language", "emoji", "capitalization"][:n_folds]
    w = _make_weights(n_folds, d_model)
    return {name: w[i] for i, name in enumerate(names)}


def test_plot_fold_generalization_returns_figure():
    rng = np.random.default_rng(42)
    n_folds, d_model = 4, 16
    fold_weights = _make_weights(n_folds, d_model)
    fold_group_names = ["format", "language", "emoji", "capitalization"]
    fold_cmds = _make_fold_cmds_dict(n_folds, d_model, fold_group_names)
    fold_scores = rng.uniform(0.6, 0.95, n_folds)
    fig = plot_fold_generalization(
        fold_weights, fold_cmds, fold_scores, fold_group_names, layer=15
    )
    assert isinstance(fig, go.Figure)


def test_plot_fold_generalization_with_array():
    """Also accepts a plain ndarray for backward compat."""
    rng = np.random.default_rng(42)
    n_folds, d_model = 4, 16
    fold_weights = _make_weights(n_folds, d_model)
    fold_cmds_arr = _make_weights(n_folds, d_model)
    fold_scores = rng.uniform(0.6, 0.95, n_folds)
    fold_group_names = ["format", "language", "emoji", "capitalization"]
    fig = plot_fold_generalization(
        fold_weights, fold_cmds_arr, fold_scores, fold_group_names, layer=15
    )
    assert isinstance(fig, go.Figure)


def test_plot_fold_generalization_with_constraint_cmds():
    """Third row is added when constraint_cmds is provided."""
    rng = np.random.default_rng(42)
    n_folds, d_model = 4, 16
    fold_group_names = ["format", "language", "emoji", "capitalization"]
    fold_weights = _make_weights(n_folds, d_model)
    fold_cmds = _make_fold_cmds_dict(n_folds, d_model, fold_group_names)
    fold_scores = rng.uniform(0.6, 0.95, n_folds)
    constraint_cmds = _make_fold_cmds_dict(n_folds, d_model, fold_group_names)
    fig = plot_fold_generalization(
        fold_weights, fold_cmds, fold_scores, fold_group_names,
        layer=15, constraint_cmds=constraint_cmds,
    )
    assert isinstance(fig, go.Figure)


def test_plot_fold_generalization_with_save(tmp_path):
    rng = np.random.default_rng(42)
    n_folds, d_model = 4, 16
    fold_weights = _make_weights(n_folds, d_model)
    fold_group_names = ["format", "language", "emoji", "capitalization"]
    fold_cmds = _make_fold_cmds_dict(n_folds, d_model, fold_group_names)
    fold_scores = rng.uniform(0.6, 0.95, n_folds)
    constraint_cmds = _make_fold_cmds_dict(n_folds, d_model, fold_group_names)
    save_path = tmp_path / "fold_gen.html"
    fig = plot_fold_generalization(
        fold_weights, fold_cmds, fold_scores, fold_group_names,
        layer=15, constraint_cmds=constraint_cmds, save_path=save_path,
    )
    assert isinstance(fig, go.Figure)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
