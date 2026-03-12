"""Tests for phase1_linear_probing.steer module.

All tests use synthetic data and mocks — no GPU or real model needed.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from phase1_linear_probing.steer import (
    compute_steered_scr,
    load_steering_directions,
    run_steering_sweep,
    score_steered_output,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def d_model():
    return 16


@pytest.fixture
def synthetic_probe_result(d_model):
    """Minimal ProbeResult-like object for direction loading tests."""
    from sklearn.linear_model import LogisticRegression

    from phase1_linear_probing.probe import ProbeResult

    n_layers = 4
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((n_layers, d_model)).astype(np.float32)
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    weights = weights / norms

    # Use real LogisticRegression instances (picklable) instead of MagicMock
    clfs = []
    for _ in range(n_layers):
        clf = LogisticRegression()
        clf.classes_ = np.array([0, 1])
        clf.coef_ = rng.standard_normal((1, d_model)).astype(np.float64)
        clf.intercept_ = np.array([0.0])
        clfs.append(clf)

    return ProbeResult(
        cv_scores=pd.DataFrame(
            [
                {
                    "layer": i,
                    "roc_auc_mean": 0.8,
                    "roc_auc_std": 0.05,
                    "balanced_accuracy_mean": 0.75,
                    "balanced_accuracy_std": 0.06,
                }
                for i in range(n_layers)
            ]
        ),
        weights=weights,
        weights_raw=rng.standard_normal((n_layers, d_model)).astype(np.float32),
        biases=np.zeros(n_layers),
        scalers=[None] * n_layers,
        classifiers=clfs,
        pos_name="last_prompt",
        cv_mode="grouped",
        use_scaler=False,
    )


@pytest.fixture
def run_dir_with_results(tmp_path, synthetic_probe_result, d_model):
    """Create a tmp run dir with saved probe results and CMD files."""
    import joblib

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Save probe results in the expected format
    payload = {
        "results": {"last_prompt": synthetic_probe_result},
        "model_name": "test-model",
    }
    joblib.dump(payload, run_dir / "results_grouped_unscaled.joblib", compress=3)

    # Save fold CMDs (4 folds)
    rng = np.random.default_rng(99)
    fold_cmds = {}
    for name in ["type_a", "type_b", "type_c", "type_d"]:
        v = rng.standard_normal(d_model).astype(np.float32)
        fold_cmds[name] = v / np.linalg.norm(v)
    np.savez(run_dir / "fold_cmds_L2.npz", **fold_cmds)

    # Save per-constraint CMDs
    constraint_cmds = {}
    for name in ["format", "language", "emoji"]:
        v = rng.standard_normal(d_model).astype(np.float32)
        constraint_cmds[name] = v / np.linalg.norm(v)
    np.savez(run_dir / "constraint_cmds_L2.npz", **constraint_cmds)

    return run_dir


@pytest.fixture
def sweep_df():
    """Synthetic sweep results DataFrame."""
    rows = []
    for alpha in [-1.0, 0.0, 1.0, 3.0]:
        for i in range(10):
            # Higher alpha → more followed_system
            if alpha > 0:
                label = "followed_system" if i < 7 else "followed_user"
            elif alpha < 0:
                label = "followed_user" if i < 7 else "followed_system"
            else:
                label = "followed_system" if i < 5 else "followed_user"
            rows.append(
                {
                    "conflict_id": f"conflict_{i % 3}",
                    "direction": "a_to_b",
                    "constraint_type": f"type_{i % 3}",
                    "alpha": alpha,
                    "response": f"response at alpha={alpha}",
                    "label": label,
                    "confidence": 1.0,
                    "sys_ok": label in ("followed_system", "followed_both"),
                    "usr_ok": label in ("followed_user", "followed_both"),
                    "system_prompt": "sys",
                    "user_prompt": "usr",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def steering_samples_df():
    """Minimal Condition C samples for sweep tests."""
    return pd.DataFrame(
        [
            {
                "conflict_id": "formal_vs_casual_tone",
                "direction": "a_to_b",
                "constraint_type": "tone",
                "instruction_args": json.dumps({"topic": "photosynthesis"}),
                "system_prompt": "Respond formally.",
                "user_prompt": "Explain photosynthesis.",
            },
            {
                "conflict_id": "formal_vs_casual_tone",
                "direction": "b_to_a",
                "constraint_type": "tone",
                "instruction_args": json.dumps({"topic": "recipe"}),
                "system_prompt": "Respond casually.",
                "user_prompt": "Give me a recipe.",
            },
        ]
    )


# ── load_steering_directions ──────────────────────────────────────────────────


def test_load_steering_directions_probe_keys(run_dir_with_results):
    """Loaded directions contain probe and probe_raw keys."""
    dirs = load_steering_directions(run_dir_with_results, "last_prompt", layer=2)
    assert "probe" in dirs
    assert "probe_raw" in dirs


def test_load_steering_directions_probe_shape(run_dir_with_results, d_model):
    """Probe direction has correct shape."""
    dirs = load_steering_directions(run_dir_with_results, "last_prompt", layer=2)
    assert dirs["probe"].shape == (d_model,)
    assert dirs["probe_raw"].shape == (d_model,)


def test_load_steering_directions_probe_unit_norm(run_dir_with_results):
    """Probe direction is unit-normalized."""
    dirs = load_steering_directions(run_dir_with_results, "last_prompt", layer=2)
    np.testing.assert_allclose(np.linalg.norm(dirs["probe"]), 1.0, atol=1e-5)


def test_load_steering_directions_cmd_overall(run_dir_with_results, d_model):
    """cmd_overall is present and unit-normalized."""
    dirs = load_steering_directions(run_dir_with_results, "last_prompt", layer=2)
    assert "cmd_overall" in dirs
    assert dirs["cmd_overall"].shape == (d_model,)
    np.testing.assert_allclose(
        np.linalg.norm(dirs["cmd_overall"]), 1.0, atol=1e-5
    )


def test_load_steering_directions_cmd_per_constraint(run_dir_with_results, d_model):
    """cmd_per_constraint is a dict of vectors."""
    dirs = load_steering_directions(run_dir_with_results, "last_prompt", layer=2)
    assert "cmd_per_constraint" in dirs
    cpc = dirs["cmd_per_constraint"]
    assert isinstance(cpc, dict)
    assert len(cpc) == 3
    for k, v in cpc.items():
        assert v.shape == (d_model,)


def test_load_steering_directions_missing_cmds(tmp_path, synthetic_probe_result):
    """When CMD files are missing, probe keys still load and cmd keys are absent."""
    import joblib

    run_dir = tmp_path / "no_cmds"
    run_dir.mkdir()
    payload = {
        "results": {"last_prompt": synthetic_probe_result},
        "model_name": "test-model",
    }
    joblib.dump(payload, run_dir / "results_grouped_unscaled.joblib", compress=3)

    dirs = load_steering_directions(run_dir, "last_prompt", layer=0)
    assert "probe" in dirs
    assert "cmd_overall" not in dirs
    assert "cmd_per_constraint" not in dirs


# ── score_steered_output ─────────────────────────────────────────────────────


_PATCH_GET_CONFLICT = "phase0_v2.conflicts.registry.get_conflict"
_PATCH_CLASSIFY = "phase0_v2.src.classifiers.classify_response"


def test_score_steered_output_returns_tuple():
    """Returns (label, confidence, sys_ok, usr_ok) tuple."""
    with patch(_PATCH_GET_CONFLICT) as mock_gc, \
         patch(_PATCH_CLASSIFY) as mock_cr:
        mock_conflict = MagicMock()
        mock_gc.return_value = mock_conflict
        mock_cr.return_value = ("followed_system", 1.0)

        label, conf, sys_ok, usr_ok = score_steered_output(
            "response text",
            "formal_vs_casual_tone",
            "a_to_b",
            {"topic": "test"},
        )

        assert label == "followed_system"
        assert conf == 1.0
        assert sys_ok is True
        assert usr_ok is False


def test_score_steered_output_json_string_args():
    """instruction_args as JSON string is auto-parsed."""
    with patch(_PATCH_GET_CONFLICT) as mock_gc, \
         patch(_PATCH_CLASSIFY) as mock_cr:
        mock_gc.return_value = MagicMock()
        mock_cr.return_value = ("followed_user", 1.0)

        label, _, sys_ok, usr_ok = score_steered_output(
            "response",
            "formal_vs_casual_tone",
            "a_to_b",
            '{"topic": "test"}',
        )

        assert label == "followed_user"
        assert sys_ok is False
        assert usr_ok is True


def test_score_steered_output_followed_both():
    """followed_both sets both sys_ok and usr_ok to True."""
    with patch(_PATCH_GET_CONFLICT) as mock_gc, \
         patch(_PATCH_CLASSIFY) as mock_cr:
        mock_gc.return_value = MagicMock()
        mock_cr.return_value = ("followed_both", 0.5)

        label, conf, sys_ok, usr_ok = score_steered_output(
            "response", "test_conflict", "a_to_b", {},
        )

        assert label == "followed_both"
        assert sys_ok is True
        assert usr_ok is True


def test_score_steered_output_followed_neither():
    """followed_neither sets both sys_ok and usr_ok to False."""
    with patch(_PATCH_GET_CONFLICT) as mock_gc, \
         patch(_PATCH_CLASSIFY) as mock_cr:
        mock_gc.return_value = MagicMock()
        mock_cr.return_value = ("followed_neither", 1.0)

        _, _, sys_ok, usr_ok = score_steered_output(
            "response", "test_conflict", "a_to_b", {},
        )

        assert sys_ok is False
        assert usr_ok is False


def test_score_steered_output_unknown_conflict():
    """Unknown conflict_id raises ValueError."""
    with patch(_PATCH_GET_CONFLICT) as mock_gc:
        mock_gc.return_value = None

        with pytest.raises(ValueError, match="Unknown conflict_id"):
            score_steered_output(
                "response", "nonexistent_conflict", "a_to_b", {},
            )


# ── compute_steered_scr ──────────────────────────────────────────────────────


def test_compute_steered_scr_single_alpha(sweep_df):
    """SCR for alpha=0 returns a float."""
    scr = compute_steered_scr(sweep_df, alpha=0.0)
    assert isinstance(scr, float)
    assert 0.0 <= scr <= 1.0
    # alpha=0: 5 followed_system, 5 followed_user → SCR=0.5
    assert scr == pytest.approx(0.5)


def test_compute_steered_scr_positive_alpha(sweep_df):
    """Positive alpha should yield higher SCR than baseline."""
    scr_pos = compute_steered_scr(sweep_df, alpha=1.0)
    scr_zero = compute_steered_scr(sweep_df, alpha=0.0)
    assert scr_pos > scr_zero


def test_compute_steered_scr_negative_alpha(sweep_df):
    """Negative alpha should yield lower SCR than baseline."""
    scr_neg = compute_steered_scr(sweep_df, alpha=-1.0)
    scr_zero = compute_steered_scr(sweep_df, alpha=0.0)
    assert scr_neg < scr_zero


def test_compute_steered_scr_all_alphas(sweep_df):
    """Without alpha arg, returns a Series indexed by alpha."""
    scr_series = compute_steered_scr(sweep_df)
    assert isinstance(scr_series, pd.Series)
    assert len(scr_series) == 4
    assert set(scr_series.index) == {-1.0, 0.0, 1.0, 3.0}


def test_compute_steered_scr_invalid_alpha(sweep_df):
    """Requesting an alpha not in the data raises ValueError."""
    with pytest.raises(ValueError, match="No rows for alpha"):
        compute_steered_scr(sweep_df, alpha=999.0)


def test_compute_steered_scr_monotonicity(sweep_df):
    """SCR should increase with alpha in synthetic data."""
    scr = compute_steered_scr(sweep_df)
    assert scr[-1.0] < scr[0.0] < scr[1.0]


# ── run_steering_sweep ────────────────────────────────────────────────────────


def _patch_sweep():
    """Return patch context managers for the steer module's internal calls.

    We patch on the ``phase1_linear_probing.steer`` module object (the one
    that the test's ``from phase1_linear_probing.steer import ...`` imported
    ``run_steering_sweep`` from).
    """
    import phase1_linear_probing.steer as _steer_mod

    return (
        patch.object(_steer_mod, "steer_and_generate"),
        patch.object(_steer_mod, "score_steered_output"),
        patch.object(_steer_mod, "build_formatted_prompt",
                     create=True),
    )


def test_run_steering_sweep_output_columns(steering_samples_df, d_model):
    """Sweep returns DataFrame with expected columns."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    direction = np.random.default_rng(0).standard_normal(d_model).astype(np.float32)

    p_sag, p_sso, p_bfp = _patch_sweep()
    with p_sag as mock_sag, p_sso as mock_sso, p_bfp as mock_bfp:
        mock_bfp.return_value = "formatted prompt"
        mock_sag.return_value = "generated response"
        mock_sso.return_value = ("followed_system", 1.0, True, False)

        result = run_steering_sweep(
            mock_model,
            mock_tokenizer,
            steering_samples_df,
            direction,
            layer=2,
            alphas=[0.0, 1.0],
        )

    expected_cols = {
        "conflict_id", "direction", "constraint_type", "alpha",
        "response", "label", "confidence", "sys_ok", "usr_ok",
        "system_prompt", "user_prompt",
    }
    assert expected_cols == set(result.columns)
    # 2 samples × 2 alphas = 4 rows
    assert len(result) == 4


def test_run_steering_sweep_calls_per_sample(steering_samples_df, d_model):
    """Sweep calls steer_and_generate once per (alpha, sample) combination."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    direction = np.zeros(d_model, dtype=np.float32)

    p_sag, p_sso, p_bfp = _patch_sweep()
    with p_sag as mock_sag, p_sso as mock_sso, p_bfp as mock_bfp:
        mock_bfp.return_value = "prompt"
        mock_sag.return_value = "response"
        mock_sso.return_value = ("followed_system", 1.0, True, False)

        run_steering_sweep(
            mock_model, mock_tokenizer,
            steering_samples_df, direction,
            layer=0, alphas=[-1.0, 0.0, 1.0],
        )

    # 2 samples × 3 alphas = 6 calls
    assert mock_sag.call_count == 6
    assert mock_sso.call_count == 6


def test_run_steering_sweep_preserves_alpha(steering_samples_df, d_model):
    """Each row's alpha matches the sweep alpha it came from."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    direction = np.zeros(d_model, dtype=np.float32)

    p_sag, p_sso, p_bfp = _patch_sweep()
    with p_sag as mock_sag, p_sso as mock_sso, p_bfp as mock_bfp:
        mock_bfp.return_value = "prompt"
        mock_sag.return_value = "response"
        mock_sso.return_value = ("followed_user", 1.0, False, True)

        result = run_steering_sweep(
            mock_model, mock_tokenizer,
            steering_samples_df, direction,
            layer=0, alphas=[0.0, 5.0],
        )

    assert set(result["alpha"].unique()) == {0.0, 5.0}
    assert (result[result["alpha"] == 0.0].shape[0] == 2)
    assert (result[result["alpha"] == 5.0].shape[0] == 2)
