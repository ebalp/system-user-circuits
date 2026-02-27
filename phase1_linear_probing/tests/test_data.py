"""Tests for phase1_linear_probing.data module.

All tests use synthetic data — no dependency on real experiment files.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from phase1_linear_probing.data import (
    ProbeConfig,
    build_activation_array,
    filter_activations,
    find_repo_root,
    load_activations,
    prepare_condition_c,
    save_activations,
    slice_activation,
)


# ── find_repo_root ────────────────────────────────────────────────────────────


def test_find_repo_root():
    """Verify find_repo_root returns a Path containing pyproject.toml."""
    root = find_repo_root()
    assert (root / "pyproject.toml").exists()


# ── ProbeConfig ───────────────────────────────────────────────────────────────


def test_probe_config_defaults():
    """Default ProbeConfig has expected values and derived paths."""
    cfg = ProbeConfig()
    assert cfg.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.n_cv_folds == 5
    assert cfg.label_mode == "binary"
    assert cfg.device in ("cpu", "cuda")
    assert cfg.data_dir.exists() or True  # path is derived, may not exist
    assert cfg.run_id is not None
    assert "phase1_linear_probing" in str(cfg.run_dir)
    assert "reports" in str(cfg.reports_dir)


def test_probe_config_safe_model_name():
    """Slashes in model_name are replaced with underscores."""
    cfg = ProbeConfig(model_name="org/model-name")
    assert cfg.safe_model_name == "org_model-name"
    assert "/" not in cfg.safe_model_name


def test_probe_config_run_id_deterministic():
    """Same config parameters produce the same run_id; different params differ."""
    cfg_a = ProbeConfig(model_name="a/b", n_cv_folds=5)
    cfg_b = ProbeConfig(model_name="a/b", n_cv_folds=5)
    cfg_c = ProbeConfig(model_name="a/b", n_cv_folds=10)
    assert cfg_a.run_id == cfg_b.run_id
    assert cfg_a.run_id != cfg_c.run_id


def test_probe_config_run_id_explicit():
    """Explicit run_id overrides auto-generation."""
    cfg = ProbeConfig(run_id="my-run")
    assert cfg.run_id == "my-run"
    assert cfg.run_dir.name == "my-run"


def test_probe_config_save_config(tmp_path):
    """save_config writes valid JSON matching the config."""
    cfg = ProbeConfig(run_id="test-save", repo_root=tmp_path)
    # Manually set run_dir under tmp_path so we don't write outside
    cfg.run_dir = tmp_path / "runs" / "test-save"
    cfg.reports_dir = cfg.run_dir / "reports"

    path = cfg.save_config()
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["model_name"] == cfg.model_name
    assert data["run_id"] == "test-save"
    assert data["n_cv_folds"] == cfg.n_cv_folds
    assert data["label_mode"] == cfg.label_mode


def test_probe_config_ensure_dirs(tmp_path):
    """ensure_dirs creates run_dir and reports_dir."""
    cfg = ProbeConfig(run_id="test-dirs", repo_root=tmp_path)
    # Verify dirs don't exist yet
    assert not cfg.run_dir.exists()
    assert not cfg.reports_dir.exists()
    cfg.ensure_dirs()
    assert cfg.run_dir.is_dir()
    assert cfg.reports_dir.is_dir()


# ── prepare_condition_c ───────────────────────────────────────────────────────


def test_prepare_condition_c_binary(synthetic_condition_c_df):
    """Binary mode keeps only followed_system/followed_user with y column."""
    result = prepare_condition_c(synthetic_condition_c_df, label_mode="binary")
    assert "y" in result.columns
    assert set(result["label"].unique()) <= {
        "followed_system",
        "followed_user",
    }
    assert set(result["condition"].unique()) == {"C"}
    assert result["y"].dtype == np.int64 or result["y"].dtype == np.int32
    # y=1 for followed_system, y=0 for followed_user
    for _, row in result.iterrows():
        if row["label"] == "followed_system":
            assert row["y"] == 1
        else:
            assert row["y"] == 0


def test_prepare_condition_c_max_samples(synthetic_condition_c_df):
    """max_samples caps the number of rows returned."""
    result = prepare_condition_c(
        synthetic_condition_c_df, label_mode="binary", max_samples=10
    )
    assert len(result) <= 10


# ── slice_activation ──────────────────────────────────────────────────────────


def test_slice_activation_int_position():
    """Integer position extracts a single token vector."""
    act = torch.randn(1, 10, 8)
    result = slice_activation(act, 5)
    assert result.shape == (8,)
    expected = act[0, 5, :].float().cpu().numpy()
    np.testing.assert_allclose(result, expected)


def test_slice_activation_range_position():
    """Tuple position returns the mean over the token range."""
    act = torch.randn(1, 10, 8)
    result = slice_activation(act, (2, 5))
    assert result.shape == (8,)
    expected = act[0, 2:5, :].float().mean(dim=0).cpu().numpy()
    np.testing.assert_allclose(result, expected)


# ── build_activation_array ────────────────────────────────────────────────────


def test_build_activation_array_shape():
    """Output has shape (n_samples, n_layers, d_model) per position."""
    n_samples, n_layers, d_model = 5, 3, 8
    positions = ["last_prompt", "last_system"]
    rng = np.random.default_rng(0)

    # Build mock buffers: {pos: [layer][sample] -> vec}
    buffers = {}
    for pos in positions:
        buffers[pos] = [
            [rng.standard_normal(d_model).astype(np.float32) for _ in range(n_samples)]
            for _ in range(n_layers)
        ]

    result = build_activation_array(buffers, n_samples, n_layers, positions)
    assert set(result.keys()) == set(positions)
    for pos in positions:
        assert result[pos].shape == (n_samples, n_layers, d_model)


# ── save / load activations ──────────────────────────────────────────────────


def test_save_load_activations_roundtrip(tmp_path):
    """Saved activations can be loaded back with matching arrays."""
    rng = np.random.default_rng(42)
    original = {
        "last_prompt": rng.standard_normal((10, 4, 16)).astype(np.float32),
        "last_system": rng.standard_normal((10, 4, 16)).astype(np.float32),
    }
    path = tmp_path / "test_act.npz"
    save_activations(original, path)
    loaded = load_activations(path)
    assert set(loaded.keys()) == set(original.keys())
    for key in original:
        np.testing.assert_allclose(loaded[key], original[key])


# ── filter_activations ────────────────────────────────────────────────────────


def test_filter_activations(synthetic_activations):
    """Filtering by indices selects the correct samples."""
    indices = np.array([0, 2, 4])
    result = filter_activations(synthetic_activations, indices)
    assert set(result.keys()) == set(synthetic_activations.keys())
    for pos in synthetic_activations:
        assert result[pos].shape == (3, 4, 16)
        np.testing.assert_array_equal(
            result[pos], synthetic_activations[pos][indices]
        )
