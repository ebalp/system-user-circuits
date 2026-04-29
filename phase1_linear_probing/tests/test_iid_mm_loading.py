"""Tests for IID-MM direction loading + server unpacking + scale loading.

All tests use synthetic data — no GPU or real model needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from phase1_linear_probing.probe import ProbeResult
from phase1_linear_probing.steer import load_steering_directions
from phase1_linear_probing.steering_server import (
    _load_direction_scales,
    _proj_std_npz_to_server_name,
    _unpack_directions,
)


D_MODEL = 16
N_LAYERS = 4
LAYERS = [1, 2]
CONFLICTS = ["format", "language"]


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_probe_result(d_model: int = D_MODEL, n_layers: int = N_LAYERS) -> ProbeResult:
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((n_layers, d_model)).astype(np.float32)
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
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
                    "layer": i, "roc_auc_mean": 0.8, "roc_auc_std": 0.05,
                    "balanced_accuracy_mean": 0.75, "balanced_accuracy_std": 0.06,
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


def _save_probe_results(run_dir: Path, pr: ProbeResult) -> None:
    payload = {"results": {"last_prompt": pr}, "model_name": "test-model"}
    joblib.dump(payload, run_dir / "results_grouped_unscaled.joblib", compress=3)


def _save_constraint_probes(run_dir: Path, layers: list[int]) -> None:
    """Minimal `constraint_probes.npz` (existing format) for graceful coexistence."""
    rng = np.random.default_rng(7)
    payload: dict[str, np.ndarray] = {}
    for layer in layers:
        for c in CONFLICTS:
            v = rng.standard_normal(D_MODEL).astype(np.float32)
            payload[f"{c}_probe_L{layer}"] = v / np.linalg.norm(v)
            payload[f"{c}_probe_raw_L{layer}"] = rng.standard_normal(D_MODEL).astype(np.float32)
    np.savez(run_dir / "constraint_probes.npz", **payload)


def _save_iid_mm_artifacts(run_dir: Path, layers: list[int]) -> None:
    """Save `iid_mm_directions.npz` and `iid_mm_proj_std.json`."""
    rng = np.random.default_rng(123)
    payload: dict[str, np.ndarray] = {}
    proj_std: dict[str, float] = {}
    for layer in layers:
        # overall (rawspace) and z-scored
        ov = rng.standard_normal(D_MODEL).astype(np.float32)
        ov_z = rng.standard_normal(D_MODEL).astype(np.float32)
        mean = rng.standard_normal(D_MODEL).astype(np.float32)
        payload[f"overall_iid_mm_L{layer}"] = ov
        payload[f"overall_iid_mm_zscored_L{layer}"] = ov_z
        payload[f"mean_iid_mm_L{layer}"] = mean
        proj_std[f"overall_iid_mm_L{layer}"] = 1.5 + layer
        proj_std[f"mean_iid_mm_L{layer}"] = 0.7 + layer
        for c in CONFLICTS:
            v = rng.standard_normal(D_MODEL).astype(np.float32)
            v_z = rng.standard_normal(D_MODEL).astype(np.float32)
            payload[f"{c}_iid_mm_L{layer}"] = v
            payload[f"{c}_iid_mm_zscored_L{layer}"] = v_z
            proj_std[f"{c}_iid_mm_L{layer}"] = 0.4 + layer
    np.savez(run_dir / "iid_mm_directions.npz", **payload)
    (run_dir / "iid_mm_proj_std.json").write_text(json.dumps(proj_std))


@pytest.fixture
def run_dir_with_iid_mm(tmp_path: Path) -> Path:
    run_dir = tmp_path / "iid_mm_run"
    run_dir.mkdir()
    _save_probe_results(run_dir, _make_probe_result())
    _save_constraint_probes(run_dir, LAYERS)
    _save_iid_mm_artifacts(run_dir, LAYERS)
    return run_dir


@pytest.fixture
def run_dir_without_iid_mm(tmp_path: Path) -> Path:
    run_dir = tmp_path / "no_iid_mm_run"
    run_dir.mkdir()
    _save_probe_results(run_dir, _make_probe_result())
    _save_constraint_probes(run_dir, LAYERS)
    return run_dir


# ── load_steering_directions: iid_mm presence ────────────────────────────────


def test_load_iid_mm_keys_present(run_dir_with_iid_mm: Path) -> None:
    """All 5 new keys present with correct types/shapes when artifacts exist."""
    dirs = load_steering_directions(run_dir_with_iid_mm, "last_prompt", layer=2)
    assert "iid_mm_overall" in dirs
    assert "iid_mm_overall_zscored" in dirs
    assert "iid_mm_mean" in dirs
    assert "iid_mm_per_constraint" in dirs
    assert "iid_mm_per_constraint_zscored" in dirs

    assert isinstance(dirs["iid_mm_overall"], np.ndarray)
    assert dirs["iid_mm_overall"].shape == (D_MODEL,)
    assert dirs["iid_mm_overall_zscored"].shape == (D_MODEL,)
    assert dirs["iid_mm_mean"].shape == (D_MODEL,)

    pc = dirs["iid_mm_per_constraint"]
    pcz = dirs["iid_mm_per_constraint_zscored"]
    assert isinstance(pc, dict) and isinstance(pcz, dict)
    assert set(pc.keys()) == set(CONFLICTS)
    assert set(pcz.keys()) == set(CONFLICTS)
    for v in pc.values():
        assert v.shape == (D_MODEL,)


def test_load_iid_mm_absent_graceful(run_dir_without_iid_mm: Path) -> None:
    """When iid_mm artifacts absent, no new keys present (other keys still load)."""
    dirs = load_steering_directions(run_dir_without_iid_mm, "last_prompt", layer=2)
    assert "probe" in dirs
    for k in (
        "iid_mm_overall",
        "iid_mm_overall_zscored",
        "iid_mm_mean",
        "iid_mm_per_constraint",
        "iid_mm_per_constraint_zscored",
    ):
        assert k not in dirs


# ── _unpack_directions: server-name mapping ──────────────────────────────────


def test_unpack_iid_mm_mapping() -> None:
    """Feed a synthetic load_steering_directions output, verify server names."""
    layer = 12
    flat = np.ones(D_MODEL, dtype=np.float32)
    dirs = {
        "iid_mm_overall": flat * 1,
        "iid_mm_overall_zscored": flat * 2,
        "iid_mm_mean": flat * 3,
        "iid_mm_per_constraint": {"format": flat * 4, "language": flat * 5},
        "iid_mm_per_constraint_zscored": {"format": flat * 6, "language": flat * 7},
    }
    out = _unpack_directions(dirs, layer)
    assert f"iid_mm_L{layer}" in out
    assert f"iid_mm_zscored_L{layer}" in out
    assert f"iid_mm_mean_L{layer}" in out
    assert f"iid_mm_format_L{layer}" in out
    assert f"iid_mm_language_L{layer}" in out
    assert f"iid_mm_zscored_format_L{layer}" in out
    assert f"iid_mm_zscored_language_L{layer}" in out
    # value identity preserved
    np.testing.assert_array_equal(out[f"iid_mm_L{layer}"], flat * 1)
    np.testing.assert_array_equal(out[f"iid_mm_format_L{layer}"], flat * 4)


def test_unpack_prefix_isolation(run_dir_with_iid_mm: Path) -> None:
    """Old + new artifacts coexist; iid_mm names never collide with probe/cmd names."""
    layer = 2
    dirs = load_steering_directions(run_dir_with_iid_mm, "last_prompt", layer=layer)
    out = _unpack_directions(dirs, layer)

    iid_mm_names = [k for k in out if k.startswith("iid_mm")]
    assert len(iid_mm_names) > 0  # we did populate iid_mm artifacts

    # No iid_mm name should ALSO appear with probe_/cmd_ prefix
    for name in iid_mm_names:
        assert not name.startswith("probe_")
        assert not name.startswith("cmd_")
    # And no probe_/cmd_ name accidentally embeds iid_mm
    for name in out:
        if name.startswith("probe_") or name.startswith("cmd_"):
            assert "iid_mm" not in name


def test_unpack_existing_probe_branch_unaffected() -> None:
    """The existing probe/cmd dispatch must still work when iid_mm absent."""
    layer = 5
    flat = np.ones(D_MODEL, dtype=np.float32)
    dirs = {
        "probe": flat,
        "probe_raw": flat * 2,
        "cmd_overall": flat * 3,
        "cmd_per_constraint": {"format": flat * 4},
        "probe_per_constraint": {"format": flat * 5},
    }
    out = _unpack_directions(dirs, layer)
    assert f"probe_L{layer}" in out
    assert f"probe_raw_L{layer}" in out
    assert f"cmd_overall_L{layer}" in out
    assert f"cmd_format_L{layer}" in out
    assert f"probe_format_L{layer}" in out


# ── Direction-scale loading + name translation ──────────────────────────────


def test_proj_std_npz_to_server_name() -> None:
    assert _proj_std_npz_to_server_name("overall_iid_mm_L12") == "iid_mm_L12"
    assert (
        _proj_std_npz_to_server_name("overall_iid_mm_zscored_L12")
        == "iid_mm_zscored_L12"
    )
    assert _proj_std_npz_to_server_name("mean_iid_mm_L12") == "iid_mm_mean_L12"
    assert (
        _proj_std_npz_to_server_name("format_iid_mm_L7")
        == "iid_mm_format_L7"
    )
    assert (
        _proj_std_npz_to_server_name("format_iid_mm_zscored_L7")
        == "iid_mm_zscored_format_L7"
    )
    assert _proj_std_npz_to_server_name("not_a_match") is None


def test_load_direction_scales_uses_server_names(run_dir_with_iid_mm: Path) -> None:
    scales = _load_direction_scales(run_dir_with_iid_mm)
    assert len(scales) > 0
    # All keys must be server-style names, never NPZ-style
    for k in scales:
        assert not k.startswith("overall_iid_mm")
        assert not k.startswith("mean_iid_mm")
        assert k.startswith("iid_mm_") or k.startswith("iid_mm_zscored_") or k.startswith("iid_mm_mean_") or k.startswith("iid_mm_L")
    # Spot-check specific entries
    assert "iid_mm_L1" in scales
    assert "iid_mm_L2" in scales
    assert "iid_mm_mean_L1" in scales
    assert "iid_mm_format_L2" in scales
    assert isinstance(scales["iid_mm_L1"], float)


def test_load_direction_scales_missing_returns_empty(run_dir_without_iid_mm: Path) -> None:
    assert _load_direction_scales(run_dir_without_iid_mm) == {}
