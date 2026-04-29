"""Tests for phase1_linear_probing.compute_iid_mm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compute_iid_mm import compute_iid_mm_for_run, fit_iid_mm, main


# Two real, registered conflict ids (see phase0_v2.conflicts.registry).
CONFLICT_A = "imperative_vs_declarative"
CONFLICT_B = "address_reader_directly"


# ── Synthetic data fixtures ──────────────────────────────────────────────────


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def n_layers():
    return 2


@pytest.fixture
def n_per_conflict():
    # Big enough for IID-MM on d_model=64; spread across 2 conflicts.
    return 200


@pytest.fixture
def synthetic_iid_data(d_model, n_layers, n_per_conflict):
    """Two-class well-separated Gaussians in d_model dims, repeated for 2 conflicts.

    Returns (X, y, conflict_ids) where X has shape (n, n_layers, d_model).
    """
    rng = np.random.default_rng(0)

    direction = rng.standard_normal(d_model).astype(np.float32)
    direction /= np.linalg.norm(direction)

    # Per-conflict, generate balanced classes.
    Xs = []
    ys = []
    cids = []
    for cid in (CONFLICT_A, CONFLICT_B):
        n = n_per_conflict
        # Half pos, half neg.
        labels = np.zeros(n, dtype=int)
        labels[: n // 2] = 1
        rng.shuffle(labels)

        # Base noise: identity covariance.
        base = rng.standard_normal((n, d_model)).astype(np.float32)
        # Shift along direction by class.
        shift = np.where(labels[:, None] == 1, 3.0, -3.0).astype(np.float32)
        x = base + shift * direction[None, :]
        Xs.append(x)
        ys.append(labels)
        cids.extend([cid] * n)

    X_flat = np.concatenate(Xs, axis=0)  # (n_total, d_model)
    y = np.concatenate(ys, axis=0)
    conflict_ids = np.array(cids)

    # Replicate across layers with slight variation to make layers differ.
    X = np.stack([X_flat + 0.01 * layer for layer in range(n_layers)], axis=1)
    # X shape: (n_total, n_layers, d_model)
    assert X.shape == (len(y), n_layers, d_model)
    return X.astype(np.float32), y, conflict_ids


# ── Unit tests for fit_iid_mm ────────────────────────────────────────────────


def test_fit_iid_mm_rawspace_separates_classes(synthetic_iid_data):
    X, y, _ = synthetic_iid_data
    Xl = X[:, 0, :]
    direction, ba = fit_iid_mm(Xl, y, fit_basis="rawspace")
    assert direction.shape == (Xl.shape[1],)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-5
    assert ba > 0.9


def test_fit_iid_mm_zscored_separates_classes(synthetic_iid_data):
    X, y, _ = synthetic_iid_data
    Xl = X[:, 0, :]
    direction, ba = fit_iid_mm(Xl, y, fit_basis="zscored")
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-5
    assert ba > 0.9


def test_fit_iid_mm_rawspace_vs_zscored_close_ba(synthetic_iid_data):
    X, y, _ = synthetic_iid_data
    Xl = X[:, 0, :]
    _, ba_raw = fit_iid_mm(Xl, y, fit_basis="rawspace")
    _, ba_z = fit_iid_mm(Xl, y, fit_basis="zscored")
    # Fisher-invariance sanity: within ~5pp on this clean data.
    assert abs(ba_raw - ba_z) < 0.05


# ── Tests for compute_iid_mm_for_run ─────────────────────────────────────────


def test_compute_iid_mm_returns_expected_keys(synthetic_iid_data, n_layers):
    X, y, conflict_ids = synthetic_iid_data
    layers = list(range(n_layers))
    directions, proj_std, metrics = compute_iid_mm_for_run(
        X, y, conflict_ids, layers, min_class_samples=50,
    )

    for layer in layers:
        # Globals
        assert f"overall_iid_mm_L{layer}" in directions
        assert f"overall_iid_mm_zscored_L{layer}" in directions
        # Per-conflict
        for cid in (CONFLICT_A, CONFLICT_B):
            assert f"{cid}_iid_mm_L{layer}" in directions
            assert f"{cid}_iid_mm_zscored_L{layer}" in directions
        # Mean
        assert f"mean_iid_mm_L{layer}" in directions

    # All directions unit-norm
    for k, v in directions.items():
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5, f"{k} not unit-norm"

    # proj_std: only rawspace keys, all positive
    for k, v in proj_std.items():
        assert "_zscored" not in k
        assert v > 0
    # No zscored keys present in proj_std
    assert not any("_zscored" in k for k in proj_std)
    # All rawspace direction keys appear in proj_std
    for k in directions:
        if "_zscored" not in k:
            assert k in proj_std

    # Metrics CSV structure
    assert set(metrics.columns) == {
        "layer", "scope", "conflict_id", "fit_basis", "bal_acc",
    }
    # Has overall + per_conflict + mean rows for each layer
    for layer in layers:
        sub = metrics[metrics["layer"] == layer]
        assert ((sub["scope"] == "overall") & (sub["fit_basis"] == "rawspace")).any()
        assert ((sub["scope"] == "overall") & (sub["fit_basis"] == "zscored")).any()
        assert ((sub["scope"] == "per_conflict")).any()
        assert ((sub["scope"] == "mean")).any()


def test_compute_iid_mm_skips_conflict_under_min_samples(d_model, n_layers):
    """If a conflict has fewer than min_class_samples positives, it is skipped."""
    rng = np.random.default_rng(1)

    # Conflict A: 200 balanced samples (100 pos, 100 neg) — passes.
    nA = 200
    yA = np.concatenate([np.ones(nA // 2, dtype=int), np.zeros(nA // 2, dtype=int)])
    XA = rng.standard_normal((nA, d_model)).astype(np.float32)
    XA[yA == 1] += 3.0

    # Conflict B: only 10 positives (below min_class_samples=50) — skipped.
    nB_pos = 10
    nB_neg = 100
    yB = np.concatenate([np.ones(nB_pos, dtype=int), np.zeros(nB_neg, dtype=int)])
    XB = rng.standard_normal((nB_pos + nB_neg, d_model)).astype(np.float32)
    XB[yB == 1] += 3.0

    X_flat = np.concatenate([XA, XB], axis=0)
    y = np.concatenate([yA, yB])
    cids = np.array([CONFLICT_A] * nA + [CONFLICT_B] * (nB_pos + nB_neg))

    X = np.stack([X_flat for _ in range(n_layers)], axis=1)

    directions, proj_std, metrics = compute_iid_mm_for_run(
        X, y, cids, list(range(n_layers)), min_class_samples=50,
    )
    # Conflict A keys present
    assert f"{CONFLICT_A}_iid_mm_L0" in directions
    # Conflict B keys absent (skipped)
    assert f"{CONFLICT_B}_iid_mm_L0" not in directions
    assert f"{CONFLICT_B}_iid_mm_zscored_L0" not in directions
    # Mean still present (from conflict A only)
    assert "mean_iid_mm_L0" in directions
    # No proj_std for conflict B
    assert not any(CONFLICT_B in k for k in proj_std)


# ── End-to-end test via main() ───────────────────────────────────────────────


@pytest.fixture
def fake_run(tmp_path, synthetic_iid_data, d_model, n_layers, monkeypatch):
    """Build a fake run dir with config.json, activation .npz, and results.jsonl."""
    X, y, conflict_ids = synthetic_iid_data

    model_name = "test/test-model"
    safe_name = model_name.replace("/", "_")

    # Build run_dir under tmp_path/data/runs/<run_id>
    runs_root = tmp_path / "data" / "runs"
    run_id = "test-run"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)

    # data dir for phase0 results
    data_dir = tmp_path / "phase0_data"
    data_dir.mkdir()

    # Write a fake jsonl with rows aligned to (X, y, conflict_ids).
    rows = []
    for i in range(len(y)):
        rows.append({
            "condition": "C",
            "label": "followed_system" if y[i] == 1 else "followed_user",
            "conflict_id": conflict_ids[i],
        })
    jsonl_path = data_dir / f"{safe_name}_results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Write config.json
    config = {
        "model_name": model_name,
        "label_mode": "binary",
        "token_positions": ["last_prompt"],
        "cv_mode": "grouped",
        "n_cv_folds": 5,
        "max_samples": None,
        "use_scaler": False,
        "conflict_ids": [CONFLICT_A, CONFLICT_B],
        "data_dir": str(data_dir),
        "device": "cpu",
        "run_id": run_id,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Save activation npz at expected path. prepare_condition_c sorts by
    # conflict_id (in main()), so we save activations in conflict_id order.
    sort_idx = np.argsort(conflict_ids, kind="stable")
    X_sorted = X[sort_idx]
    np.savez(run_dir / f"act_nn_{safe_name}.npz", last_prompt=X_sorted)

    # Patch _phase1_dir + load_run_config so run_dir resolves under tmp_path.
    import compute_iid_mm as mod
    monkeypatch.setattr(mod, "_phase1_dir", tmp_path)

    real_load = mod.load_run_config

    def patched_load(rd):
        cfg = real_load(rd)
        cfg.run_dir = run_dir
        cfg.reports_dir = run_dir / "reports"
        cfg.data_dir = data_dir
        return cfg

    monkeypatch.setattr(mod, "load_run_config", patched_load)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "data_dir": data_dir,
        "model_name": model_name,
        "safe_name": safe_name,
        "n_layers": n_layers,
    }


def test_main_end_to_end(fake_run):
    main(["--run-id", fake_run["run_id"]])

    run_dir = fake_run["run_dir"]
    n_layers = fake_run["n_layers"]

    # NPZ present with expected keys
    npz_path = run_dir / "iid_mm_directions.npz"
    assert npz_path.exists()
    npz = np.load(npz_path)
    keys = set(npz.files)

    expected_keys = set()
    for layer in range(n_layers):
        expected_keys.add(f"overall_iid_mm_L{layer}")
        expected_keys.add(f"overall_iid_mm_zscored_L{layer}")
        expected_keys.add(f"mean_iid_mm_L{layer}")
        for cid in (CONFLICT_A, CONFLICT_B):
            expected_keys.add(f"{cid}_iid_mm_L{layer}")
            expected_keys.add(f"{cid}_iid_mm_zscored_L{layer}")
    assert expected_keys.issubset(keys)

    for k in keys:
        v = npz[k]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5, f"{k} not unit-norm"

    # proj_std json
    proj_path = run_dir / "iid_mm_proj_std.json"
    assert proj_path.exists()
    with open(proj_path) as f:
        proj = json.load(f)
    assert all("_zscored" not in k for k in proj)
    assert all(v > 0 for v in proj.values())
    # Each rawspace key present
    for k in keys:
        if "_zscored" not in k:
            assert k in proj

    # Metrics CSV
    metrics_path = run_dir / "iid_mm_metrics.csv"
    assert metrics_path.exists()
    df = pd.read_csv(metrics_path)
    assert set(df.columns) == {
        "layer", "scope", "conflict_id", "fit_basis", "bal_acc",
    }
    # Fisher-invariance sanity at global scope, layer 0
    sub = df[(df["layer"] == 0) & (df["scope"] == "overall")]
    ba_raw = float(sub[sub["fit_basis"] == "rawspace"]["bal_acc"].iloc[0])
    ba_z = float(sub[sub["fit_basis"] == "zscored"]["bal_acc"].iloc[0])
    assert abs(ba_raw - ba_z) < 0.05
