"""Tests for plot_iid_mm_geometry.py.

Synthetic-only — builds a tiny fake run dir mirroring the artifacts produced
by ``compute_iid_mm.py`` (Task A) and exercises the plotting entry point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# Make sure phase1 dir is importable as for other tests (conftest does this,
# but be explicit in case of isolated execution).
_phase1_dir = Path(__file__).resolve().parent.parent
_repo_root = _phase1_dir.parent
for _p in (_phase1_dir, _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plot_iid_mm_geometry as plot_mod
from plot_iid_mm_geometry import compute_axes, main


# Use 2 real conflict_ids so phase0 load_results filter doesn't drop them.
from phase0_v2.conflicts.registry import get_all_conflicts

_REAL_CONFLICTS = [c.conflict_id for c in get_all_conflicts()][:2]
assert len(_REAL_CONFLICTS) == 2


# ── Fixtures ────────────────────────────────────────────────────────────────


D_MODEL = 32
N_LAYERS = 2
N_SAMPLES = 200
MODEL_NAME = "test-org/test-model"
SAFE_MODEL = MODEL_NAME.replace("/", "_")


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _build_synthetic_run(
    tmp_path: Path,
    *,
    degenerate: bool = False,
) -> tuple[Path, Path]:
    """Build a fake run dir and a fake phase0 data dir.

    Returns (run_dir, data_dir).
    """
    rng = np.random.default_rng(123)

    # Phase 0 results dir + jsonl with Condition C records.
    data_dir = tmp_path / "phase0_data"
    data_dir.mkdir()

    records = []
    for i in range(N_SAMPLES):
        cid = _REAL_CONFLICTS[i % len(_REAL_CONFLICTS)]
        label = "followed_system" if i % 2 == 0 else "followed_user"
        records.append({
            "condition": "C",
            "label": label,
            "conflict_id": cid,
            "constraint_type": cid,
            "system_style": "bare",
            "user_style": "with_instruction",
            "task_id": "t",
            "direction": "a_to_b",
            "system_prompt": "sys",
            "user_prompt": "usr",
        })
    jsonl_path = data_dir / f"{SAFE_MODEL}_results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Run dir setup.
    run_id = "synthetic_test_run"
    run_dir = tmp_path / "phase1_runs" / run_id
    run_dir.mkdir(parents=True)

    # config.json — load_run_config will read this.
    config = {
        "model_name": MODEL_NAME,
        "label_mode": "binary",
        "token_positions": ["last_prompt"],
        "cv_mode": "grouped",
        "n_cv_folds": 5,
        "max_samples": None,
        "use_scaler": False,
        "conflict_ids": None,
        "data_dir": str(data_dir),
        "device": "cpu",
        "run_id": run_id,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Activations: (n_samples, n_layers, d_model) — sort by conflict_id like
    # production prepare_condition_c().sort_values("conflict_id").
    df_records = pd.DataFrame(records)
    sort_idx = df_records.sort_values("conflict_id").index.values
    X = rng.standard_normal((N_SAMPLES, N_LAYERS, D_MODEL)).astype(np.float32)
    # Realign so row order matches the sorted df rows.
    X_sorted = X[sort_idx]  # treat as if extraction was performed in that order
    np.savez(
        run_dir / f"act_nn_{SAFE_MODEL}.npz",
        last_prompt=X_sorted,
    )

    # iid_mm_directions.npz with required keys per layer.
    directions: dict[str, np.ndarray] = {}
    proj_std: dict[str, float] = {}
    for layer in range(N_LAYERS):
        overall = _unit(rng.standard_normal(D_MODEL).astype(np.float32))
        if degenerate:
            mean = overall.copy()
        else:
            mean = _unit(rng.standard_normal(D_MODEL).astype(np.float32))

        directions[f"overall_iid_mm_L{layer}"] = overall
        directions[f"overall_iid_mm_zscored_L{layer}"] = overall  # placeholder
        directions[f"mean_iid_mm_L{layer}"] = mean
        for cid in _REAL_CONFLICTS:
            v = _unit(rng.standard_normal(D_MODEL).astype(np.float32))
            directions[f"{cid}_iid_mm_L{layer}"] = v
            directions[f"{cid}_iid_mm_zscored_L{layer}"] = v
        proj_std[f"overall_iid_mm_L{layer}"] = 0.5
        proj_std[f"mean_iid_mm_L{layer}"] = 0.4
        for cid in _REAL_CONFLICTS:
            proj_std[f"{cid}_iid_mm_L{layer}"] = 0.3

    np.savez(run_dir / "iid_mm_directions.npz", **directions)
    with open(run_dir / "iid_mm_proj_std.json", "w") as f:
        json.dump(proj_std, f)

    return run_dir, data_dir


def _patch_runs_root(monkeypatch, run_dir: Path):
    """Make ``main()`` resolve --run-id to our tmp run_dir."""
    fake_runs_dir = run_dir.parent  # tmp_path/phase1_runs
    fake_phase1_data = fake_runs_dir.parent / "phase1_data"
    fake_phase1_data.mkdir(exist_ok=True)
    # The script computes: _phase1_dir / "data" / "runs" / args.run_id.
    # Easiest: monkeypatch _phase1_dir in plot_mod to a dir whose
    # data/runs/<id> -> our run_dir. Do this via symlink.
    fake_root = fake_runs_dir.parent / "fake_phase1"
    (fake_root / "data" / "runs").mkdir(parents=True, exist_ok=True)
    target = fake_root / "data" / "runs" / run_dir.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(run_dir)
    monkeypatch.setattr(plot_mod, "_phase1_dir", fake_root)


# ── Pure-helper unit tests ───────────────────────────────────────────────────


def test_compute_axes_normal_case():
    rng = np.random.default_rng(0)
    overall = _unit(rng.standard_normal(8))
    mean = _unit(rng.standard_normal(8))
    x, y, fb = compute_axes(overall, mean)
    assert not fb
    np.testing.assert_allclose(np.linalg.norm(x), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=1e-6)
    assert abs(float(np.dot(x, y))) < 1e-6


def test_compute_axes_degenerate_with_activations():
    rng = np.random.default_rng(1)
    overall = _unit(rng.standard_normal(8))
    mean = overall.copy()  # collapse
    X = rng.standard_normal((50, 8))
    x, y, fb = compute_axes(overall, mean, X)
    assert fb is True
    np.testing.assert_allclose(np.linalg.norm(x), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=1e-6)
    assert abs(float(np.dot(x, y))) < 1e-6


def test_compute_axes_degenerate_no_activations():
    rng = np.random.default_rng(2)
    overall = _unit(rng.standard_normal(8))
    mean = overall.copy()
    x, y, fb = compute_axes(overall, mean, None)
    assert fb is True
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=1e-6)
    assert abs(float(np.dot(x, y))) < 1e-6


# ── Integration: main() produces files ───────────────────────────────────────


def test_main_produces_png_and_pdf(tmp_path, monkeypatch):
    run_dir, _ = _build_synthetic_run(tmp_path, degenerate=False)
    _patch_runs_root(monkeypatch, run_dir)

    out_dir = tmp_path / "out"
    rc = main([
        "--run-id", run_dir.name,
        "--layers", "0", "1",
        "--pos", "last_prompt",
        "--out-dir", str(out_dir),
        "--also-png",
    ])
    assert rc == 0

    pdf = out_dir / "iid_mm_geometry.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0

    for layer in (0, 1):
        png = out_dir / f"iid_mm_geometry_L{layer}.png"
        assert png.exists() and png.stat().st_size > 0


def test_main_handles_degenerate(tmp_path, monkeypatch):
    """Degenerate case: mean == overall. Should not raise; figure produced."""
    run_dir, _ = _build_synthetic_run(tmp_path, degenerate=True)
    _patch_runs_root(monkeypatch, run_dir)

    out_dir = tmp_path / "out_deg"
    rc = main([
        "--run-id", run_dir.name,
        "--layers", "0",
        "--pos", "last_prompt",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    pdf = out_dir / "iid_mm_geometry.pdf"
    assert pdf.exists() and pdf.stat().st_size > 0
