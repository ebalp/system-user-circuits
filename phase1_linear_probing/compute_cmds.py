"""Compute raw and normalized CMDs for all layers, all constraints, and overall.

Saves a single ``constraint_cmds.npz`` in the run directory with keys:
  - ``{constraint}_L{layer}``       — unit-norm per-constraint CMD
  - ``{constraint}_L{layer}_raw``   — raw (unnormalized) per-constraint CMD
  - ``overall_L{layer}``            — unit-norm overall CMD (mean of all samples)
  - ``overall_L{layer}_raw``        — raw overall CMD

Usage:
  uv run python phase1_linear_probing/compute_cmds.py --run-id curated35-8b-v001
  uv run python phase1_linear_probing/compute_cmds.py --run-id curated35-8b-v001 --layers 14 15 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_phase1_dir = Path(__file__).resolve().parent
_repo_root = _phase1_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_repo_root))

from data import (
    ProbeConfig,
    load_activations,
    load_results as load_phase0_results,
    load_sync_env,
    prepare_condition_c,
)


def compute_all_cmds(
    activations: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    layers: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute raw and normalized CMDs for all layers and constraints.

    Parameters
    ----------
    activations : (n_samples, n_layers, d_model)
        Residual stream activations for a single token position.
    y : (n_samples,) binary labels (1 = followed_system).
    groups : (n_samples,) constraint type labels.
    layers : list of layer indices, or None for all layers.

    Returns
    -------
    dict ready to pass to ``np.savez``.
    """
    n_samples, n_layers, d_model = activations.shape
    if layers is None:
        layers = list(range(n_layers))

    mask_pos = y == 1
    mask_neg = y == 0
    unique_groups = np.unique(groups)

    results: dict[str, np.ndarray] = {}

    for layer in layers:
        X = activations[:, layer, :]  # (n_samples, d_model)

        # Overall CMD
        cmd_raw = X[mask_pos].mean(axis=0) - X[mask_neg].mean(axis=0)
        norm = np.linalg.norm(cmd_raw)
        results[f"overall_L{layer}_raw"] = cmd_raw
        results[f"overall_L{layer}"] = cmd_raw / norm if norm > 0 else cmd_raw

        # Per-constraint CMDs
        for group_name in unique_groups:
            g_mask = groups == group_name
            y_g = y[g_mask]
            if y_g.sum() == 0 or y_g.sum() == len(y_g):
                continue
            X_g = X[g_mask]
            cmd_raw = X_g[y_g == 1].mean(axis=0) - X_g[y_g == 0].mean(axis=0)
            norm = np.linalg.norm(cmd_raw)
            key = str(group_name)
            results[f"{key}_L{layer}_raw"] = cmd_raw
            results[f"{key}_L{layer}"] = cmd_raw / norm if norm > 0 else cmd_raw

    return results


def load_run_config(run_dir: Path) -> ProbeConfig:
    """Load ProbeConfig from a run's saved config.json."""
    import json
    config_path = run_dir / "config.json"
    with open(config_path) as f:
        d = json.load(f)
    return ProbeConfig(
        model_name=d["model_name"],
        label_mode=d.get("label_mode", "binary"),
        token_positions=d.get("token_positions", ["last_prompt"]),
        cv_mode=d.get("cv_mode", "grouped"),
        n_cv_folds=d.get("n_cv_folds", 5),
        max_samples=d.get("max_samples"),
        use_scaler=d.get("use_scaler", False),
        conflict_ids=d.get("conflict_ids"),
        batch_size=1,
        data_dir=Path(d["data_dir"]) if d.get("data_dir") else None,
        run_id=d["run_id"],
    )


def main():
    parser = argparse.ArgumentParser(description="Compute all CMDs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to compute (default: all)")
    parser.add_argument("--pos", default="last_prompt",
                        help="Token position (default: last_prompt)")
    args = parser.parse_args()

    # Build run_dir to find config.json, then load full config
    run_dir = _phase1_dir / "data" / "runs" / args.run_id
    cfg = load_run_config(run_dir)
    load_sync_env(cfg.repo_root)

    # Load phase 0 results and filter to condition C
    print(f"Loading phase 0 results for {cfg.model_name}...")
    df_all = load_phase0_results(cfg.data_dir, cfg.model_name)
    df = prepare_condition_c(df_all, cfg.label_mode, cfg.max_samples, cfg.conflict_ids)
    df = df.sort_values("conflict_id").reset_index(drop=True)
    y = df["y"].values
    groups = df["constraint_type"].values
    print(f"Samples: {len(y)}, constraints: {len(np.unique(groups))}")

    # Load activations
    act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    print(f"Loading activations: {act_path}")
    activations = load_activations(act_path)
    X = activations[args.pos]  # (n_samples, n_layers, d_model)
    n_layers = X.shape[1]
    print(f"Shape: {X.shape} ({X.dtype})")

    layers = args.layers if args.layers else list(range(n_layers))
    print(f"Computing CMDs for {len(layers)} layers...")

    t0 = time.time()
    cmds = compute_all_cmds(X, y, groups, layers=layers)
    elapsed = time.time() - t0

    out_path = cfg.run_dir / "constraint_cmds.npz"
    np.savez(out_path, **cmds)

    n_keys = len(cmds)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path}")
    print(f"  {n_keys} arrays, {size_mb:.1f} MB, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
