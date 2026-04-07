"""Compute per-constraint probe directions for all layers and constraints.

Saves ``constraint_probes.npz`` in the run directory with keys:
  - ``{constraint}_probe_L{layer}``     — unit-norm per-constraint probe weight
  - ``{constraint}_probe_raw_L{layer}`` — raw (unnormalized) per-constraint probe weight

Usage:
  uv run python phase1_linear_probing/compute_per_constraint_probes.py --run-id curated4-8b-v002
  uv run python phase1_linear_probing/compute_per_constraint_probes.py --run-id curated4-8b-v002 --layers 4 12 14
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

from compute_cmds import load_run_config
from data import load_activations, load_results as load_phase0_results, load_sync_env, prepare_condition_c
from probe import probe_per_constraint


def main():
    parser = argparse.ArgumentParser(description="Compute per-constraint probes")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to compute (default: all)")
    parser.add_argument("--pos", default="last_prompt",
                        help="Token position (default: last_prompt)")
    parser.add_argument("--probe-C", type=float, default=0.01,
                        help="Inverse regularization. Per-constraint probes are "
                             "underdetermined (d_model=4096 >> minority class ~200), "
                             "so strong regularization (low C) is needed. "
                             "Default 0.01; try 0.001-0.1.")
    parser.add_argument("--min-class-samples", type=int, default=20)
    args = parser.parse_args()

    run_dir = _phase1_dir / "data" / "runs" / args.run_id
    cfg = load_run_config(run_dir)
    load_sync_env(cfg.repo_root)

    print(f"Loading phase 0 results for {cfg.model_name}...")
    df_all = load_phase0_results(cfg.data_dir, cfg.model_name)
    df = prepare_condition_c(df_all, cfg.label_mode, cfg.max_samples, cfg.conflict_ids)
    df = df.sort_values("conflict_id").reset_index(drop=True)
    y = df["y"].values
    groups = df["constraint_type"].values
    print(f"Samples: {len(y)}, constraints: {len(np.unique(groups))}")

    act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    print(f"Loading activations: {act_path}")
    activations = load_activations(act_path)
    X = activations[args.pos]
    n_layers = X.shape[1]
    print(f"Shape: {X.shape} ({X.dtype})")

    layers = args.layers if args.layers else list(range(n_layers))
    print(f"Computing per-constraint probes for {len(layers)} layers, "
          f"{len(np.unique(groups))} constraints...")

    t0 = time.time()
    probes = probe_per_constraint(
        {args.pos: X},
        y,
        groups,
        args.pos,
        layers=layers,
        probe_C=args.probe_C,
        min_class_samples=args.min_class_samples,
    )
    elapsed = time.time() - t0

    out_path = cfg.run_dir / "constraint_probes.npz"
    np.savez(out_path, **probes)

    n_keys = len(probes)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path}")
    print(f"  {n_keys} arrays, {size_mb:.1f} MB, {elapsed:.1f}s")

    # Summary
    constraints = set()
    for key in probes:
        parts = key.rsplit("_probe_", 1)
        if len(parts) == 2:
            constraints.add(parts[0])
    print(f"  Constraints with probes: {sorted(constraints)}")


if __name__ == "__main__":
    main()
