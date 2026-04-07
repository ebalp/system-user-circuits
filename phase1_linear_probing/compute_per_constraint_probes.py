"""Compute per-constraint probe directions for all layers and constraints.

Saves ``constraint_probes.npz`` in the run directory with keys:
  - ``{constraint}_probe_L{layer}``     — unit-norm per-constraint probe weight
  - ``{constraint}_probe_raw_L{layer}`` — raw (unnormalized) per-constraint probe weight

Also saves ``constraint_probes_cv.csv`` with per-layer ROC AUC scores.

When ``--search-C`` is given, tries multiple C values at the reference layer
(default L12), picks the best C per constraint, then uses that C for all layers.

Usage:
  uv run python phase1_linear_probing/compute_per_constraint_probes.py --run-id curated4-8b-v002
  uv run python phase1_linear_probing/compute_per_constraint_probes.py --run-id curated4-8b-v002 --search-C
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
                        help="Inverse regularization (default: 0.01). "
                             "Ignored when --search-C is used.")
    parser.add_argument("--search-C", action="store_true",
                        help="Search over C=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] "
                             "at --ref-layer, pick best per constraint, use for all layers.")
    parser.add_argument("--ref-layer", type=int, default=12,
                        help="Layer for C selection (default: 12)")
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
    n_constraints = len(np.unique(groups))
    print(f"Samples: {len(y)}, constraints: {n_constraints}")

    act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    print(f"Loading activations: {act_path}")
    activations = load_activations(act_path)
    X = activations[args.pos]
    n_layers = X.shape[1]
    print(f"Shape: {X.shape} ({X.dtype})")

    layers = args.layers if args.layers else list(range(n_layers))

    if args.search_C:
        # Step 1: Search C at reference layer
        c_candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        print(f"\nSearching C at L{args.ref_layer}: {c_candidates}")
        _, cv_df = probe_per_constraint(
            {args.pos: X}, y, groups, args.pos,
            layers=[args.ref_layer],
            probe_C=c_candidates,
            min_class_samples=args.min_class_samples,
        )

        # Print C search results
        print(f"\n{'constraint':<40s} {'best_C':>8s} {'AUC':>8s}")
        print("-" * 58)
        best_c_per_constraint: dict[str, float] = {}
        for cname in cv_df["constraint"].unique():
            sub = cv_df[(cv_df["constraint"] == cname) & (cv_df["layer"] == args.ref_layer)]
            best_row = sub.loc[sub["roc_auc_mean"].idxmax()]
            best_c_per_constraint[cname] = best_row["C"]
            print(f"  {cname:<38s} {best_row['C']:>8.3f} {best_row['roc_auc_mean']:>8.3f}")

        # Save full C search results
        cv_path = cfg.run_dir / "constraint_probes_c_search.csv"
        cv_df.to_csv(cv_path, index=False)
        print(f"\nC search results saved to {cv_path}")

        # Step 2: Train with best C per constraint for all layers
        # Use median best C as a single value (probe_per_constraint takes one C)
        # If constraints need different C values, we'd need per-constraint calls
        unique_best_cs = set(best_c_per_constraint.values())
        if len(unique_best_cs) == 1:
            best_c = unique_best_cs.pop()
            print(f"\nAll constraints agree: C={best_c}")
        else:
            best_c = float(np.median(list(best_c_per_constraint.values())))
            print(f"\nConstraints disagree on C, using median: C={best_c}")
            print(f"  Per-constraint: {best_c_per_constraint}")

        print(f"\nComputing probes for {len(layers)} layers with C={best_c}...")
        t0 = time.time()
        probes, cv_all = probe_per_constraint(
            {args.pos: X}, y, groups, args.pos,
            layers=layers, probe_C=best_c,
            min_class_samples=args.min_class_samples,
        )
        elapsed = time.time() - t0
    else:
        print(f"\nComputing per-constraint probes for {len(layers)} layers, "
              f"{n_constraints} constraints, C={args.probe_C}...")
        t0 = time.time()
        probes, cv_all = probe_per_constraint(
            {args.pos: X}, y, groups, args.pos,
            layers=layers, probe_C=args.probe_C,
            min_class_samples=args.min_class_samples,
        )
        elapsed = time.time() - t0

    # Save probe weights
    out_path = cfg.run_dir / "constraint_probes.npz"
    np.savez(out_path, **probes)
    n_keys = len(probes)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}")
    print(f"  {n_keys} arrays, {size_mb:.1f} MB, {elapsed:.1f}s")

    # Save CV scores
    cv_path = cfg.run_dir / "constraint_probes_cv.csv"
    cv_all.to_csv(cv_path, index=False)
    print(f"Saved: {cv_path}")

    # Print summary table
    print(f"\n{'constraint':<40s} {'n_layers':>8s} {'mean_AUC':>10s} {'best_layer':>10s} {'best_AUC':>10s}")
    print("-" * 80)
    for cname in sorted(cv_all["constraint"].unique()):
        sub = cv_all[cv_all["constraint"] == cname]
        mean_auc = sub["roc_auc_mean"].mean()
        best_row = sub.loc[sub["roc_auc_mean"].idxmax()]
        print(f"  {cname:<38s} {len(sub):>8d} {mean_auc:>10.3f} "
              f"L{int(best_row['layer']):>2d}       {best_row['roc_auc_mean']:>10.3f}")

    constraints = set()
    for key in probes:
        parts = key.rsplit("_probe_", 1)
        if len(parts) == 2:
            constraints.add(parts[0])
    print(f"\n  Constraints with probes: {sorted(constraints)}")


if __name__ == "__main__":
    main()
