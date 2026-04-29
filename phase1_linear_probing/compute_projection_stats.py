"""Compute per-(constraint, layer, direction) projection statistics.

Loads the big activations NPZ once and produces a lightweight JSON summary
that agents can use to understand baseline distributions without loading
the full ~1GB array.

Usage:
  uv run python phase1_linear_probing/compute_projection_stats.py --run-id curated35-8b-v001
  uv run python phase1_linear_probing/compute_projection_stats.py --run-id curated35-8b-v001 --layers 14 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_phase1_dir = Path(__file__).resolve().parent
_repo_root = _phase1_dir.parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_repo_root))

from compute_cmds import load_run_config
from data import (
    load_activations,
    load_results as load_phase0_results,
    load_sync_env,
    prepare_condition_c,
)
from steer import load_steering_directions


def _class_stats(values: np.ndarray) -> dict:
    """Compute summary statistics for a 1-D array of projection values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": int(len(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def compute_projection_stats(
    activations: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    directions_by_layer: dict[int, dict[str, np.ndarray]],
) -> dict:
    """Compute projection statistics for all (group, layer, direction) triples.

    Parameters
    ----------
    activations : (n_samples, n_layers, d_model)
    y : (n_samples,) binary labels (1 = followed_system)
    groups : (n_samples,) constraint_type labels
    directions_by_layer : {layer: {direction_name: (d_model,) vector}}

    Returns
    -------
    Nested dict: {group_or_overall: {L{layer}: {direction: {stats}}}}
    """
    mask_pos = y == 1
    mask_neg = y == 0
    unique_groups = sorted(np.unique(groups))

    results: dict = {}

    for layer, directions in sorted(directions_by_layer.items()):
        X = activations[:, layer, :]  # (n_samples, d_model)
        layer_key = f"L{layer}"

        # Project all samples onto all directions at once
        dir_names = list(directions.keys())
        D = np.stack([directions[n] for n in dir_names], axis=1)  # (d_model, k)
        projections = X @ D  # (n_samples, k)

        for di, dname in enumerate(dir_names):
            proj = projections[:, di]
            proj_pos = proj[mask_pos]
            proj_neg = proj[mask_neg]

            # Overall stats
            overall_entry = results.setdefault("_overall", {}).setdefault(layer_key, {})
            pooled_std = np.sqrt((proj_pos.std() ** 2 + proj_neg.std() ** 2) / 2)
            cohens_d = float((proj_pos.mean() - proj_neg.mean()) / pooled_std) if pooled_std > 0 else 0.0
            try:
                auc = float(roc_auc_score(y, proj))
            except ValueError:
                auc = 0.5
            overall_entry[dname] = {
                "y0": _class_stats(proj_neg),
                "y1": _class_stats(proj_pos),
                "cohens_d": cohens_d,
                "auc": auc,
                "baseline_range": [float(np.percentile(proj, 5)), float(np.percentile(proj, 95))],
            }

            # Per-constraint stats
            for gname in unique_groups:
                g_mask = groups == gname
                y_g = y[g_mask]
                g_pos = g_mask & mask_pos
                g_neg = g_mask & mask_neg
                if g_pos.sum() == 0 or g_neg.sum() == 0:
                    continue

                proj_g_pos = proj[g_pos]
                proj_g_neg = proj[g_neg]
                proj_g = proj[g_mask]

                ps = np.sqrt((proj_g_pos.std() ** 2 + proj_g_neg.std() ** 2) / 2)
                cd = float((proj_g_pos.mean() - proj_g_neg.mean()) / ps) if ps > 0 else 0.0
                try:
                    a = float(roc_auc_score(y_g, proj_g))
                except ValueError:
                    a = 0.5

                constraint_entry = results.setdefault(str(gname), {}).setdefault(layer_key, {})
                constraint_entry[dname] = {
                    "y0": _class_stats(proj_g_neg),
                    "y1": _class_stats(proj_g_pos),
                    "cohens_d": cd,
                    "auc": a,
                    "baseline_range": [float(np.percentile(proj_g, 5)), float(np.percentile(proj_g, 95))],
                }

    return results


def _select_directions(
    run_dir: Path, layer: int,
) -> dict[str, np.ndarray]:
    """Load a representative subset of directions for a layer.

    Includes: probe, cmd_overall, and per-constraint CMDs (normalized only).
    Skips probe_raw and cmd_raw_* to keep output manageable.
    """
    all_dirs = load_steering_directions(run_dir, "last_prompt", layer)
    selected: dict[str, np.ndarray] = {}

    for name, vec in all_dirs.items():
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            # Flat directions: probe, probe_raw, cmd_overall, cmd_overall_raw,
            # and iid_mm_overall / iid_mm_overall_zscored / iid_mm_mean.
            if name == "iid_mm_overall":
                selected["iid_mm"] = vec
            elif name == "iid_mm_overall_zscored":
                selected["iid_mm_zscored"] = vec
            elif name == "iid_mm_mean":
                selected["iid_mm_mean"] = vec
            else:
                selected[name] = vec
        elif isinstance(vec, dict):
            # iid_mm per-constraint dicts — include both rawspace and zscored.
            if name == "iid_mm_per_constraint":
                for cid, cvec in vec.items():
                    if isinstance(cvec, np.ndarray) and cvec.ndim == 1:
                        selected[f"iid_mm_{cid}"] = cvec
                continue
            if name == "iid_mm_per_constraint_zscored":
                for cid, cvec in vec.items():
                    if isinstance(cvec, np.ndarray) and cvec.ndim == 1:
                        selected[f"iid_mm_zscored_{cid}"] = cvec
                continue
            # CMD/probe per-constraint dicts (skip raw variants).
            if name.endswith("_raw"):
                continue
            if name.startswith("probe"):
                prefix = "probe"
            else:
                prefix = "cmd"
            for cname, cvec in vec.items():
                if isinstance(cvec, np.ndarray) and cvec.ndim == 1:
                    selected[f"{prefix}_{cname}"] = cvec

    return selected


def main():
    parser = argparse.ArgumentParser(description="Compute projection statistics")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to compute (default: all)")
    parser.add_argument("--pos", default="last_prompt",
                        help="Token position (default: last_prompt)")
    args = parser.parse_args()

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
    X = activations[args.pos]
    n_layers = X.shape[1]
    print(f"Shape: {X.shape} ({X.dtype})")

    layers = args.layers if args.layers else list(range(n_layers))

    # Load directions for each layer
    print(f"Loading directions for {len(layers)} layers...")
    directions_by_layer: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        directions_by_layer[layer] = _select_directions(run_dir, layer)
    n_dirs = len(next(iter(directions_by_layer.values())))
    print(f"  {n_dirs} directions per layer")

    # Compute stats
    print("Computing projection statistics...")
    t0 = time.time()
    stats = compute_projection_stats(X, y, groups, directions_by_layer)
    elapsed = time.time() - t0

    out_path = cfg.run_dir / "projection_stats.json"
    out_path.write_text(json.dumps(stats, indent=1) + "\n")
    size_mb = out_path.stat().st_size / 1e6
    n_groups = len(stats)
    print(f"Saved: {out_path}")
    print(f"  {n_groups} groups, {len(layers)} layers, {n_dirs} directions/layer")
    print(f"  {size_mb:.1f} MB, {elapsed:.1f}s")

    # Compute and save cosine similarity matrix between all directions at each layer
    print("Computing cosine similarity matrices...")
    cos_sim: dict = {}
    for layer, directions in sorted(directions_by_layer.items()):
        layer_key = f"L{layer}"
        names = sorted(directions.keys())
        D = np.stack([directions[n] for n in names], axis=1)  # (d_model, k)
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        D_normed = D / np.where(norms > 0, norms, 1)
        sim = (D_normed.T @ D_normed).tolist()  # (k, k)
        cos_sim[layer_key] = {"names": names, "matrix": sim}

    cos_path = cfg.run_dir / "direction_cosine_sim.json"
    cos_path.write_text(json.dumps(cos_sim, indent=1) + "\n")
    print(f"Saved: {cos_path} ({len(cos_sim)} layers)")


if __name__ == "__main__":
    main()
