"""Compute IID Mass-Mean (Fisher's discriminant) directions for steering & analysis.

Saves three outputs into the run directory:

- ``iid_mm_directions.npz`` — unit-norm direction vectors with keys:
    * ``overall_iid_mm_L{layer}``                      (global rawspace)
    * ``overall_iid_mm_zscored_L{layer}``              (global zscored)
    * ``mean_iid_mm_L{layer}``                         (mean of per-conflict rawspace, renormalized)
    * ``{conflict_id}_iid_mm_L{layer}``                (per-conflict rawspace)
    * ``{conflict_id}_iid_mm_zscored_L{layer}``        (per-conflict zscored)
- ``iid_mm_proj_std.json`` — std of projection of all curated condition C
  activations onto each rawspace direction (rawspace keys only).
- ``iid_mm_metrics.csv`` — columns: layer, scope, conflict_id, fit_basis, bal_acc.

Usage::

    uv run python phase1_linear_probing/compute_iid_mm.py --run-id curated4-8b-v002
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

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


# ── Core IID Mass-Mean fit ───────────────────────────────────────────────────


def fit_iid_mm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_basis: str = "rawspace",
    reg_eps_rawspace: float = 1e-3,
    reg_eps_zscored: float = 1e-6,
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    """Fit IID Mass-Mean (Fisher's discriminant) on a single layer.

    Parameters
    ----------
    X : (n_samples, d_model)
    y : (n_samples,) binary labels (1 = positive class).
    fit_basis : ``"rawspace"`` (no per-feature standardization) or
        ``"zscored"`` (zscore features before fitting).

    Returns
    -------
    direction_unit : (d_model,) unit-norm direction (in the original raw
        feature space). For zscored fit, the direction is converted back so
        that the same projection equation can be applied to raw activations:
        ``proj_z = (X - mu) / std @ eff_z``  is equivalent to
        ``proj_z = X @ (eff_z / std) + const``. Since we only need a
        direction (not a threshold), we return ``eff_z / std`` renormalized.
    bal_acc : float, balanced accuracy of the fitted projection on the
        training data using the midpoint threshold (matching reference
        implementation).
    """
    dev = _resolve_device(device)
    y_np = np.asarray(y).astype(bool)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev)
    y_t = torch.as_tensor(y_np, device=dev)
    d = Xt.shape[1]

    if fit_basis == "zscored":
        mu = Xt.mean(dim=0)
        std = Xt.std(dim=0).clamp(min=1e-12)
        Xf = (Xt - mu) / std
    elif fit_basis == "rawspace":
        Xf = Xt
        std = None
    else:
        raise ValueError(f"Unknown fit_basis: {fit_basis!r}")

    mu_pos = Xf[y_t].mean(dim=0)
    mu_neg = Xf[~y_t].mean(dim=0)
    theta_mm = mu_pos - mu_neg

    # Within-class centered features
    Xc = Xf.clone()
    Xc[y_t] -= mu_pos
    Xc[~y_t] -= mu_neg
    n = Xc.shape[0]
    Sigma = Xc.T @ Xc / max(n - 1, 1)

    if fit_basis == "rawspace":
        ridge = float(reg_eps_rawspace * Sigma.diagonal().mean().item())
    else:
        ridge = reg_eps_zscored
    Sigma = Sigma + torch.eye(d, dtype=Sigma.dtype, device=dev) * ridge

    # Solve on CPU (matches reference; CUDA linalg.solve can be unstable).
    eff = torch.linalg.solve(Sigma.cpu(), theta_mm.cpu()).to(dev)

    # Compute balanced accuracy in the basis used for fitting.
    proj = Xf @ eff
    threshold = (proj[y_t].mean() + proj[~y_t].mean()) / 2.0
    preds = (proj > threshold)
    tp = preds[y_t].float().mean().item() if y_np.any() else 0.0
    tn = (~preds[~y_t]).float().mean().item() if (~y_np).any() else 0.0
    bal_acc = float((tp + tn) / 2.0)

    # If we fit in zscored space, convert direction back to raw space.
    if fit_basis == "zscored":
        eff = eff / std

    norm = float(eff.norm().item())
    if norm < 1e-12:
        direction_unit = eff
    else:
        direction_unit = eff / norm

    return direction_unit.cpu().numpy().astype(np.float32), bal_acc


# ── Main computation ─────────────────────────────────────────────────────────


def compute_iid_mm_for_run(
    X: np.ndarray,
    y: np.ndarray,
    conflict_ids: np.ndarray,
    layers: list[int],
    *,
    reg_eps_rawspace: float = 1e-3,
    reg_eps_zscored: float = 1e-6,
    min_class_samples: int = 50,
    device: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], pd.DataFrame]:
    """Compute all IID-MM directions and return (directions, proj_std, metrics).

    Parameters
    ----------
    X : (n_samples, n_layers, d_model) activations at the chosen position.
    y : (n_samples,) binary labels.
    conflict_ids : (n_samples,) string array of conflict ids.
    layers : list of layer indices.
    """
    dev = _resolve_device(device)
    print(f"compute_iid_mm: device={dev}, layers={layers}")
    y = np.asarray(y).astype(bool)
    conflict_ids = np.asarray(conflict_ids)
    unique_conflicts = sorted(np.unique(conflict_ids).tolist())

    directions: dict[str, np.ndarray] = {}
    proj_std: dict[str, float] = {}
    metric_rows: list[dict] = []

    for li, layer in enumerate(layers):
        print(f"  L{layer} ({li+1}/{len(layers)})...", flush=True)
        Xl = X[:, layer, :]

        # Global rawspace fit
        d_overall_raw, ba_overall_raw = fit_iid_mm(
            Xl, y,
            fit_basis="rawspace",
            reg_eps_rawspace=reg_eps_rawspace,
            reg_eps_zscored=reg_eps_zscored,
            device=dev,
        )
        key_overall_raw = f"overall_iid_mm_L{layer}"
        directions[key_overall_raw] = d_overall_raw
        metric_rows.append({
            "layer": layer, "scope": "overall", "conflict_id": None,
            "fit_basis": "rawspace", "bal_acc": ba_overall_raw,
        })

        # Global zscored fit
        d_overall_z, ba_overall_z = fit_iid_mm(
            Xl, y,
            fit_basis="zscored",
            reg_eps_rawspace=reg_eps_rawspace,
            reg_eps_zscored=reg_eps_zscored,
            device=dev,
        )
        key_overall_z = f"overall_iid_mm_zscored_L{layer}"
        directions[key_overall_z] = d_overall_z
        metric_rows.append({
            "layer": layer, "scope": "overall", "conflict_id": None,
            "fit_basis": "zscored", "bal_acc": ba_overall_z,
        })

        # Per-conflict
        per_conflict_raw_dirs: list[np.ndarray] = []
        for cid in unique_conflicts:
            mask = conflict_ids == cid
            y_c = y[mask]
            n_pos = int(y_c.sum())
            n_neg = int((~y_c).sum())
            if n_pos < min_class_samples or n_neg < min_class_samples:
                continue
            X_c = Xl[mask]

            d_raw, ba_raw = fit_iid_mm(
                X_c, y_c,
                fit_basis="rawspace",
                reg_eps_rawspace=reg_eps_rawspace,
                reg_eps_zscored=reg_eps_zscored,
                device=dev,
            )
            key_raw = f"{cid}_iid_mm_L{layer}"
            directions[key_raw] = d_raw
            per_conflict_raw_dirs.append(d_raw)
            metric_rows.append({
                "layer": layer, "scope": "per_conflict", "conflict_id": cid,
                "fit_basis": "rawspace", "bal_acc": ba_raw,
            })

            d_z, ba_z = fit_iid_mm(
                X_c, y_c,
                fit_basis="zscored",
                reg_eps_rawspace=reg_eps_rawspace,
                reg_eps_zscored=reg_eps_zscored,
                device=dev,
            )
            key_z = f"{cid}_iid_mm_zscored_L{layer}"
            directions[key_z] = d_z
            metric_rows.append({
                "layer": layer, "scope": "per_conflict", "conflict_id": cid,
                "fit_basis": "zscored", "bal_acc": ba_z,
            })

        # Mean of per-conflict rawspace directions, renormalized
        if per_conflict_raw_dirs:
            stacked = np.stack(per_conflict_raw_dirs, axis=0)
            mean_dir = stacked.mean(axis=0)
            norm = float(np.linalg.norm(mean_dir))
            if norm > 1e-12:
                mean_dir = mean_dir / norm
            mean_key = f"mean_iid_mm_L{layer}"
            directions[mean_key] = mean_dir.astype(np.float32)
            metric_rows.append({
                "layer": layer, "scope": "mean", "conflict_id": None,
                "fit_basis": "rawspace", "bal_acc": float("nan"),
            })

    # Compute projection std for rawspace direction keys (no "_zscored"),
    # batched per layer on GPU.
    print("Computing projection stds...", flush=True)
    for layer in layers:
        keys = [
            k for k in directions
            if "_zscored" not in k and k.endswith(f"_L{layer}")
        ]
        if not keys:
            continue
        Xl_t = torch.as_tensor(X[:, layer, :], dtype=torch.float64, device=dev)
        W = torch.as_tensor(
            np.stack([directions[k] for k in keys], axis=1),
            dtype=torch.float64, device=dev,
        )
        projs = Xl_t @ W  # (n, k)
        stds = projs.std(dim=0).cpu().numpy()
        for k, s in zip(keys, stds):
            proj_std[k] = float(s)

    metrics_df = pd.DataFrame(metric_rows, columns=[
        "layer", "scope", "conflict_id", "fit_basis", "bal_acc",
    ])
    return directions, proj_std, metrics_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute IID Mass-Mean directions")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Layers to compute (default: all)")
    parser.add_argument("--pos", default="last_prompt",
                        help="Token position (default: last_prompt)")
    parser.add_argument("--reg-eps-rawspace", type=float, default=1e-3)
    parser.add_argument("--reg-eps-zscored", type=float, default=1e-6)
    parser.add_argument("--min-class-samples", type=int, default=50)
    args = parser.parse_args(argv)

    run_dir = _phase1_dir / "data" / "runs" / args.run_id
    cfg = load_run_config(run_dir)
    load_sync_env(cfg.repo_root)

    print(f"Loading phase 0 results for {cfg.model_name}...")
    df_all = load_phase0_results(cfg.data_dir, cfg.model_name)
    df = prepare_condition_c(df_all, cfg.label_mode, cfg.max_samples, cfg.conflict_ids)
    df = df.sort_values("conflict_id").reset_index(drop=True)
    y = df["y"].values
    conflict_ids = df["conflict_id"].values
    print(f"Samples: {len(y)}, conflicts: {len(np.unique(conflict_ids))}")

    act_path = cfg.run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    print(f"Loading activations: {act_path}")
    activations = load_activations(act_path)
    X = activations[args.pos]
    n_layers = X.shape[1]
    print(f"Shape: {X.shape} ({X.dtype})")

    layers = args.layers if args.layers else list(range(n_layers))

    directions, proj_std, metrics_df = compute_iid_mm_for_run(
        X, y, conflict_ids, layers,
        reg_eps_rawspace=args.reg_eps_rawspace,
        reg_eps_zscored=args.reg_eps_zscored,
        min_class_samples=args.min_class_samples,
    )

    # Save outputs
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cfg.run_dir / "iid_mm_directions.npz"
    np.savez(npz_path, **directions)
    print(f"Saved: {npz_path} ({len(directions)} directions)")

    proj_path = cfg.run_dir / "iid_mm_proj_std.json"
    with open(proj_path, "w") as f:
        json.dump(proj_std, f, indent=2)
    print(f"Saved: {proj_path} ({len(proj_std)} entries)")

    metrics_path = cfg.run_dir / "iid_mm_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path} ({len(metrics_df)} rows)")


if __name__ == "__main__":
    main()
