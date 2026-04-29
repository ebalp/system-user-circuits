"""Pre-experiment visual sanity tool for iid_mm steering directions.

Consumes artifacts produced by ``compute_iid_mm.py``:
  - ``{run_dir}/iid_mm_directions.npz``
  - ``{run_dir}/iid_mm_proj_std.json``
  - ``{run_dir}/act_nn_{safe_model}.npz``

Produces per-layer 2D scatter figures of Condition C activations projected onto
the (overall iid_mm direction, mean_perp direction) plane, with vector overlays
for steering and per-conflict side panels.

Usage:
  uv run python phase1_linear_probing/plot_iid_mm_geometry.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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


# ── Pure helpers (unit-testable) ─────────────────────────────────────────────


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def compute_axes(
    overall: np.ndarray,
    mean: np.ndarray,
    activations_at_layer: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Compute (x_axis, y_axis, fallback_used) for the scatter plane.

    x_axis = unit(overall).
    y_axis = unit(mean - (mean . overall) * overall) — the component of
    ``mean`` orthogonal to ``overall``.

    If ``||y||`` is near zero (mean ~ overall), fall back to a principal-residual
    axis: top PC of activations after projecting ``overall`` out. If
    activations are not provided in that degenerate case, fall back to a
    deterministic orthonormal direction picked from the standard basis.

    Returns (x_axis, y_axis, fallback_used).
    """
    x_axis = _unit(np.asarray(overall, dtype=np.float64))
    mean = np.asarray(mean, dtype=np.float64)

    proj = float(np.dot(mean, x_axis))
    perp = mean - proj * x_axis
    norm_perp = float(np.linalg.norm(perp))

    if norm_perp >= eps:
        return x_axis, perp / norm_perp, False

    # Degenerate: mean direction collapses onto overall.
    fallback_used = True

    if activations_at_layer is not None and activations_at_layer.size > 0:
        X = np.asarray(activations_at_layer, dtype=np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        # remove component along x_axis
        Xc = Xc - np.outer(Xc @ x_axis, x_axis)
        # PC1 via SVD of (n, d_model); right singular vector
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            y_axis = _unit(vt[0])
            # ensure orthogonality numerically
            y_axis = _unit(y_axis - float(np.dot(y_axis, x_axis)) * x_axis)
            if np.linalg.norm(y_axis) >= eps:
                return x_axis, y_axis, fallback_used
        except np.linalg.LinAlgError:
            pass

    # Last-resort: pick the standard basis vector least aligned with x_axis.
    d = x_axis.shape[0]
    idx = int(np.argmin(np.abs(x_axis)))
    e = np.zeros(d, dtype=np.float64)
    e[idx] = 1.0
    y_axis = _unit(e - float(np.dot(e, x_axis)) * x_axis)
    return x_axis, y_axis, fallback_used


def project_onto_axes(
    X: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project (n, d) activations onto (x, y) coordinates."""
    return X @ x_axis, X @ y_axis


def balanced_accuracy_from_projection(
    proj_x: np.ndarray, y: np.ndarray
) -> float:
    """BA from 1D projection thresholded at midpoint between class means."""
    if len(np.unique(y)) < 2:
        return float("nan")
    m1 = proj_x[y == 1].mean()
    m0 = proj_x[y == 0].mean()
    thr = 0.5 * (m1 + m0)
    pred = (proj_x >= thr).astype(int) if m1 >= m0 else (proj_x <= thr).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    p = float((y == 1).sum())
    n = float((y == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    return 0.5 * (tp / p + tn / n)


def cohens_d(proj_x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    a = proj_x[y == 1]
    b = proj_x[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = math.sqrt((sa**2 + sb**2) / 2.0)
    if sp <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


# ── Plotting ─────────────────────────────────────────────────────────────────


_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]
_COLOR_USER = "#1f77b4"  # blue, y=0 (followed_user)
_COLOR_SYS = "#d62728"   # red, y=1 (followed_system)


def _scatter_panel(
    ax,
    px: np.ndarray,
    py: np.ndarray,
    y: np.ndarray,
    conflict_ids: np.ndarray,
    conflict_to_marker: dict[str, str],
    title: str,
    *,
    show_means: bool = True,
):
    for cid, marker in conflict_to_marker.items():
        m = conflict_ids == cid
        if not m.any():
            continue
        ms_user = m & (y == 0)
        ms_sys = m & (y == 1)
        if ms_user.any():
            ax.scatter(
                px[ms_user],
                py[ms_user],
                c=_COLOR_USER,
                marker=marker,
                s=20,
                alpha=0.6,
                edgecolors="none",
                label=f"{cid} (user)" if False else None,
            )
        if ms_sys.any():
            ax.scatter(
                px[ms_sys],
                py[ms_sys],
                c=_COLOR_SYS,
                marker=marker,
                s=20,
                alpha=0.6,
                edgecolors="none",
            )

    if show_means and len(np.unique(y)) > 1:
        m1 = px[y == 1].mean()
        m0 = px[y == 0].mean()
        ax.axvline(m1, color=_COLOR_SYS, linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(m0, color=_COLOR_USER, linestyle="--", linewidth=1, alpha=0.7)

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("proj on overall", fontsize=8)
    ax.set_ylabel("proj on mean_perp", fontsize=8)
    ax.tick_params(labelsize=7)


def make_layer_figure(
    layer: int,
    X_layer: np.ndarray,           # (n, d_model)
    y: np.ndarray,                 # (n,)
    conflict_ids: np.ndarray,      # (n,)
    overall: np.ndarray,           # (d_model,)
    mean: np.ndarray,              # (d_model,)
    proj_std_overall: float,
    proj_std_mean: float | None,
) -> tuple[plt.Figure, dict]:
    x_axis, y_axis, fallback = compute_axes(overall, mean, X_layer)
    px, py = project_onto_axes(X_layer, x_axis, y_axis)

    ba = balanced_accuracy_from_projection(px, y)
    d = cohens_d(px, y)

    unique_conflicts = sorted(set(conflict_ids.tolist()))
    conflict_to_marker = {
        cid: _MARKERS[i % len(_MARKERS)] for i, cid in enumerate(unique_conflicts)
    }

    n_conflicts = len(unique_conflicts)
    if n_conflicts == 0:
        n_rows = n_cols = 1
    else:
        n_cols = int(math.ceil(math.sqrt(n_conflicts)))
        n_rows = int(math.ceil(n_conflicts / n_cols))

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(
        nrows=max(1, n_rows),
        ncols=2 + n_cols,
        wspace=0.45,
        hspace=0.45,
    )
    main_ax = fig.add_subplot(gs[:, :2])

    fallback_tag = " | y=fallback" if fallback else ""
    main_title = (
        f"L{layer} | proj_std={proj_std_overall:.3f} "
        f"| Cohen's d={d:.2f} | BA={ba:.2f}{fallback_tag}"
    )
    _scatter_panel(
        main_ax, px, py, y, conflict_ids, conflict_to_marker, main_title
    )

    # Vector overlays from origin in (overall, mean_perp) basis.
    # Unit overall along X, length 1.
    main_ax.annotate(
        "", xy=(1.0, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    main_ax.text(1.02, 0.0, "u_overall", fontsize=8)

    # Unit mean_perp along Y, length 1.
    main_ax.annotate(
        "", xy=(0.0, 1.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    main_ax.text(0.02, 1.02, "u_mean_perp", fontsize=8)

    # Steering vector at alpha=1 along overall axis: length proj_std_overall.
    main_ax.annotate(
        "", xy=(proj_std_overall, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="purple", lw=2.0),
    )
    main_ax.text(
        proj_std_overall, -0.05, f"steer_overall ({proj_std_overall:.2f})",
        fontsize=8, color="purple",
    )

    # Steering vector for mean direction: project (proj_std_mean * unit(mean))
    # onto (x_axis, y_axis) basis.
    if proj_std_mean is not None:
        u_mean = _unit(mean)
        steer_mean = proj_std_mean * u_mean
        sx = float(np.dot(steer_mean, x_axis))
        sy = float(np.dot(steer_mean, y_axis))
        main_ax.annotate(
            "", xy=(sx, sy), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.0),
        )
        main_ax.text(
            sx, sy + 0.05, f"steer_mean ({proj_std_mean:.2f})",
            fontsize=8, color="green",
        )

    # Side panels: per-conflict scatters.
    for i, cid in enumerate(unique_conflicts):
        r = i // n_cols
        c = 2 + (i % n_cols)
        ax = fig.add_subplot(gs[r, c])
        m = conflict_ids == cid
        if not m.any():
            ax.set_visible(False)
            continue
        single_marker = {cid: conflict_to_marker[cid]}
        _scatter_panel(
            ax, px[m], py[m], y[m], conflict_ids[m], single_marker,
            title=f"{cid} (n={int(m.sum())})", show_means=True,
        )

    fig.suptitle(
        f"iid_mm geometry — layer {layer}"
        + (" — DEGENERATE/fallback Y axis" if fallback else ""),
        fontsize=11,
    )

    info = {
        "fallback_used": fallback,
        "ba": ba,
        "cohens_d": d,
        "proj_std_overall": proj_std_overall,
    }
    return fig, info


# ── Discovery / loading ──────────────────────────────────────────────────────


def discover_layers(directions: dict[str, np.ndarray]) -> list[int]:
    layers: set[int] = set()
    for k in directions.keys():
        if k.startswith("overall_iid_mm_L") and not k.endswith("_zscored"):
            try:
                layers.add(int(k.split("_L")[-1]))
            except ValueError:
                pass
    return sorted(layers)


def load_iid_mm_artifacts(
    run_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    npz_path = run_dir / "iid_mm_directions.npz"
    json_path = run_dir / "iid_mm_proj_std.json"
    with np.load(npz_path) as f:
        directions = {k: f[k] for k in f.files}
    with open(json_path) as f:
        proj_std = json.load(f)
    return directions, proj_std


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot iid_mm geometry")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--pos", default="last_prompt")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=1000,
                        help="Stratified downsample to this many scatter points (default 1000; 0 = no downsample)")
    parser.add_argument("--also-png", action="store_true",
                        help="Also save per-layer PNGs (off by default; one PDF is the default output)")
    args = parser.parse_args(argv)

    run_dir = _phase1_dir / "data" / "runs" / args.run_id
    cfg = load_run_config(run_dir)
    load_sync_env(cfg.repo_root)

    out_dir = args.out_dir if args.out_dir is not None else (run_dir / "iid_mm_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    directions, proj_std = load_iid_mm_artifacts(run_dir)

    available_layers = discover_layers(directions)
    layers = args.layers if args.layers else available_layers
    if not layers:
        raise ValueError("No layers found in iid_mm_directions.npz")

    # Load phase0 results (Condition C) and align to activations.
    df_all = load_phase0_results(cfg.data_dir, cfg.model_name)
    df = prepare_condition_c(
        df_all, cfg.label_mode, cfg.max_samples, cfg.conflict_ids
    )
    df = df.sort_values("conflict_id").reset_index(drop=True)
    y = df["y"].values
    conflict_ids = df["conflict_id"].values

    act_path = run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    activations = load_activations(act_path)
    X_all = activations[args.pos]  # (n_samples, n_layers, d_model)

    if X_all.shape[0] != len(y):
        raise ValueError(
            f"Activation rows ({X_all.shape[0]}) != Condition C samples ({len(y)})"
        )

    # Stratified downsample by (conflict_id, y) for plotting.
    if args.max_points and args.max_points > 0 and len(y) > args.max_points:
        rng = np.random.default_rng(0)
        n_total = len(y)
        groups: dict[tuple, list[int]] = {}
        for i in range(n_total):
            groups.setdefault((conflict_ids[i], int(y[i])), []).append(i)
        keep_idx: list[int] = []
        for key, idxs in groups.items():
            quota = max(1, int(round(args.max_points * len(idxs) / n_total)))
            quota = min(quota, len(idxs))
            chosen = rng.choice(idxs, size=quota, replace=False)
            keep_idx.extend(chosen.tolist())
        keep_idx = sorted(keep_idx)
        keep_idx = np.array(keep_idx, dtype=np.int64)
        X_all = np.ascontiguousarray(X_all[keep_idx])
        y = y[keep_idx]
        conflict_ids = conflict_ids[keep_idx]
        print(f"Downsampled from {n_total} to {len(y)} points for scatter")

    pdf_path = out_dir / "iid_mm_geometry.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer in layers:
            overall_key = f"overall_iid_mm_L{layer}"
            mean_key = f"mean_iid_mm_L{layer}"
            if overall_key not in directions or mean_key not in directions:
                print(f"Skipping L{layer}: missing keys")
                continue

            overall = directions[overall_key]
            mean = directions[mean_key]
            proj_std_overall = float(proj_std.get(overall_key, 0.0))
            proj_std_mean = (
                float(proj_std[mean_key]) if mean_key in proj_std else None
            )

            X_layer = X_all[:, layer, :]
            fig, info = make_layer_figure(
                layer=layer,
                X_layer=X_layer,
                y=y,
                conflict_ids=conflict_ids,
                overall=overall,
                mean=mean,
                proj_std_overall=proj_std_overall,
                proj_std_mean=proj_std_mean,
            )

            pdf.savefig(fig, bbox_inches="tight")
            if args.also_png:
                png_path = out_dir / f"iid_mm_geometry_L{layer}.png"
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"L{layer} done (fallback={info['fallback_used']})", flush=True)
    print(f"Saved: {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
