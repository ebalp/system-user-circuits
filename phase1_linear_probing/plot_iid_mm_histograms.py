"""Per-layer 1D histograms of activations projected onto the iid_mm overall direction.

Answers the steering questions directly:
  - Are the two classes well separated along the overall direction?
  - Where would alpha = N * proj_std land relative to the followed_system cloud?
  - Where would y1_mean + N_extra * y1_std targets land?

Per layer: one figure with a top "all" panel and per-conflict subplots, each showing
class-conditional histograms of (X @ unit_overall) with vertical reference lines
at y0_mean, y1_mean, and the candidate steering targets.

Usage:
  uv run python phase1_linear_probing/plot_iid_mm_histograms.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
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


_COLOR_USER = "#1f77b4"
_COLOR_SYS = "#d62728"


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else v / n


def _hist_panel(
    ax,
    proj: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    bins: int = 50,
    additive_alphas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    proj_targets: tuple[float, ...] = (-0.5, 0.0, 0.5, 1.0),
    proj_std: float = 1.0,
):
    p_user = proj[y == 0]
    p_sys = proj[y == 1]
    lo = float(min(proj.min(), 0.0)) - 0.5 * proj.std()
    hi = float(max(proj.max(), 0.0)) + max(additive_alphas) * proj_std + 0.5 * proj.std()
    edges = np.linspace(lo, hi, bins)

    if len(p_user):
        ax.hist(p_user, bins=edges, color=_COLOR_USER, alpha=0.55, label="followed_user")
    if len(p_sys):
        ax.hist(p_sys, bins=edges, color=_COLOR_SYS, alpha=0.55, label="followed_system")

    if len(p_user) and len(p_sys):
        m0 = float(p_user.mean()); m1 = float(p_sys.mean())
        s1 = float(p_sys.std())
        ax.axvline(m0, color=_COLOR_USER, lw=1.2, ls="--", label=f"y0_mean={m0:.2f}")
        ax.axvline(m1, color=_COLOR_SYS, lw=1.2, ls="--", label=f"y1_mean={m1:.2f}")

        # Projection-mode targets: y1_mean + N_extra * y1_std
        for N_extra in proj_targets:
            t = m1 + N_extra * s1
            ax.axvline(t, color="green", lw=0.8, ls=":", alpha=0.7)
            ax.text(
                t, ax.get_ylim()[1] * 0.95,
                f"+{N_extra}σ_y1" if N_extra >= 0 else f"{N_extra}σ_y1",
                rotation=90, fontsize=6, color="green",
                va="top", ha="right" if N_extra < 0 else "left",
            )

        # Additive-mode targets: m0 + N * proj_std (shift from the followed_user
        # cloud by N steering units; rough proxy for "where does an unsteered
        # user-following sample land after alpha=N steering").
        for N in additive_alphas:
            t = m0 + N * proj_std
            ax.axvline(t, color="purple", lw=0.6, ls="-.", alpha=0.6)
            ax.text(
                t, ax.get_ylim()[1] * 0.4,
                f"+{N}σ_p", rotation=90, fontsize=6, color="purple",
                va="top", ha="left",
            )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("proj on overall_iid_mm", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)


def make_layer_histogram_figure(
    layer: int,
    proj_all: np.ndarray,
    y: np.ndarray,
    conflict_ids: np.ndarray,
    proj_std: float,
    bal_acc: float | None = None,
) -> plt.Figure:
    unique = sorted(set(conflict_ids.tolist()))
    n = len(unique)
    n_cols = 2
    n_rows = 1 + (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(11, 2.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.3)

    top = fig.add_subplot(gs[0, :])
    suffix = f" | BA={bal_acc:.2f}" if bal_acc is not None else ""
    _hist_panel(
        top, proj_all, y,
        title=f"L{layer} — all (proj_std={proj_std:.3f}{suffix})",
        proj_std=proj_std,
    )
    top.legend(loc="upper right", fontsize=7)

    for i, cid in enumerate(unique):
        r = 1 + i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(gs[r, c])
        m = conflict_ids == cid
        _hist_panel(
            ax, proj_all[m], y[m],
            title=f"{cid} (n={int(m.sum())})",
            proj_std=proj_std,
        )

    fig.suptitle(f"iid_mm overall projection — layer {layer}", fontsize=11)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot iid_mm projection histograms")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--pos", default="last_prompt")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--direction", default="overall_iid_mm",
                        help="Direction key prefix: overall_iid_mm | mean_iid_mm")
    parser.add_argument("--also-png", action="store_true",
                        help="Also save per-layer PNGs (off by default)")
    args = parser.parse_args(argv)

    run_dir = _phase1_dir / "data" / "runs" / args.run_id
    cfg = load_run_config(run_dir)
    load_sync_env(cfg.repo_root)

    out_dir = args.out_dir if args.out_dir is not None else (run_dir / "iid_mm_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = run_dir / "iid_mm_directions.npz"
    json_path = run_dir / "iid_mm_proj_std.json"
    with np.load(npz_path) as f:
        directions = {k: f[k] for k in f.files}
    with open(json_path) as f:
        proj_std_map = json.load(f)

    available_layers = sorted({
        int(k.split("_L")[-1])
        for k in directions
        if k.startswith(args.direction + "_L") and "_zscored" not in k
    })
    layers = args.layers if args.layers else available_layers
    if not layers:
        raise ValueError("No layers found for direction prefix " + args.direction)

    df_all = load_phase0_results(cfg.data_dir, cfg.model_name)
    df = prepare_condition_c(df_all, cfg.label_mode, cfg.max_samples, cfg.conflict_ids)
    df = df.sort_values("conflict_id").reset_index(drop=True)
    y = df["y"].values.astype(int)
    conflict_ids = df["conflict_id"].values

    act_path = run_dir / f"act_nn_{cfg.safe_model_name}.npz"
    activations = load_activations(act_path)
    X_all = activations[args.pos]

    if X_all.shape[0] != len(y):
        raise ValueError(
            f"Activation rows ({X_all.shape[0]}) != Condition C samples ({len(y)})"
        )

    pdf_path = out_dir / f"iid_mm_hist_{args.direction}.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer in layers:
            key = f"{args.direction}_L{layer}"
            if key not in directions:
                print(f"Skipping L{layer}: missing key {key}")
                continue
            u = _unit(directions[key].astype(np.float32))
            Xl = np.ascontiguousarray(X_all[:, layer, :], dtype=np.float32)
            proj = (Xl @ u).astype(np.float32)
            proj_std = float(proj_std_map.get(key, float(proj.std())))

            if len(np.unique(y)) > 1:
                m1 = proj[y == 1].mean()
                m0 = proj[y == 0].mean()
                thr = 0.5 * (m1 + m0)
                pred = (proj >= thr).astype(int) if m1 >= m0 else (proj <= thr).astype(int)
                tp = float(((pred == 1) & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
                tn = float(((pred == 0) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
                ba = 0.5 * (tp + tn)
            else:
                ba = None

            fig = make_layer_histogram_figure(layer, proj, y, conflict_ids, proj_std, ba)
            pdf.savefig(fig, bbox_inches="tight")
            if args.also_png:
                png_path = out_dir / f"iid_mm_hist_{args.direction}_L{layer}.png"
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"L{layer} done", flush=True)
    print(f"Saved: {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
