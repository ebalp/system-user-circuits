"""2D scatter of ~10k activations projected onto probe × CMD⊥ directions.

Axes: x = probe direction, y = CMD orthogonalized against probe (Gram-Schmidt).
Color: 4 conflict hues × red (followed_system) or blue (followed_user).
Marker: circle = a_to_b, x = b_to_a.
Ellipses: 2σ covariance ellipses per (conflict, label) group.
Overlays: unit steering vectors (probe & CMD at true angle), class centroids as stars.

Usage:  uv run python phase1_linear_probing/plot_probe_cmd_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

_phase1_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_phase1_dir))
sys.path.insert(0, str(_phase1_dir.parent))

from data import load_results as load_data, prepare_condition_c, load_sync_env
from probe import load_results, results_path

# ── Config ────────────────────────────────────────────────────────────────────
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 25
CONFLICT_IDS = [
    "json_only_vs_plain",
    "list_bullets_vs_numbered",
    "past_vs_present_tense",
    "starting_word_hello_greetings",
]

# Red hues for followed_system, blue hues for followed_user
COLORS_SYSTEM = [
    "rgba(220, 38, 38, 0.6)",   # red-600
    "rgba(239, 68, 68, 0.6)",   # red-500
    "rgba(248, 113, 113, 0.6)", # red-400
    "rgba(185, 28, 28, 0.6)",   # red-700
]
COLORS_USER = [
    "rgba(37, 99, 235, 0.6)",   # blue-600
    "rgba(59, 130, 246, 0.6)",  # blue-500
    "rgba(96, 165, 250, 0.6)",  # blue-400
    "rgba(29, 78, 216, 0.6)",   # blue-700
]

# Lighter versions for ellipse fill
COLORS_SYSTEM_FILL = [
    "rgba(220, 38, 38, 0.08)",
    "rgba(239, 68, 68, 0.08)",
    "rgba(248, 113, 113, 0.08)",
    "rgba(185, 28, 28, 0.08)",
]
COLORS_USER_FILL = [
    "rgba(37, 99, 235, 0.08)",
    "rgba(59, 130, 246, 0.08)",
    "rgba(96, 165, 250, 0.08)",
    "rgba(29, 78, 216, 0.08)",
]

SHORT_NAMES = {
    "json_only_vs_plain": "json_vs_plain",
    "list_bullets_vs_numbered": "bullets_vs_numbered",
    "past_vs_present_tense": "past_vs_present",
    "starting_word_hello_greetings": "hello_greetings",
}

N_SIGMA = 2  # ellipse covers ±2σ (~95%)
ELLIPSE_PTS = 100


def ellipse_coords(
    cx: float, cy: float, cov: np.ndarray, n_sigma: float = 2, n_pts: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y arrays tracing an n_sigma covariance ellipse."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Clamp tiny/negative eigenvalues
    eigvals = np.maximum(eigvals, 1e-12)
    angle = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

    t = np.linspace(0, 2 * np.pi, n_pts)
    a = n_sigma * np.sqrt(eigvals[1])
    b = n_sigma * np.sqrt(eigvals[0])

    xs = a * np.cos(t)
    ys = b * np.sin(t)

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_rot = cos_a * xs - sin_a * ys + cx
    y_rot = sin_a * xs + cos_a * ys + cy

    return x_rot, y_rot


def main():
    run_dir = _phase1_dir / "data" / "runs" / "curated4-8b-v001"

    # ── Load labels (sorted by conflict_id to align with activations) ─────
    load_sync_env(_phase1_dir.parent)
    df_all = load_data(
        Path("phase0_v2/data/results"),
        "meta-llama/Llama-3.1-8B-Instruct",
    )
    df_c = prepare_condition_c(df_all, "binary", conflict_ids=CONFLICT_IDS)
    df_c = df_c.sort_values("conflict_id").reset_index(drop=True)
    y = df_c["y"].values

    # ── Load activations ──────────────────────────────────────────────────
    act_data = np.load(run_dir / "act_nn_meta-llama_Llama-3.1-8B-Instruct.npz")
    X = act_data["last_prompt"][:, LAYER, :]

    # ── Load directions ───────────────────────────────────────────────────
    rpath = results_path(run_dir, cv_mode="grouped", use_scaler=False)
    pr = load_results(rpath)["last_prompt"]
    w_probe = pr.weights[LAYER]

    fold_cmds = np.load(run_dir / f"fold_cmds_L{LAYER}.npz")
    cmd_raw = np.stack([fold_cmds[k] for k in fold_cmds.files]).mean(axis=0)
    w_cmd = cmd_raw / np.linalg.norm(cmd_raw)

    # ── Gram-Schmidt: orthogonalize CMD against probe ─────────────────────
    cos_angle_raw = np.dot(w_probe, w_cmd)
    w_cmd_perp = w_cmd - cos_angle_raw * w_probe
    w_cmd_perp = w_cmd_perp / np.linalg.norm(w_cmd_perp)

    # Verify orthogonality
    dot_check = np.dot(w_probe, w_cmd_perp)
    angle_raw = np.degrees(np.arccos(np.clip(cos_angle_raw, -1, 1)))
    print(f"Original: cos(probe, CMD) = {cos_angle_raw:.4f}  →  {angle_raw:.1f}°")
    print(f"After GS: dot(probe, CMD⊥) = {dot_check:.2e}  (should be ~0)")

    # ── Project onto orthogonal basis ─────────────────────────────────────
    proj_x = X @ w_probe       # probe axis
    proj_y = X @ w_cmd_perp    # CMD⊥ axis

    # ── Build figure ──────────────────────────────────────────────────────
    fig = go.Figure()

    # Add ellipses first (behind points)
    for ci, cid in enumerate(CONFLICT_IDS):
        short = SHORT_NAMES[cid]
        mask_c = (df_c["conflict_id"] == cid).values

        for label_val, label_name, colors_line, colors_fill in [
            (1, "S", COLORS_SYSTEM, COLORS_SYSTEM_FILL),
            (0, "U", COLORS_USER, COLORS_USER_FILL),
        ]:
            mask = mask_c & (y == label_val)
            if mask.sum() < 3:
                continue

            px = proj_x[mask]
            py = proj_y[mask]
            cov = np.cov(px, py)
            ex, ey = ellipse_coords(px.mean(), py.mean(), cov, N_SIGMA, ELLIPSE_PTS)

            fig.add_trace(go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color=colors_line[ci], width=1.5, dash="dot"),
                fill="toself",
                fillcolor=colors_fill[ci],
                name=f"{short} {label_name} {N_SIGMA}σ",
                legendgroup=f"{cid}_{label_name}",
                showlegend=False,
                hoverinfo="skip",
            ))

    # Add scatter points on top
    for ci, cid in enumerate(CONFLICT_IDS):
        short = SHORT_NAMES[cid]
        mask_c = (df_c["conflict_id"] == cid).values

        for label_val, label_name, colors in [
            (1, "S", COLORS_SYSTEM),
            (0, "U", COLORS_USER),
        ]:
            mask_label = y == label_val
            for dir_val, dir_symbol in [("a_to_b", "circle"), ("b_to_a", "x")]:
                mask_dir = (df_c["direction"] == dir_val).values
                mask = mask_c & mask_label & mask_dir

                if mask.sum() == 0:
                    continue

                fig.add_trace(go.Scattergl(
                    x=proj_x[mask],
                    y=proj_y[mask],
                    mode="markers",
                    marker=dict(
                        symbol=dir_symbol,
                        size=4,
                        color=colors[ci],
                        line=dict(width=0.5, color=colors[ci]),
                    ),
                    name=f"{short} {label_name} {dir_val}",
                    legendgroup=f"{cid}_{label_name}",
                    legendgrouptitle_text=f"{short} ({label_name})" if dir_val == "a_to_b" else None,
                    hovertemplate=(
                        f"{short} | {label_name} | {dir_val}<br>"
                        "probe=%{x:.3f}<br>CMD⊥=%{y:.3f}<extra></extra>"
                    ),
                ))

    # ── Centroids and arrows (added last → rendered on top) ─────────────
    cx_s, cy_s = proj_x[y == 1].mean(), proj_y[y == 1].mean()
    cx_u, cy_u = proj_x[y == 0].mean(), proj_y[y == 0].mean()
    origin_x = (cx_s + cx_u) / 2
    origin_y = (cy_s + cy_u) / 2

    # Steering direction arrows from global centroid
    arrow_len = 1.0  # unit vectors
    theta_rad = np.radians(angle_raw)
    arrows = [
        ("probe", 1.0, 0.0, "green"),
        ("CMD", np.cos(theta_rad), np.sin(theta_rad), "orange"),
    ]
    for aname, dx, dy, color in arrows:
        fig.add_annotation(
            x=origin_x + dx * arrow_len,
            y=origin_y + dy * arrow_len,
            ax=origin_x, ay=origin_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor=color,
        )
        fig.add_annotation(
            x=origin_x + dx * arrow_len * 1.12,
            y=origin_y + dy * arrow_len * 1.12,
            text=f"<b>{aname}</b>",
            showarrow=False,
            font=dict(size=13, color=color),
        )

    # Global centroid (arrow origin)
    fig.add_trace(go.Scatter(
        x=[origin_x], y=[origin_y], mode="markers",
        marker=dict(symbol="circle", size=10, color="black",
                    line=dict(width=1.5, color="black")),
        name="Global centroid (arrow origin)",
        legendgroup="centroids",
        legendgrouptitle_text="Centroids",
    ))

    # Class centroids as stars — plotted last so they're on top
    fig.add_trace(go.Scatter(
        x=[cx_s], y=[cy_s], mode="markers",
        marker=dict(symbol="star", size=20, color="red",
                    line=dict(width=2, color="darkred")),
        name="Centroid S (followed_system)",
        legendgroup="centroids",
    ))
    fig.add_trace(go.Scatter(
        x=[cx_u], y=[cy_u], mode="markers",
        marker=dict(symbol="star", size=20, color="blue",
                    line=dict(width=2, color="darkblue")),
        name="Centroid U (followed_user)",
        legendgroup="centroids",
    ))

    fig.update_layout(
        title=(
            f"Activations in Probe × CMD⊥ space (Layer {LAYER})<br>"
            f"<sub>CMD orthogonalized via Gram-Schmidt — original angle {angle_raw:.1f}°, "
            f"ellipses at {N_SIGMA}σ, arrows = unit steering directions</sub>"
        ),
        xaxis_title="Probe direction",
        yaxis_title="CMD⊥ (orthogonal to probe)",
        width=1100,
        height=850,
        template="plotly_white",
        legend=dict(
            font=dict(size=10),
            itemsizing="constant",
        ),
    )

    out_path = _phase1_dir / f"probe_cmd_scatter_L{LAYER}.html"
    fig.write_html(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
