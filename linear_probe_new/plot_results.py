#!/usr/bin/env python3
"""
plot_results.py — Generate all figures from saved run_probing.py results.

Reads metrics JSONs, directions .npy, and activations to produce every plot
that was in the original notebooks. No probing is re-run.

Usage:
    python -u plot_results.py \
        --results-dir ./results/llama-8b \
        --act-dir /workspace/activations/meta-llama_Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_METHODS = ["dim", "iid_mm", "lr"]
METHOD_LABELS = {"dim": "Diff-in-Mean", "iid_mm": "IID Mass-Mean", "lr": "Logistic Reg."}

COLORS = {
    "last_prompt":     "rgba(31,119,180,1)",
    "last_system":     "rgba(44,160,44,1)",
    "last_user":       "rgba(148,103,189,1)",
    "first_generated": "rgba(255,127,14,1)",
    "mid_generated":   "rgba(227,119,194,1)",
    "last_generated":  "rgba(214,39,40,1)",
}

METHOD_COLORS = {
    "dim":    "rgba(31,119,180,1)",
    "iid_mm": "rgba(255,127,14,1)",
    "lr":     "rgba(44,160,44,1)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same auto-detect as run_probing.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_activations(act_dir):
    """Load activations with auto-detect for both naming conventions."""
    act_dir = Path(act_dir)
    KNOWN_POSITIONS = [
        "last_prompt", "last_system", "last_user",
        "first_generated", "mid_generated", "last_generated",
    ]

    activations = {}
    for f in sorted(act_dir.glob("activations_*.npy")):
        for pos in KNOWN_POSITIONS:
            if f.stem.endswith(f"_{pos}"):
                activations[pos] = np.load(f)
                print(f"  Loaded {pos}: {activations[pos].shape}")
                break

    # Labels
    if (act_dir / "labels.npz").exists():
        y = np.load(act_dir / "labels.npz")["y"]
    else:
        y_files = sorted(act_dir.glob("y_*.npy"))
        y = np.load(y_files[0])

    # Metadata
    if (act_dir / "metadata.csv").exists():
        df = pd.read_csv(act_dir / "metadata.csv")
    else:
        meta_files = sorted(act_dir.glob("meta_*.csv"))
        df = pd.read_csv(meta_files[0])

    positions = sorted(activations.keys())
    n_layers = activations[positions[0]].shape[1]
    d_model = activations[positions[0]].shape[2]

    return activations, y, df, positions, n_layers, d_model


def load_directions(results_dir, positions, n_layers):
    """Load all saved direction vectors."""
    art_dir = Path(results_dir) / "artifacts" / "directions"
    directions = {}
    for method in DIRECTION_METHODS:
        directions[method] = {}
        for pos in positions:
            dirs = []
            for layer in range(n_layers):
                d = np.load(art_dir / f"{method}_{pos}_L{layer}.npy")
                dirs.append(d)
            directions[method][pos] = dirs
    return directions


def save_fig(fig, path_stem, formats=("png", "pdf")):
    for fmt in formats:
        fig.write_image(f"{path_stem}.{fmt}", scale=2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Layer-wise accuracy (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_accuracy(metrics_df, positions, n_layers, fig_dir, fmts):
    print("  Layer-wise accuracy...")
    for method in DIRECTION_METHODS:
        fig = go.Figure()
        for pos in positions:
            mask = (metrics_df["method"] == method) & (metrics_df["position"] == pos)
            df_m = metrics_df[mask].sort_values("layer")
            col = "bal_acc" if "bal_acc" in df_m.columns else "mean"
            fig.add_trace(go.Scatter(
                x=df_m["layer"].tolist(), y=df_m[col].tolist(),
                name=pos, mode="lines+markers",
                line=dict(color=COLORS.get(pos, "gray"), width=2),
                marker=dict(size=3),
            ))
        fig.update_layout(
            title=f"Balanced accuracy — {METHOD_LABELS[method]}",
            xaxis_title="Layer", yaxis_title="Balanced accuracy",
            template="plotly_white", width=900, height=400,
        )
        save_fig(fig, str(fig_dir / f"layer_acc_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUROC per layer (×3 methods) — recompute from directions + activations
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc(activations, y, directions, positions, n_layers, fig_dir, fmts):
    print("  AUROC per layer...")
    for method in DIRECTION_METHODS:
        fig = go.Figure()
        for pos in positions:
            aurocs = []
            for layer in range(n_layers):
                X_l = StandardScaler().fit_transform(activations[pos][:, layer, :])
                d = directions[method][pos][layer]
                proj = X_l @ d
                from sklearn.metrics import roc_auc_score
                try:
                    aurocs.append(roc_auc_score(y, proj))
                except ValueError:
                    aurocs.append(0.5)
            fig.add_trace(go.Scatter(
                x=list(range(n_layers)), y=aurocs,
                name=pos, mode="lines+markers",
                line=dict(color=COLORS.get(pos, "gray"), width=2),
                marker=dict(size=3),
            ))
        fig.update_layout(
            title=f"AUROC — {METHOD_LABELS[method]}",
            xaxis_title="Layer", yaxis_title="AUROC",
            template="plotly_white", width=900, height=400,
        )
        save_fig(fig, str(fig_dir / f"auroc_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Projection gap per layer (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_projection_gap(dir_analysis, positions, n_layers, fig_dir, fmts):
    print("  Projection gap...")
    for method in DIRECTION_METHODS:
        fig = go.Figure()
        for pos in positions:
            key = f"projection_gap_{pos}"
            if key in dir_analysis.get(method, {}):
                gaps = dir_analysis[method][key]
                fig.add_trace(go.Scatter(
                    x=list(range(len(gaps))), y=gaps,
                    name=pos, mode="lines+markers",
                    line=dict(color=COLORS.get(pos, "gray"), width=2),
                    marker=dict(size=3),
                ))
        fig.update_layout(
            title=f"Projection gap — {METHOD_LABELS[method]}",
            xaxis_title="Layer", yaxis_title="Gap (system − user)",
            template="plotly_white", width=900, height=400,
        )
        save_fig(fig, str(fig_dir / f"projection_gap_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cosine similarity per layer (×positions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cosine_similarity(dir_analysis, positions, n_layers, fig_dir, fmts):
    print("  Cosine similarity...")
    pairs = [("dim", "iid_mm"), ("dim", "lr"), ("iid_mm", "lr")]
    pair_labels = {"dim_vs_iid_mm": "DiM↔IID-MM", "dim_vs_lr": "DiM↔LR", "iid_mm_vs_lr": "IID-MM↔LR"}

    for pos in positions:
        fig = go.Figure()
        key = f"cross_method_cos_{pos}"
        if key not in dir_analysis:
            continue
        cos_data = dir_analysis[key]
        for pair_key, values in cos_data.items():
            label = pair_labels.get(pair_key, pair_key)
            fig.add_trace(go.Scatter(
                x=list(range(len(values))), y=values,
                name=label, mode="lines+markers", marker=dict(size=3),
            ))
        fig.update_layout(
            title=f"Cross-method cosine similarity — {pos}",
            xaxis_title="Layer", yaxis_title="|cos sim|",
            template="plotly_white", width=900, height=400,
        )
        save_fig(fig, str(fig_dir / f"cosine_sim_{pos}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-position transfer heatmap (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_position_transfer(dir_analysis, fig_dir, fmts):
    print("  Cross-position transfer heatmap...")
    for method in DIRECTION_METHODS:
        xfer = dir_analysis.get(method, {}).get("cross_position_transfer")
        if not xfer:
            continue
        df_xfer = pd.DataFrame(xfer)
        fig = go.Figure(data=go.Heatmap(
            z=df_xfer.values.astype(float),
            x=list(df_xfer.columns), y=list(df_xfer.index),
            text=df_xfer.values.astype(float).round(3).astype(str),
            texttemplate="%{text}", colorscale="RdYlGn", zmin=0.4, zmax=1.0,
        ))
        fig.update_layout(
            title=f"Cross-position transfer — {METHOD_LABELS[method]}",
            width=600, height=500, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"xfer_position_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-category transfer heatmap (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_category_transfer(dir_analysis, fig_dir, fmts):
    print("  Cross-category transfer heatmap...")
    for method in DIRECTION_METHODS:
        xfer = dir_analysis.get(method, {}).get("cross_category_transfer")
        if not xfer:
            continue
        df_xfer = pd.DataFrame(xfer)
        fig = go.Figure(data=go.Heatmap(
            z=df_xfer.values.astype(float),
            x=list(df_xfer.columns), y=list(df_xfer.index),
            text=df_xfer.values.astype(float).round(3).astype(str),
            texttemplate="%{text}", colorscale="RdYlGn", zmin=0.4, zmax=1.0,
        ))
        fig.update_layout(
            title=f"Cross-category transfer — {METHOD_LABELS[method]}",
            width=600, height=500, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"xfer_category_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Confusion matrix at peak layer (×3 methods × positions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(activations, y, directions, peak_layers, positions, fig_dir, fmts):
    print("  Confusion matrices...")
    for method in DIRECTION_METHODS:
        fig = make_subplots(
            rows=1, cols=len(positions),
            subplot_titles=[f"{pos}" for pos in positions],
            horizontal_spacing=0.08,
        )
        for idx, pos in enumerate(positions):
            pk = peak_layers[method][pos]
            d = directions[method][pos][pk]
            X_l = StandardScaler().fit_transform(activations[pos][:, pk, :])
            proj = X_l @ d
            threshold = (proj[y == 1].mean() + proj[y == 0].mean()) / 2
            preds = (proj > threshold).astype(int)
            cm = confusion_matrix(y, preds)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

            fig.add_trace(go.Heatmap(
                z=cm_norm, x=["user", "system"], y=["user", "system"],
                text=cm.astype(str), texttemplate="%{text}",
                colorscale="Blues", showscale=False, zmin=0, zmax=1,
            ), row=1, col=idx + 1)

        fig.update_layout(
            title=f"Confusion matrix (peak layer) — {METHOD_LABELS[method]}",
            height=350, width=250 * len(positions), template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"confusion_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PCA scatter at peak layer (×3 methods × positions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_scatter(activations, y, peak_layers, positions, fig_dir, fmts):
    print("  PCA scatter at peak layer...")
    for method in DIRECTION_METHODS:
        n_pos = len(positions)
        cols = min(n_pos, 4)
        rows = (n_pos + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{pos} (L{peak_layers[method][pos]})" for pos in positions],
            horizontal_spacing=0.08, vertical_spacing=0.15,
        )
        for idx, pos in enumerate(positions):
            r, c = idx // cols + 1, idx % cols + 1
            pk = peak_layers[method][pos]
            X_l = StandardScaler().fit_transform(activations[pos][:, pk, :])
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_l)

            for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
                mask = y == lv
                fig.add_trace(go.Scattergl(
                    x=X_pca[mask, 0], y=X_pca[mask, 1],
                    mode="markers", name=ln if idx == 0 else None,
                    marker=dict(color=clr, size=2, opacity=0.15),
                    showlegend=(idx == 0),
                ), row=r, col=c)

        fig.update_layout(
            title=f"PCA at peak layer — {METHOD_LABELS[method]}",
            height=350 * rows, width=900, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"pca_scatter_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Projection distribution at peak layer (×3 methods × positions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_projection_distribution(activations, y, directions, peak_layers, positions, fig_dir, fmts):
    print("  Projection distributions...")
    for method in DIRECTION_METHODS:
        n_pos = len(positions)
        cols = min(n_pos, 4)
        rows = (n_pos + cols - 1) // cols
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{pos} (L{peak_layers[method][pos]})" for pos in positions],
            horizontal_spacing=0.08, vertical_spacing=0.15,
        )
        for idx, pos in enumerate(positions):
            r, c = idx // cols + 1, idx % cols + 1
            pk = peak_layers[method][pos]
            d = directions[method][pos][pk]
            X_l = StandardScaler().fit_transform(activations[pos][:, pk, :])
            proj = X_l @ d

            for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
                mask = y == lv
                fig.add_trace(go.Histogram(
                    x=proj[mask], name=ln if idx == 0 else None,
                    marker_color=clr, opacity=0.5, nbinsx=80,
                    showlegend=(idx == 0),
                ), row=r, col=c)

        fig.update_layout(
            title=f"Projection distribution — {METHOD_LABELS[method]}",
            barmode="overlay",
            height=350 * rows, width=900, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"proj_dist_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 10. IID-MM direction vs Residual PC1 scatter (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_pca_scatter(activations, y, directions, peak_layers, positions, fig_dir, fmts):
    print("  Residual PCA scatter (direction vs residual PC1)...")
    for method in DIRECTION_METHODS:
        best_pos = max(positions, key=lambda p: peak_layers[method][p])
        # Use the position with highest accuracy from metrics if available
        # Fallback: just use first position
        for pos in positions:
            pk = peak_layers[method][pos]
            d = directions[method][pos][pk]

            X_t = torch.as_tensor(activations[pos][:, pk, :], device="cpu")
            mu = X_t.mean(0)
            std = X_t.std(0).clamp(min=1e-12)
            X_scaled = ((X_t - mu) / std).numpy()

            proj_d = X_scaled @ d
            residual = X_scaled - np.outer(proj_d, d)
            pca = PCA(n_components=1, random_state=42)
            pc1 = pca.fit_transform(residual)[:, 0]

            fig = go.Figure()
            for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
                mask = y == lv
                fig.add_trace(go.Scattergl(
                    x=proj_d[mask], y=pc1[mask], mode="markers",
                    marker=dict(color=clr, size=3, opacity=0.15), name=ln,
                ))
            fig.update_layout(
                title=f"{METHOD_LABELS[method]} direction vs Residual PC1 — {pos} @ L{pk}",
                xaxis_title=f"{METHOD_LABELS[method]} projection",
                yaxis_title="Residual PC1",
                height=500, width=700, template="plotly_white",
            )
            save_fig(fig, str(fig_dir / f"residual_scatter_{method}_{pos}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 11. PC1 histogram by label (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pc_histogram(activations, y, df, directions, peak_layers, positions, fig_dir, fmts):
    print("  PC1 histogram by label & constraint type...")
    for method in DIRECTION_METHODS:
        for pos in positions:
            pk = peak_layers[method][pos]
            X_t = torch.as_tensor(activations[pos][:, pk, :], device="cpu")
            mu = X_t.mean(0)
            std = X_t.std(0).clamp(min=1e-12)
            X_scaled = ((X_t - mu) / std).numpy()

            pca = PCA(n_components=5, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["PC1 by label", "PC1 by constraint_type (top 5)"],
            )

            for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
                fig.add_trace(go.Histogram(
                    x=X_pca[y == lv, 0], name=ln, marker_color=clr,
                    opacity=0.6, nbinsx=50,
                ), row=1, col=1)

            # Top 5 constraint types by mean PC1
            ct_means = df.groupby("constraint_type").apply(
                lambda g: X_pca[g.index, 0].mean()
            )
            top_cts = ct_means.nlargest(3).index.tolist() + ct_means.nsmallest(2).index.tolist()
            for ct in top_cts:
                mask = df["constraint_type"] == ct
                fig.add_trace(go.Histogram(
                    x=X_pca[mask, 0], name=ct[:20], opacity=0.5, nbinsx=30,
                ), row=1, col=2)

            fig.update_layout(
                title=f"PCA components — {METHOD_LABELS[method]} — {pos} @ L{pk}",
                barmode="overlay", height=400, width=1000, template="plotly_white",
            )
            save_fig(fig, str(fig_dir / f"pc_histogram_{method}_{pos}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Neuron importance heatmap (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_neuron_importance(directions, positions, n_layers, d_model, peak_layers, fig_dir, fmts):
    print("  Neuron importance heatmap...")
    N_TOP = 50
    for method in DIRECTION_METHODS:
        # Use best position
        best_pos = max(positions, key=lambda p: peak_layers[method][p])

        all_weights = np.zeros((n_layers, d_model))
        for layer in range(n_layers):
            d = directions[method][best_pos][layer]
            all_weights[layer] = np.abs(d)

        # Top N neurons by max weight across layers
        max_per_neuron = all_weights.max(axis=0)
        top_neurons = np.argsort(max_per_neuron)[::-1][:N_TOP]

        fig = go.Figure(data=go.Heatmap(
            z=all_weights[:, top_neurons].T,
            x=list(range(n_layers)),
            y=[str(n) for n in top_neurons],
            colorscale="Hot", reversescale=True,
        ))
        fig.update_layout(
            title=f"Top {N_TOP} neuron importance — {METHOD_LABELS[method]} — {best_pos}",
            xaxis_title="Layer", yaxis_title="Neuron index",
            height=800, width=900, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"neuron_heatmap_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Probe accuracy by SCR group (×3 methods)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scr_group_accuracy(diagnostics, fig_dir, fmts):
    print("  Probe accuracy by SCR group...")
    fig = go.Figure()
    groups = ["high", "mid", "low"]
    x_labels = ["High (≥0.3)", "Mid (0.1-0.3)", "Low (<0.1)"]

    for method in DIRECTION_METHODS:
        scr_data = diagnostics.get(method, {}).get("accuracy_by_scr_group", {})
        accs = []
        for g in groups:
            acc = scr_data.get(g, {}).get("acc")
            accs.append(acc if acc is not None else 0)
        fig.add_trace(go.Bar(
            x=x_labels, y=accs, name=METHOD_LABELS[method],
            marker_color=METHOD_COLORS[method],
        ))

    fig.update_layout(
        title="Probe accuracy by SCR group",
        xaxis_title="SCR group", yaxis_title="Balanced accuracy",
        barmode="group", height=400, width=700, template="plotly_white",
    )
    save_fig(fig, str(fig_dir / "scr_group_accuracy"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Condition PCA scatter (colored by SCR)
# ─────────────────────────────────────────────────────────────────────────────

def plot_condition_pca(activations, y, df, peak_layers, positions, n_layers, fig_dir, fmts):
    print("  Condition PCA scatter...")
    condition_cols = ["constraint_type"]
    if "system_style" in df.columns:
        condition_cols.append("system_style")
    if "user_style" in df.columns:
        condition_cols.append("user_style")

    # Use one representative peak layer
    for method in DIRECTION_METHODS:
        best_pos = max(positions, key=lambda p: peak_layers[method][p])
        pk = peak_layers[method][best_pos]

        X_peak = activations[best_pos][:, pk, :]
        X_scaled = StandardScaler().fit_transform(X_peak)

        groups = df.groupby(condition_cols)
        cond_means, cond_scrs, cond_labels = [], [], []
        for name, grp in groups:
            indices = grp.index.values
            if len(indices) < 5:
                continue
            cond_means.append(X_scaled[indices].mean(axis=0))
            cond_scrs.append(y[indices].mean())
            cond_labels.append(name)

        X_cond = np.array(cond_means)
        cond_scr = np.array(cond_scrs)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_cond)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0], y=X_pca[:, 1], mode="markers",
            marker=dict(
                color=cond_scr, colorscale="RdBu", cmin=0, cmax=1,
                size=8, colorbar=dict(title="SCR"), line=dict(width=0.5, color="gray"),
            ),
            text=[str(l) for l in cond_labels],
            hovertemplate="%{text}<br>SCR=%{marker.color:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Condition PCA — {METHOD_LABELS[method]} — {best_pos} @ L{pk}",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
            height=600, width=800, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"condition_pca_{method}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Conceptor analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_conceptor_analysis(conceptor_data, positions, n_layers, fig_dir, fmts):
    print("  Conceptor analysis plots...")
    if not conceptor_data:
        print("    No conceptor data found, skipping.")
        return

    for pos in positions:
        pos_data = conceptor_data.get(pos, {})
        if not pos_data:
            continue

        # Effective dimensionality per layer
        layers = []
        eff_sys_90, eff_usr_90 = [], []
        overlaps = []
        for layer in range(n_layers):
            lk = f"L{layer}"
            if lk not in pos_data:
                continue
            layers.append(layer)
            eff_sys_90.append(pos_data[lk]["system_eff_dim_90"])
            eff_usr_90.append(pos_data[lk]["user_eff_dim_90"])
            overlaps.append(pos_data[lk]["subspace_overlap"])

        if not layers:
            continue

        # Effective dim plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=layers, y=eff_sys_90, name="System (90%)",
                                  mode="lines+markers", marker=dict(size=3),
                                  line=dict(color="#d62728")))
        fig.add_trace(go.Scatter(x=layers, y=eff_usr_90, name="User (90%)",
                                  mode="lines+markers", marker=dict(size=3),
                                  line=dict(color="#1f77b4")))
        fig.update_layout(
            title=f"Conceptor effective dimensionality — {pos}",
            xaxis_title="Layer", yaxis_title="Effective dim (90% energy)",
            height=400, width=900, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"conceptor_effdim_{pos}"), fmts)

        # Subspace overlap plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=layers, y=overlaps, name="Overlap",
                                  mode="lines+markers", marker=dict(size=3),
                                  line=dict(color="#9467bd")))
        fig.update_layout(
            title=f"Conceptor subspace overlap (system vs user) — {pos}",
            xaxis_title="Layer", yaxis_title="Frobenius overlap",
            height=400, width=900, template="plotly_white",
        )
        save_fig(fig, str(fig_dir / f"conceptor_overlap_{pos}"), fmts)

        # Alignment with probe directions
        for method in DIRECTION_METHODS:
            align_key_sys = f"align_sys_{method}"
            align_key_usr = f"align_usr_{method}"
            if align_key_sys not in pos_data.get(f"L0", {}):
                continue
            align_sys = [pos_data[f"L{l}"][align_key_sys] for l in layers]
            align_usr = [pos_data[f"L{l}"][align_key_usr] for l in layers]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=layers, y=align_sys, name="C_system top eigvec",
                                      mode="lines+markers", marker=dict(size=3),
                                      line=dict(color="#d62728")))
            fig.add_trace(go.Scatter(x=layers, y=align_usr, name="C_user top eigvec",
                                      mode="lines+markers", marker=dict(size=3),
                                      line=dict(color="#1f77b4")))
            fig.update_layout(
                title=f"Conceptor↔{METHOD_LABELS[method]} alignment — {pos}",
                xaxis_title="Layer",
                yaxis_title=f"|cos sim| with {METHOD_LABELS[method]} direction",
                height=400, width=900, template="plotly_white",
            )
            save_fig(fig, str(fig_dir / f"conceptor_align_{method}_{pos}"), fmts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all plots from saved probing results.")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--act-dir", type=str, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    act_dir = Path(args.act_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir = results_dir / "metrics"
    fmts = args.formats

    print("── Loading data ──")
    activations, y, df, positions, n_layers, d_model = load_activations(act_dir)
    directions = load_directions(results_dir, positions, n_layers)

    # Load metrics
    metrics_df = pd.read_csv(met_dir / "probe_metrics.csv")
    with open(met_dir / "peak_layers.json") as f:
        peak_layers = json.load(f)

    dir_analysis = {}
    if (met_dir / "direction_analysis.json").exists():
        with open(met_dir / "direction_analysis.json") as f:
            dir_analysis = json.load(f)

    diagnostics = {}
    if (met_dir / "diagnostics.json").exists():
        with open(met_dir / "diagnostics.json") as f:
            diagnostics = json.load(f)

    conceptor_data = {}
    if (met_dir / "conceptor_analysis.json").exists():
        with open(met_dir / "conceptor_analysis.json") as f:
            conceptor_data = json.load(f)

    # ── Generate all plots ──
    print("\n── Generating plots ──")

    plot_layer_accuracy(metrics_df, positions, n_layers, fig_dir, fmts)
    plot_auroc(activations, y, directions, positions, n_layers, fig_dir, fmts)
    plot_projection_gap(dir_analysis, positions, n_layers, fig_dir, fmts)
    plot_cosine_similarity(dir_analysis, positions, n_layers, fig_dir, fmts)
    plot_cross_position_transfer(dir_analysis, fig_dir, fmts)
    plot_cross_category_transfer(dir_analysis, fig_dir, fmts)
    plot_confusion_matrices(activations, y, directions, peak_layers, positions, fig_dir, fmts)
    plot_pca_scatter(activations, y, peak_layers, positions, fig_dir, fmts)
    plot_projection_distribution(activations, y, directions, peak_layers, positions, fig_dir, fmts)
    plot_residual_pca_scatter(activations, y, directions, peak_layers, positions, fig_dir, fmts)
    plot_pc_histogram(activations, y, df, directions, peak_layers, positions, fig_dir, fmts)
    plot_neuron_importance(directions, positions, n_layers, d_model, peak_layers, fig_dir, fmts)
    plot_scr_group_accuracy(diagnostics, fig_dir, fmts)
    plot_condition_pca(activations, y, df, peak_layers, positions, n_layers, fig_dir, fmts)
    plot_conceptor_analysis(conceptor_data, positions, n_layers, fig_dir, fmts)

    n_files = len(list(fig_dir.glob("*")))
    print(f"\n── Done: {n_files} files in {fig_dir}/ ──")


if __name__ == "__main__":
    main()