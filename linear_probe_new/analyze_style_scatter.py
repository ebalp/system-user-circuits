#!/usr/bin/env python3
"""
analyze_style_scatter.py — Residual scatter colored by system_style and user_style.

Same projection space as residual_scatter (IID-MM direction vs Residual PC1),
but colored by metadata fields instead of label. Helps identify what the
residual PC1 axis captures.

Reads saved activations, directions, and metadata. No model loading needed.

Usage:
    python -u analyze_style_scatter.py \
        --act-dir /workspace/activations/meta-llama_Llama-3.1-8B-Instruct \
        --probe-dir ./results/llama-8b \
        --out-dir ./results/llama-8b/figures
"""

import argparse
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as analyze_per_category.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(act_dir, probe_dir, method="iid_mm", position="last_prompt"):
    act_dir = Path(act_dir)
    probe_dir = Path(probe_dir)

    # Peak layer
    with open(probe_dir / "metrics" / "peak_layers.json") as f:
        peak_layers = json.load(f)
    peak_layer = peak_layers[method][position]

    # Direction
    d = np.load(probe_dir / "artifacts" / "directions" / f"{method}_{position}_L{peak_layer}.npy")
    d = d / (np.linalg.norm(d) + 1e-12)

    # Activations
    act_path = None
    for f in act_dir.glob("activations_*.npy"):
        if f.stem.endswith(f"_{position}"):
            act_path = f
            break
    if act_path is None:
        raise FileNotFoundError(f"No activation file for position '{position}' in {act_dir}")
    X_full = np.load(act_path)
    X_peak = X_full[:, peak_layer, :]
    del X_full
    gc.collect()

    # Labels
    labels_file = act_dir / "labels.npz"
    if labels_file.exists():
        y = np.load(labels_file)["y"]
    else:
        y_files = sorted(act_dir.glob("y_*.npy"))
        y = np.load(y_files[0])

    # Metadata
    if (act_dir / "metadata.csv").exists():
        df = pd.read_csv(act_dir / "metadata.csv")
    else:
        meta_files = sorted(act_dir.glob("meta_*.csv"))
        df = pd.read_csv(meta_files[0])

    return X_peak, y, df, d, peak_layer


def save_fig(fig, path_stem, formats=("png", "pdf")):
    for fmt in formats:
        fig.write_image(f"{path_stem}.{fmt}", scale=2)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_colored_scatter(proj_d, pc1, color_values, color_col_name, peak_layer, position, out_dir, fmts):
    """
    Single scatter plot of IID-MM projection vs Residual PC1,
    colored by a categorical metadata field.
    """
    unique_vals = sorted(color_values.unique())
    # Use a qualitative color palette
    palette = px.colors.qualitative.Set2 if len(unique_vals) <= 8 else px.colors.qualitative.Alphabet
    color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}

    fig = go.Figure()
    for val in unique_vals:
        mask = color_values == val
        fig.add_trace(go.Scattergl(
            x=proj_d[mask],
            y=pc1[mask],
            mode="markers",
            name=str(val),
            marker=dict(
                color=color_map[val],
                size=3,
                opacity=0.15,
            ),
        ))

    fig.update_layout(
        title=f"IID-MM vs Residual PC1 — colored by {color_col_name} — L{peak_layer} ({position})",
        xaxis_title="IID-MM projection",
        yaxis_title="Residual PC1",
        height=600,
        width=800,
        template="plotly_white",
        legend=dict(title=color_col_name),
    )

    out_name = f"style_colored_scatter_{color_col_name}"
    save_fig(fig, str(Path(out_dir) / out_name), fmts)
    print(f"  Saved {out_name}.png/pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate residual scatter plots colored by system_style and user_style."
    )
    parser.add_argument("--act-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", default="iid_mm")
    parser.add_argument("--position", default="last_prompt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmts = ("png", "pdf")

    print("Loading data...")
    X_peak, y, df, d, peak_layer = load_data(
        args.act_dir, args.probe_dir, args.method, args.position
    )

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_peak)

    # Project onto d
    proj_d = X_scaled @ d

    # Residual PC1
    residual = X_scaled - np.outer(proj_d, d)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(residual)[:, 0]

    print(f"  {len(y)} samples, peak layer {peak_layer}")

    # ── Plot 1: colored by system_style ──
    if "system_style" in df.columns:
        print("\nPlot: colored by system_style...")
        plot_colored_scatter(
            proj_d, pc1, df["system_style"],
            "system_style", peak_layer, args.position, out_dir, fmts,
        )
    else:
        print("  ⚠ system_style not found in metadata, skipping.")

    # ── Plot 2: colored by user_style ──
    if "user_style" in df.columns:
        print("Plot: colored by user_style...")
        plot_colored_scatter(
            proj_d, pc1, df["user_style"],
            "user_style", peak_layer, args.position, out_dir, fmts,
        )
    else:
        print("  ⚠ user_style not found in metadata, skipping.")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()