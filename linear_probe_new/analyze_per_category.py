#!/usr/bin/env python3
"""
analyze_per_category.py — Per-category scatter on IID-MM direction.

Answers: does every constraint category show user/system separation along
the IID-MM direction, or is the direction just encoding constraint type?

Reads saved activations, directions, and metadata. No model loading needed.

Usage:
    python -u analyze_per_category.py \
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
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Constraint category mapping (same as notebook)
# ─────────────────────────────────────────────────────────────────────────────

def assign_constraint_category(ct):
    ct_lower = ct.lower()
    if any(kw in ct_lower for kw in ["json", "bullet", "numbered", "section", "prose",
                                       "template", "capitalization", "format", "html", "markdown"]):
        return "format"
    elif any(kw in ct_lower for kw in ["language", "en_es", "en_zh", "spanish", "english", "chinese"]):
        return "language"
    elif any(kw in ct_lower for kw in ["formal", "casual", "tone", "person", "first",
                                         "third", "active", "passive", "voice"]):
        return "style"
    elif any(kw in ct_lower for kw in ["emoji", "starting_word", "self_refer",
                                         "direct_answer", "hedging", "keyword", "forbidden",
                                         "alliteration", "alphabetical", "paragraph", "address",
                                         "disclaimer", "lowercase", "questions_vs"]):
        return "content"
    else:
        return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
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
    X_full = np.load(act_path)
    X_peak = X_full[:, peak_layer, :]
    del X_full
    gc.collect()

    # Labels
    y_files = sorted(act_dir.glob("y_*.npy"))
    labels_file = act_dir / "labels.npz"
    if labels_file.exists():
        y = np.load(labels_file)["y"]
    else:
        y = np.load(y_files[0])

    # Metadata
    meta_files = sorted(act_dir.glob("meta_*.csv"))
    if (act_dir / "metadata.csv").exists():
        df = pd.read_csv(act_dir / "metadata.csv")
    else:
        df = pd.read_csv(meta_files[0])

    return X_peak, y, df, d, peak_layer


def save_fig(fig, path_stem, formats=("png", "pdf")):
    for fmt in formats:
        fig.write_image(f"{path_stem}.{fmt}", scale=2)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
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
    n_samples = len(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_peak)

    # Project onto d
    proj_d = X_scaled @ d

    # Residual PC1
    residual = X_scaled - np.outer(proj_d, d)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(residual)[:, 0]

    # Assign categories
    df["category"] = df["constraint_type"].apply(assign_constraint_category)
    categories = sorted(df["category"].unique())
    print(f"  Categories: {categories}")
    for cat in categories:
        n = (df["category"] == cat).sum()
        scr = y[df["category"] == cat].mean()
        print(f"    {cat}: {n} samples, SCR={scr:.3f}")

    # ── Plot 1: Per-category scatter, colored by label ──
    print("\nPlot 1: Per-category scatter (label-colored)...")
    n_cats = len(categories)
    fig = make_subplots(
        rows=1, cols=n_cats,
        subplot_titles=[f"{cat}" for cat in categories],
        horizontal_spacing=0.04,
    )

    for idx, cat in enumerate(categories):
        cat_mask = df["category"] == cat
        for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
            mask = cat_mask & (y == lv)
            fig.add_trace(go.Scattergl(
                x=proj_d[mask], y=pc1[mask],
                mode="markers", name=ln if idx == 0 else None,
                marker=dict(color=clr, size=2, opacity=0.15),
                showlegend=(idx == 0),
            ), row=1, col=idx + 1)

    fig.update_layout(
        title=f"Per-category separation — IID-MM @ L{peak_layer} ({args.position})",
        height=500, width=300 * n_cats, template="plotly_white",
    )
    for i in range(n_cats):
        fig.update_xaxes(title_text="IID-MM proj.", row=1, col=i + 1)
    fig.update_yaxes(title_text="Residual PC1", row=1, col=1)
    save_fig(fig, str(out_dir / "per_category_scatter"), fmts)
    print(f"  Saved per_category_scatter.png/pdf")

    # ── Plot 2: Per-category X-axis (IID-MM) distribution ──
    print("Plot 2: Per-category IID-MM projection distribution...")
    fig = go.Figure()
    for cat in categories:
        cat_mask = df["category"] == cat
        for lv, ln, clr in [(0, "user", "#1f77b4"), (1, "system", "#d62728")]:
            mask = cat_mask & (y == lv)
            fig.add_trace(go.Violin(
                x=[f"{cat}" ] * mask.sum(),
                y=proj_d[mask],
                side="positive" if lv == 1 else "negative",
                name=f"{ln}" if cat == categories[0] else None,
                showlegend=(cat == categories[0]),
                line_color=clr,
                meanline_visible=True,
                scalemode="width",
                width=0.8,
                opacity=0.7,
            ))

    fig.update_layout(
        title=f"IID-MM projection by category — L{peak_layer} ({args.position})",
        yaxis_title="IID-MM projection",
        xaxis_title="Constraint category",
        height=500, width=800, template="plotly_white",
        violingap=0, violinmode="overlay",
    )
    save_fig(fig, str(out_dir / "per_category_violin"), fmts)
    print(f"  Saved per_category_violin.png/pdf")

    # ── Plot 3: Per constraint_type (not category) balanced accuracy ──
    print("Plot 3: Per constraint_type separation quality...")
    ct_stats = []
    for ct in df["constraint_type"].unique():
        ct_mask = df["constraint_type"] == ct
        n_ct = ct_mask.sum()
        if n_ct < 10:
            continue
        y_ct = y[ct_mask]
        proj_ct = proj_d[ct_mask]
        # Balanced accuracy using median threshold
        threshold = np.median(proj_ct)
        preds = (proj_ct > threshold).astype(int)
        from sklearn.metrics import balanced_accuracy_score
        ba = balanced_accuracy_score(y_ct, preds)
        scr = y_ct.mean()
        cat = assign_constraint_category(ct)
        ct_stats.append({
            "constraint_type": ct,
            "category": cat,
            "n": n_ct,
            "scr": scr,
            "bal_acc": ba,
            "mean_sys": proj_ct[y_ct == 1].mean() if (y_ct == 1).any() else np.nan,
            "mean_usr": proj_ct[y_ct == 0].mean() if (y_ct == 0).any() else np.nan,
        })

    ct_df = pd.DataFrame(ct_stats).sort_values("bal_acc", ascending=False)
    ct_df.to_csv(out_dir / "per_constraint_type_stats.csv", index=False)

    # Color by category
    cat_colors = {"format": "#1f77b4", "language": "#ff7f0e", "style": "#2ca02c",
                  "content": "#d62728", "other": "#9467bd"}

    fig = go.Figure()
    for cat in categories:
        ct_sub = ct_df[ct_df["category"] == cat]
        fig.add_trace(go.Bar(
            x=ct_sub["constraint_type"],
            y=ct_sub["bal_acc"],
            name=cat,
            marker_color=cat_colors.get(cat, "gray"),
            text=ct_sub["bal_acc"].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
        ))

    fig.update_layout(
        title=f"Per-constraint balanced accuracy (IID-MM direction) — L{peak_layer}",
        yaxis_title="Balanced accuracy",
        xaxis_title="Constraint type",
        height=500, width=max(800, len(ct_df) * 40),
        template="plotly_white",
        xaxis_tickangle=-45,
        yaxis_range=[0.3, 1.05],
        barmode="group",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="chance", annotation_position="top left")
    save_fig(fig, str(out_dir / "per_constraint_type_accuracy"), fmts)
    print(f"  Saved per_constraint_type_accuracy.png/pdf")

    # ── Plot 4: Category-colored scatter (single plot) ──
    print("Plot 4: Category-colored scatter...")
    fig = go.Figure()
    for cat in categories:
        cat_mask = df["category"] == cat
        for lv, ln, marker in [(0, "user", "circle"), (1, "system", "diamond")]:
            mask = cat_mask & (y == lv)
            fig.add_trace(go.Scattergl(
                x=proj_d[mask], y=pc1[mask],
                mode="markers",
                name=f"{cat} ({ln})",
                marker=dict(color=cat_colors.get(cat, "gray"), size=3,
                            opacity=0.12, symbol=marker),
            ))

    fig.update_layout(
        title=f"IID-MM vs Residual PC1 — colored by category — L{peak_layer}",
        xaxis_title="IID-MM projection",
        yaxis_title="Residual PC1",
        height=600, width=800, template="plotly_white",
    )
    save_fig(fig, str(out_dir / "category_colored_scatter"), fmts)
    print(f"  Saved category_colored_scatter.png/pdf")

    # ── Summary statistics ──
    print("\n── Summary ──")
    print(f"  Overall bal_acc (IID-MM threshold): "
          f"{balanced_accuracy_score(y, (proj_d > np.median(proj_d)).astype(int)):.3f}")
    print(f"\n  Per-category bal_acc:")
    for cat in categories:
        cat_mask = df["category"] == cat
        proj_cat = proj_d[cat_mask]
        y_cat = y[cat_mask]
        if len(y_cat) > 0 and y_cat.sum() > 0 and (y_cat == 0).sum() > 0:
            ba = balanced_accuracy_score(y_cat, (proj_cat > np.median(proj_cat)).astype(int))
            print(f"    {cat}: {ba:.3f} (n={len(y_cat)}, SCR={y_cat.mean():.3f})")

    print(f"\n  Per-category mean projection:")
    print(f"  {'Category':<12} {'sys mean':>10} {'usr mean':>10} {'gap':>10}")
    for cat in categories:
        cat_mask = df["category"] == cat
        sys_mean = proj_d[cat_mask & (y == 1)].mean()
        usr_mean = proj_d[cat_mask & (y == 0)].mean()
        print(f"  {cat:<12} {sys_mean:>10.4f} {usr_mean:>10.4f} {sys_mean - usr_mean:>10.4f}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()