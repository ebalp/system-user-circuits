"""Visualization functions for linear probing analysis.

All Plotly chart-building code consolidated from the original notebook into
reusable functions.  Each function returns a :class:`plotly.graph_objects.Figure`
and optionally writes it to HTML via *save_path*.

This module is deliberately decoupled from ``data.py`` and ``probe.py`` -- it
accepts only numpy arrays, pandas DataFrames, and plain Python scalars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# ── Constants ────────────────────────────────────────────────────────────────

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ── Helper ───────────────────────────────────────────────────────────────────


def _save_fig(fig: go.Figure, save_path: Path | None) -> None:
    """Write figure to HTML if *save_path* is provided."""
    if save_path is not None:
        fig.write_html(str(save_path))
        print(f"Saved: {save_path}")


# ── Probe Curves ─────────────────────────────────────────────────────────────


def plot_probe_curves(
    probe_results: dict[str, dict[str, pd.DataFrame]],
    token_positions: list[str],
    model_name: str,
    n_folds: int,
    *,
    metric: str = "roc_auc",
    ctrl_linear: float | None = None,
    ctrl_boosted: float | None = None,
    title_suffix: str = "",
    save_path: Path | None = None,
) -> go.Figure:
    """Universal probe curve plotter.

    Replaces both the main probe visualisation (cell 46) and the format-only
    variant (cell 56) from the original notebook.

    Parameters
    ----------
    probe_results
        Maps *label* -> {*pos_name*: DataFrame}.  Each DataFrame must contain
        columns ``layer``, ``{metric}_mean``, ``{metric}_std``.
    token_positions
        Which positions to plot.
    model_name
        Model identifier shown in the title.
    n_folds
        Number of CV folds shown in the subtitle.
    metric
        Which column to use (``"roc_auc"`` or ``"balanced_accuracy"``).
    ctrl_linear, ctrl_boosted
        Optional horizontal baseline lines.
    title_suffix
        Appended to the chart title.
    save_path
        If given, figure is saved as HTML.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Dash styles cycle for different backends/labels
    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    labels = list(probe_results.keys())

    fig = go.Figure()

    for i, pos_name in enumerate(token_positions):
        color = COLORS[i % len(COLORS)]

        for bi, label in enumerate(labels):
            pos_dict = probe_results[label]
            if pos_name not in pos_dict:
                continue
            df_res = pos_dict[pos_name]
            layers = df_res["layer"].values
            means = df_res[mean_col].values
            stds = df_res[std_col].values
            dash = dash_styles[bi % len(dash_styles)]

            show_legend = bi == 0
            legend_name = pos_name if len(labels) == 1 else f"{pos_name} ({label})"
            legend_group = pos_name

            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=means,
                    mode="lines",
                    name=legend_name,
                    legendgroup=legend_group,
                    showlegend=show_legend,
                    line=dict(color=color, width=2, dash=dash),
                )
            )

            # CI band for first backend only
            if bi == 0:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([layers, layers[::-1]]),
                        y=np.concatenate(
                            [means + stds, (means - stds)[::-1]]
                        ),
                        fill="toself",
                        fillcolor=color,
                        opacity=0.10,
                        line=dict(color="rgba(0,0,0,0)"),
                        legendgroup=legend_group,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Horizontal baselines
    if ctrl_linear is not None:
        fig.add_hline(
            y=ctrl_linear,
            line_dash="dashdot",
            line_color="black",
            line_width=1.5,
            annotation_text=f"linear ctrl ({ctrl_linear:.2f})",
            annotation_position="top left",
        )

    if ctrl_boosted is not None:
        fig.add_hline(
            y=ctrl_boosted,
            line_dash="dash",
            line_color="darkred",
            line_width=1.5,
            annotation_text=f"boosted ctrl ({ctrl_boosted:.2f})",
            annotation_position="top left",
        )

    # Chance line
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="chance (0.50)",
        annotation_position="bottom right",
    )

    backend_desc = ", ".join(labels)
    metric_title = metric.replace("_", " ").title()
    title = (
        f"Linear Probe{title_suffix}<br>"
        f"<sub>{model_name} | backends: {backend_desc} | "
        f"{n_folds}-fold CV | metric={metric}</sub>"
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=metric_title,
        yaxis=dict(range=[0.4, 1.02]),
        legend_title="Token Position",
        width=1100,
        height=580,
        template="plotly_white",
        hovermode="x unified",
    )

    _save_fig(fig, save_path)
    return fig


# ── Probe Comparison ─────────────────────────────────────────────────────────


def plot_probe_comparison(
    results_stratified: pd.DataFrame,
    results_grouped: pd.DataFrame,
    pos_name: str,
    model_name: str,
    *,
    metric: str = "roc_auc",
    save_path: Path | None = None,
) -> go.Figure:
    """Side-by-side stratified vs grouped CV curves for one position.

    Parameters
    ----------
    results_stratified, results_grouped
        DataFrames with columns ``layer``, ``{metric}_mean``, ``{metric}_std``.
    pos_name
        Token position name shown in the title.
    model_name
        Model identifier shown in the title.
    metric
        Which metric column to use.
    save_path
        Optional HTML output path.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    metric_title = metric.replace("_", " ").title()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Stratified CV", "Grouped CV"],
        horizontal_spacing=0.10,
    )

    for col, (df_res, label) in enumerate(
        [
            (results_stratified, "Stratified"),
            (results_grouped, "Grouped"),
        ],
        start=1,
    ):
        layers = df_res["layer"].values
        means = df_res[mean_col].values
        stds = df_res[std_col].values

        fig.add_trace(
            go.Scatter(
                x=layers,
                y=means,
                mode="lines+markers",
                name=label,
                line=dict(color=COLORS[col - 1], width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=col,
        )

        # CI band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([layers, layers[::-1]]),
                y=np.concatenate(
                    [means + stds, (means - stds)[::-1]]
                ),
                fill="toself",
                fillcolor=COLORS[col - 1],
                opacity=0.10,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        # Peak annotation
        peak_idx = int(np.argmax(means))
        peak_val = means[peak_idx]
        fig.add_annotation(
            x=layers[peak_idx],
            y=peak_val,
            text=f"peak L{layers[peak_idx]}: {peak_val:.3f}",
            showarrow=True,
            arrowhead=2,
            row=1,
            col=col,
        )

        fig.update_xaxes(title_text="Layer", row=1, col=col)
        fig.update_yaxes(title_text=metric_title, row=1, col=col)

    # Match y-axis scale
    all_means = np.concatenate(
        [
            results_stratified[mean_col].values,
            results_grouped[mean_col].values,
        ]
    )
    y_min = max(0.4, float(all_means.min()) - 0.05)
    y_max = min(1.02, float(all_means.max()) + 0.03)
    fig.update_yaxes(range=[y_min, y_max])

    # Gap in peaks
    peak_s = results_stratified[mean_col].max()
    peak_g = results_grouped[mean_col].max()
    gap = peak_s - peak_g

    fig.update_layout(
        title_text=(
            f"Stratified vs Grouped CV — {pos_name}<br>"
            f"<sub>{model_name} | peak gap = {gap:+.4f}</sub>"
        ),
        width=1100,
        height=500,
    )

    _save_fig(fig, save_path)
    return fig


# ── Cosine Similarity Heatmaps ──────────────────────────────────────────────


def plot_cosine_heatmaps(
    weights_scaled: np.ndarray,
    weights_unscaled: np.ndarray,
    pos_name: str,
    *,
    save_path: Path | None = None,
) -> go.Figure:
    """Side-by-side cross-layer cosine similarity heatmaps.

    Parameters
    ----------
    weights_scaled, weights_unscaled
        Unit-norm weight arrays of shape ``(n_layers, d_model)``.
    pos_name
        Token position name shown in the title.
    """
    cos_scaled = cos_sim(weights_scaled)
    cos_unscaled = cos_sim(weights_unscaled)
    n_layers = cos_scaled.shape[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Scaled probe directions", "Unscaled probe directions"],
        horizontal_spacing=0.12,
    )

    for col, (mat, _label) in enumerate(
        [(cos_scaled, "Scaled"), (cos_unscaled, "Unscaled")], start=1
    ):
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=list(range(n_layers)),
                y=list(range(n_layers)),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="cos sim", x=0.45 if col == 1 else 1.0
                ),
                hovertemplate=(
                    "Layer %{x} vs %{y}<br>cos=%{z:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="Layer", row=1, col=col)
        fig.update_yaxes(title_text="Layer", row=1, col=col)

    fig.update_layout(
        title_text=(
            f"Cross-Layer Cosine Similarity of Probe Directions ({pos_name})"
        ),
        width=1100,
        height=500,
    )

    _save_fig(fig, save_path)
    return fig


# ── Cosine Similarity Curves ────────────────────────────────────────────────


def plot_cosine_curves(
    weights_scaled: np.ndarray,
    weights_unscaled: np.ndarray,
    pos_name: str,
    *,
    save_path: Path | None = None,
) -> go.Figure:
    """Consecutive and final-layer cosine similarity curves.

    Parameters
    ----------
    weights_scaled, weights_unscaled
        Unit-norm weight arrays of shape ``(n_layers, d_model)``.
    pos_name
        Token position name shown in the title.
    """
    cos_scaled = cos_sim(weights_scaled)
    cos_unscaled = cos_sim(weights_unscaled)
    n_layers = cos_scaled.shape[0]
    layers = np.arange(n_layers)

    # Consecutive: cos(w_i, w_{i+1})
    consec_scaled = np.array(
        [cos_scaled[i, i + 1] for i in range(n_layers - 1)]
    )
    consec_unscaled = np.array(
        [cos_unscaled[i, i + 1] for i in range(n_layers - 1)]
    )

    # To final layer: cos(w_i, w_final)
    final_idx = n_layers - 1
    to_final_scaled = cos_scaled[:, final_idx]
    to_final_unscaled = cos_unscaled[:, final_idx]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Consecutive layers cos(w_i, w_{i+1})",
            "Similarity to final layer cos(w_i, w_final)",
        ],
    )

    # Consecutive
    fig.add_trace(
        go.Scatter(
            x=layers[:-1],
            y=consec_scaled,
            mode="lines+markers",
            name="Scaled",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=layers[:-1],
            y=consec_unscaled,
            mode="lines+markers",
            name="Unscaled",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Layer i", row=1, col=1)
    fig.update_yaxes(title_text="Cosine similarity", row=1, col=1)

    # To final
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=to_final_scaled,
            mode="lines+markers",
            name="Scaled",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=to_final_unscaled,
            mode="lines+markers",
            name="Unscaled",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Cosine similarity", row=1, col=2)

    fig.update_layout(
        title_text=f"Probe Direction Stability Across Layers ({pos_name})",
        width=1100,
        height=450,
    )

    _save_fig(fig, save_path)
    return fig


# ── Scaled vs Unscaled Deep Comparison ───────────────────────────────────────


def plot_scaled_vs_unscaled(
    auc_scaled_cv: np.ndarray,
    auc_unscaled_cv: np.ndarray,
    direction_agreement: np.ndarray,
    scale_cv: np.ndarray,
    scale_max_min: np.ndarray,
    pos_name: str,
    *,
    save_path: Path | None = None,
) -> go.Figure:
    """Three-panel scaled vs unscaled comparison (plotting only).

    All inputs are shape ``(n_layers,)`` arrays pre-computed by the notebook.

    Parameters
    ----------
    auc_scaled_cv, auc_unscaled_cv
        CV AUC by layer for scaled and unscaled probes.
    direction_agreement
        Cosine similarity between scaled-to-original and unscaled directions.
    scale_cv
        Coefficient of variation of scaler.scale_ by layer.
    scale_max_min
        Max/min ratio of scaler.scale_ by layer.
    pos_name
        Token position name shown in the title.
    """
    n_layers = len(auc_scaled_cv)
    layers = np.arange(n_layers)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "CV AUC by layer (scaled vs unscaled)",
            "Direction agreement (scaled->orig vs unscaled)",
            "Feature scale CV by layer",
        ],
        horizontal_spacing=0.08,
    )

    # a) AUC
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=auc_scaled_cv,
            mode="lines+markers",
            name="Scaled",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=auc_unscaled_cv,
            mode="lines+markers",
            name="Unscaled",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="ROC AUC (CV)", row=1, col=1)

    # b) Direction agreement
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=direction_agreement,
            mode="lines+markers",
            name="cos(scaled->orig, unscaled)",
            line=dict(color="#2ca02c", width=2),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Cosine similarity", row=1, col=2)

    # c) Feature scale CV
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=scale_cv,
            mode="lines+markers",
            name="CV(scaler.scale_)",
            line=dict(color="#d62728", width=2),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=np.log10(scale_max_min),
            mode="lines+markers",
            name="log10(max/min scale)",
            line=dict(color="#9467bd", width=2, dash="dash"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Layer", row=1, col=3)
    fig.update_yaxes(title_text="Scale disparity", row=1, col=3)

    fig.update_layout(
        title_text=f"Scaled vs Unscaled Probe Comparison ({pos_name})",
        width=1400,
        height=450,
    )

    _save_fig(fig, save_path)
    return fig


# ── Bias Analysis ────────────────────────────────────────────────────────────


def plot_bias_analysis(
    biases_scaled: np.ndarray,
    biases_unscaled: np.ndarray,
    pos_name: str,
    *,
    scalers: list | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """Bias analysis with optional StandardScaler statistics panel.

    When *scalers* is provided and contains fitted ``StandardScaler`` objects,
    a second subplot shows per-layer scaler statistics (max/min/mean of
    ``scale_`` and mean absolute ``mean_``).  Otherwise only the raw bias
    subplot is shown.

    Parameters
    ----------
    biases_scaled, biases_unscaled
        Shape ``(n_layers,)`` intercept values.
    pos_name
        Token position name shown in the title.
    scalers
        List of ``StandardScaler`` (one per layer) from the scaled probe.
        If ``None`` or all entries are ``None``, the scaler subplot is omitted.
    """
    n_layers = len(biases_scaled)
    layers = np.arange(n_layers)

    has_scalers = (
        scalers is not None
        and len(scalers) > 0
        and any(s is not None for s in scalers)
    )

    if has_scalers:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Raw bias value", "StandardScaler statistics"],
        )
    else:
        fig = go.Figure()

    # ── Raw bias (always shown) ──────────────────────────────────────────
    raw_kwargs = dict(row=1, col=1) if has_scalers else {}

    fig.add_trace(
        go.Scatter(
            x=layers,
            y=biases_scaled,
            mode="lines+markers",
            name="Scaled",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=4),
        ),
        **raw_kwargs,
    )
    fig.add_trace(
        go.Scatter(
            x=layers,
            y=biases_unscaled,
            mode="lines+markers",
            name="Unscaled",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=4),
        ),
        **raw_kwargs,
    )

    if has_scalers:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
        fig.update_xaxes(title_text="Layer", row=1, col=1)
        fig.update_yaxes(title_text="Bias", row=1, col=1)
    else:
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_xaxes(title_text="Layer")
        fig.update_yaxes(title_text="Bias")

    # ── Scaler statistics (only when scalers are available) ───────────────
    if has_scalers:
        scale_max = np.array([
            np.max(s.scale_) for s in scalers if s is not None
        ])
        scale_min = np.array([
            np.min(s.scale_) for s in scalers if s is not None
        ])
        scale_mean = np.array([
            np.mean(s.scale_) for s in scalers if s is not None
        ])
        translation = np.array([
            np.mean(np.abs(s.mean_)) for s in scalers if s is not None
        ])
        scaler_layers = np.arange(len(scale_max))

        fig.add_trace(
            go.Scatter(
                x=scaler_layers,
                y=scale_max,
                mode="lines+markers",
                name="max(scale_)",
                line=dict(color="#d62728", width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=scaler_layers,
                y=scale_min,
                mode="lines+markers",
                name="min(scale_)",
                line=dict(color="#2ca02c", width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=scaler_layers,
                y=scale_mean,
                mode="lines+markers",
                name="mean(scale_)",
                line=dict(color="#9467bd", width=2),
                marker=dict(size=4),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=scaler_layers,
                y=translation,
                mode="lines+markers",
                name="mean(|mean_|)",
                line=dict(color="#8c564b", width=2, dash="dash"),
                marker=dict(size=4),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Layer", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)

    fig.update_layout(
        title_text=f"Probe Bias (Intercept) by Layer ({pos_name})",
        width=1100,
        height=400,
    )

    _save_fig(fig, save_path)
    return fig


# ── Direction Agreement ─────────────────────────────────────────────────────


def plot_direction_agreement(
    weights_scaled: np.ndarray,
    weights_unscaled: np.ndarray,
    pos_name: str,
    *,
    cv_scores_scaled: pd.DataFrame | None = None,
    cv_scores_unscaled: pd.DataFrame | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """Direction agreement and (optionally) ROC AUC comparison for scaled vs unscaled probes.

    Parameters
    ----------
    weights_scaled, weights_unscaled : (n_layers, d_model)
        Unit-norm direction vectors from scaled and unscaled probes.
    pos_name : str
        Token position label for the title.
    cv_scores_scaled, cv_scores_unscaled : pd.DataFrame or None
        If both provided, a second subplot shows ROC AUC per layer for each.
        Expected columns: ``layer``, ``roc_auc_mean``, ``roc_auc_std``.
    """
    agreement = np.sum(weights_scaled * weights_unscaled, axis=1)
    layers = np.arange(len(agreement))
    has_auc = cv_scores_scaled is not None and cv_scores_unscaled is not None

    if has_auc:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Direction agreement (scaled vs unscaled)",
                "ROC AUC by layer (scaled vs unscaled)",
            ],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # ── Left: direction agreement ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=layers, y=agreement,
        mode="lines+markers", name="cos(scaled, unscaled)",
        line=dict(color=COLORS[0], width=2), marker=dict(size=4),
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="perfect agreement", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Cosine Similarity", row=1, col=1)

    # ── Right: ROC AUC curves ────────────────────────────────────────────
    if has_auc:
        for df_cv, label, color, dash in [
            (cv_scores_scaled, "Scaled", COLORS[0], "solid"),
            (cv_scores_unscaled, "Unscaled", COLORS[1], "dash"),
        ]:
            l = df_cv["layer"].values
            m = df_cv["roc_auc_mean"].values
            s = df_cv["roc_auc_std"].values
            fig.add_trace(go.Scatter(
                x=l, y=m, mode="lines+markers", name=label,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=4),
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=np.concatenate([l, l[::-1]]),
                y=np.concatenate([m + s, (m - s)[::-1]]),
                fill="toself", fillcolor=color, opacity=0.10,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=2)
        fig.update_xaxes(title_text="Layer", row=1, col=2)
        fig.update_yaxes(title_text="ROC AUC", row=1, col=2)

    fig.update_layout(
        title=f"Scaled vs Unscaled Probe — {pos_name}",
        template="plotly_white",
        width=1100,
        height=400,
    )

    _save_fig(fig, save_path)
    return fig


# ── Metadata Importance ──────────────────────────────────────────────────────


def plot_metadata_importance(
    linear_coefs: np.ndarray,
    feature_names: list[str],
    boosted_imps: np.ndarray,
    boosted_stds: np.ndarray,
    feature_names_cat: list[str],
    *,
    save_path: Path | None = None,
) -> go.Figure:
    """Side-by-side linear coefficient and boosted permutation importance bars.

    Parameters
    ----------
    linear_coefs
        Shape ``(n_features,)`` — scaled logistic regression coefficients.
    feature_names
        Names for all features (linear).
    boosted_imps, boosted_stds
        Shape ``(n_features_cat,)`` — permutation importance mean and std.
    feature_names_cat
        Names for categorical features (boosted).
    """
    order_lin = np.argsort(np.abs(linear_coefs))[::-1]
    order_bst = np.argsort(boosted_imps)[::-1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Linear (LogReg scaled coefficients, all features)",
            "Boosted (permutation importance, categorical only)",
        ],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Bar(
            x=[feature_names[i] for i in order_lin],
            y=linear_coefs[order_lin],
            marker_color=[
                "#d62728" if c > 0 else "#1f77b4"
                for c in linear_coefs[order_lin]
            ],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[feature_names_cat[i] for i in order_bst],
            y=boosted_imps[order_bst],
            error_y=dict(
                type="data",
                array=boosted_stds[order_bst],
                thickness=1.2,
                width=3,
            ),
            marker_color="#2ca02c",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Metadata Classifier — Feature Importance Comparison",
        xaxis_tickangle=-40,
        xaxis2_tickangle=-40,
        width=1300,
        height=500,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Coefficient", row=1, col=1)
    fig.update_yaxes(title_text="Score drop when shuffled", row=1, col=2)

    _save_fig(fig, save_path)
    return fig


# ── Metadata Ablation ────────────────────────────────────────────────────────


def plot_metadata_ablation(
    df_abl_lin: pd.DataFrame,
    df_abl_bst: pd.DataFrame,
    ctrl_linear: float,
    ctrl_boosted: float,
    *,
    save_path: Path | None = None,
) -> go.Figure:
    """Side-by-side group ablation bars for linear and boosted classifiers.

    Parameters
    ----------
    df_abl_lin, df_abl_bst
        DataFrames from :func:`metadata_clf.run_group_ablation`.  Expected
        columns: ``group_dropped``, ``roc_auc_mean``, ``roc_auc_std``,
        ``roc_auc_drop``.
    ctrl_linear, ctrl_boosted
        Baseline horizontal lines.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Linear ablation (roc_auc)",
            "Boosted ablation — categorical only (roc_auc)",
        ],
        horizontal_spacing=0.08,
    )

    for col, (df_abl, baseline) in enumerate(
        [(df_abl_lin, ctrl_linear), (df_abl_bst, ctrl_boosted)], start=1
    ):
        for _, row in df_abl.iterrows():
            is_baseline = row["group_dropped"] == "none (baseline)"
            drop = row["roc_auc_drop"]
            color = (
                "#aec7e8"
                if is_baseline
                else (
                    "#d62728"
                    if drop > 0.02
                    else "#ff9896" if drop > 0 else "#1f77b4"
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[row["group_dropped"]],
                    y=[row["roc_auc_mean"]],
                    error_y=dict(
                        type="data",
                        array=[row["roc_auc_std"]],
                        thickness=1.5,
                        width=5,
                    ),
                    marker_color=color,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['group_dropped']}</b><br>"
                        f"roc_auc={row['roc_auc_mean']:.4f} "
                        f"+/- {row['roc_auc_std']:.4f}<br>"
                        f"drop={row['roc_auc_drop']:+.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="black",
            line_width=1.2,
            row=1,
            col=col,
        )

    y_min = min(
        df_abl_lin["roc_auc_mean"].min(), df_abl_bst["roc_auc_mean"].min()
    ) - 0.05
    y_max = max(
        df_abl_lin["roc_auc_mean"].max(), df_abl_bst["roc_auc_mean"].max()
    ) + 0.03
    fig.update_yaxes(range=[max(0.4, y_min), min(1.0, y_max)])

    fig.update_layout(
        title=(
            "Metadata Classifier — Group Ablation Comparison (roc_auc)<br>"
            "<sub>Each bar = model retrained WITHOUT that feature group. "
            "Drop = baseline - ablated.</sub>"
        ),
        width=1100,
        height=480,
        template="plotly_white",
        bargap=0.35,
    )

    _save_fig(fig, save_path)
    return fig


# ── Fold-Level Generalization ─────────────────────────────────────────────────


def plot_fold_generalization(
    fold_weights: np.ndarray,
    fold_cmds: dict[str, np.ndarray] | np.ndarray,
    fold_scores: np.ndarray,
    fold_group_names: list[str] | None = None,
    layer: int = 0,
    *,
    constraint_cmds: dict[str, np.ndarray] | None = None,
    save_path: Path | None = None,
) -> go.Figure:
    """3x3 fold-level generalization analysis for one layer.

    Parameters
    ----------
    fold_weights : (n_folds, d_model)
        Unit-norm probe directions from each CV fold.
    fold_cmds : dict[str, ndarray] or (n_folds, d_model)
        Unit-norm CMD vectors keyed by held-out group name, or a plain array
        aligned with *fold_group_names*.
    fold_scores : (n_folds,)
        Validation ROC AUC per fold.
    fold_group_names : (n_folds,) or None
        Constraint type label for each fold's held-out group.  When *None*,
        labels are inferred from *fold_cmds* keys (if dict) or default to
        "Fold 0", "Fold 1", etc.
    layer : int
        Layer index shown in the title.
    constraint_cmds : dict[str, ndarray] or None
        Per-constraint CMD vectors (computed within each constraint type).
        If provided, a third row is added.
    save_path : Path or None
        If given, figure is saved as HTML.
    """
    n_folds = fold_weights.shape[0]

    if fold_group_names is None:
        if isinstance(fold_cmds, dict):
            fold_group_names = list(fold_cmds.keys())
        else:
            fold_group_names = [f"Fold {i}" for i in range(n_folds)]

    # Labels: "w/o {name}" for fold-level axes, bare names for rightmost col
    wo_names = [f"w/o {name}" for name in fold_group_names]
    bare_names = list(fold_group_names)

    # Normalise fold_cmds to array aligned with fold_group_names
    if isinstance(fold_cmds, dict):
        fold_cmds_arr = np.array([fold_cmds[name] for name in fold_group_names])
    else:
        fold_cmds_arr = fold_cmds

    # Cosine similarity matrices
    cos_probe = cos_sim(fold_weights)
    cos_cmd = cos_sim(fold_cmds_arr)

    # Mean off-diagonal cosine similarity per fold
    mask = ~np.eye(n_folds, dtype=bool)
    mean_cos_probe = np.array([cos_probe[i, mask[i]].mean() for i in range(n_folds)])
    mean_cos_cmd = np.array([cos_cmd[i, mask[i]].mean() for i in range(n_folds)])

    # Per-fold CMD-probe cosine similarity
    cmd_probe_cos = np.sum(fold_weights * fold_cmds_arr, axis=1)

    # Per-constraint CMD data (rows 3-4)
    has_constraint = constraint_cmds is not None
    n_rows = 4 if has_constraint else 2

    if has_constraint:
        # Filter to constraints present in both fold_group_names and constraint_cmds
        cnames = [n for n in fold_group_names if n in constraint_cmds]
        constraint_cmds_arr = np.array([constraint_cmds[n] for n in cnames])
        cos_constraint = cos_sim(constraint_cmds_arr)
        n_c = len(cnames)
        mask_c = ~np.eye(n_c, dtype=bool)
        mean_cos_constraint = np.array(
            [cos_constraint[i, mask_c[i]].mean() for i in range(n_c)]
        )
        # cos(per-constraint CMD, corresponding fold CMD w/o that constraint)
        constraint_vs_fold_cmd = np.array([
            float(np.dot(constraint_cmds[n], fold_cmds_arr[i]))
            for i, n in enumerate(fold_group_names)
            if n in constraint_cmds
        ])
        # cos(per-constraint CMD, corresponding fold probe w/o that constraint)
        constraint_vs_fold_probe = np.array([
            float(np.dot(constraint_cmds[n], fold_weights[i]))
            for i, n in enumerate(fold_group_names)
            if n in constraint_cmds
        ])

    subplot_titles = [
        "Probe direction similarity (fold × fold)",
        "Mean probe cos sim per fold",
        "Generalization AUC per constraint",
        "Fold CMD similarity (fold × fold)",
        "Mean fold CMD cos sim per fold",
        "Fold CMD–Probe cos sim per fold",
    ]
    if has_constraint:
        subplot_titles += [
            "Per-constraint CMD similarity",
            "Mean per-constraint CMD cos sim",
            "Per-constraint CMD vs fold CMD",
            "",  # row 4, col 1 (empty)
            "",  # row 4, col 2 (empty)
            "Per-constraint CMD vs fold probe",
        ]

    row_heights = (
        [0.3, 0.3, 0.3, 0.1] if n_rows == 4 else None
    )
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        column_widths=[0.4, 0.3, 0.3],
        row_heights=row_heights,
    )

    # ── Row 1: Probe directions ──────────────────────────────────────────

    # Top-left: probe cosine similarity heatmap
    fig.add_trace(
        go.Heatmap(
            z=cos_probe,
            x=wo_names,
            y=wo_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(cos_probe, 2).astype(str),
            texttemplate="%{text}",
            colorbar=dict(title="cos", x=0.33, len=0.22, y=0.88),
            hovertemplate="%{y} vs %{x}<br>cos=%{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Top-middle: mean off-diagonal probe cos sim
    mean_probe_overall = float(np.mean(mean_cos_probe))
    fig.add_trace(
        go.Bar(
            x=wo_names,
            y=mean_cos_probe,
            marker_color=[
                "#d62728" if v < mean_probe_overall else "#1f77b4"
                for v in mean_cos_probe
            ],
            showlegend=False,
            hovertemplate="%{x}<br>mean cos=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=mean_probe_overall,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text=f"mean={mean_probe_overall:.3f}",
        annotation_position="top right",
        row=1,
        col=2,
    )

    # Top-right: validation AUC per constraint (bare names)
    mean_auc = float(np.mean(fold_scores))
    fig.add_trace(
        go.Bar(
            x=bare_names,
            y=fold_scores,
            marker_color=[
                "#d62728" if s < mean_auc else "#1f77b4" for s in fold_scores
            ],
            showlegend=False,
            hovertemplate="%{x}<br>AUC=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_hline(
        y=mean_auc,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text=f"mean={mean_auc:.3f}",
        annotation_position="top right",
        row=1,
        col=3,
    )

    # ── Row 2: Fold CMD directions ───────────────────────────────────────

    # Mid-left: fold CMD cosine similarity heatmap
    fig.add_trace(
        go.Heatmap(
            z=cos_cmd,
            x=wo_names,
            y=wo_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(cos_cmd, 2).astype(str),
            texttemplate="%{text}",
            colorbar=dict(title="cos", x=0.33, len=0.22, y=0.6),
            hovertemplate="%{y} vs %{x}<br>cos=%{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Mid-middle: mean off-diagonal fold CMD cos sim
    mean_cmd_overall = float(np.mean(mean_cos_cmd))
    fig.add_trace(
        go.Bar(
            x=wo_names,
            y=mean_cos_cmd,
            marker_color=[
                "#d62728" if v < mean_cmd_overall else "#9467bd"
                for v in mean_cos_cmd
            ],
            showlegend=False,
            hovertemplate="%{x}<br>mean cos=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(
        y=mean_cmd_overall,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text=f"mean={mean_cmd_overall:.3f}",
        annotation_position="top right",
        row=2,
        col=2,
    )

    # Mid-right: fold CMD–Probe cosine similarity (w/o labels)
    mean_cmd_probe = float(np.mean(cmd_probe_cos))
    fig.add_trace(
        go.Bar(
            x=wo_names,
            y=cmd_probe_cos,
            marker_color="#2ca02c",
            showlegend=False,
            hovertemplate="%{x}<br>cos(CMD,probe)=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_hline(
        y=mean_cmd_probe,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text=f"mean={mean_cmd_probe:.3f}",
        annotation_position="top right",
        row=2,
        col=3,
    )

    # ── Row 3: Per-constraint CMD (only if constraint_cmds provided) ─────

    if has_constraint:
        # Bottom-left: per-constraint CMD cosine similarity heatmap
        fig.add_trace(
            go.Heatmap(
                z=cos_constraint,
                x=cnames,
                y=cnames,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(cos_constraint, 2).astype(str),
                texttemplate="%{text}",
                colorbar=dict(title="cos", x=0.33, len=0.22, y=0.32),
                hovertemplate="%{y} vs %{x}<br>cos=%{z:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # Bottom-middle: mean off-diagonal per-constraint CMD cos sim
        mean_constraint_overall = float(np.mean(mean_cos_constraint))
        fig.add_trace(
            go.Bar(
                x=cnames,
                y=mean_cos_constraint,
                marker_color=[
                    "#d62728" if v < mean_constraint_overall else "#8c564b"
                    for v in mean_cos_constraint
                ],
                showlegend=False,
                hovertemplate="%{x}<br>mean cos=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=2,
        )
        fig.add_hline(
            y=mean_constraint_overall,
            line_dash="dash",
            line_color="black",
            line_width=1.5,
            annotation_text=f"mean={mean_constraint_overall:.3f}",
            annotation_position="top right",
            row=3,
            col=2,
        )

        # Row 3 right: per-constraint CMD vs fold CMD cos sim
        mean_c_vs_fold_cmd = float(np.mean(constraint_vs_fold_cmd))
        fig.add_trace(
            go.Bar(
                x=cnames,
                y=constraint_vs_fold_cmd,
                marker_color="#ff7f0e",
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>cos(constraint CMD, fold CMD)=%{y:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=3,
            col=3,
        )
        fig.add_hline(
            y=mean_c_vs_fold_cmd,
            line_dash="dash",
            line_color="black",
            line_width=1.5,
            annotation_text=f"mean={mean_c_vs_fold_cmd:.3f}",
            annotation_position="top right",
            row=3,
            col=3,
        )

        # ── Row 4: Per-constraint CMD vs fold probe ──────────────────────

        # Row 4 right: per-constraint CMD vs fold probe cos sim
        mean_c_vs_fold_probe = float(np.mean(constraint_vs_fold_probe))
        fig.add_trace(
            go.Bar(
                x=cnames,
                y=constraint_vs_fold_probe,
                marker_color="#e377c2",
                showlegend=False,
                hovertemplate=(
                    "%{x}<br>cos(constraint CMD, fold probe)=%{y:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=4,
            col=3,
        )
        fig.add_hline(
            y=mean_c_vs_fold_probe,
            line_dash="dash",
            line_color="black",
            line_width=1.5,
            annotation_text=f"mean={mean_c_vs_fold_probe:.3f}",
            annotation_position="top right",
            row=4,
            col=3,
        )

    # Consistent y-axis [0, 1] for all bar subplots
    for row in range(1, n_rows + 1):
        for col in (2, 3):
            fig.update_yaxes(range=[0, 1], row=row, col=col)

    fig.update_layout(
        title_text=(
            f"Fold-Level Generalization Analysis — Layer {layer}<br>"
            f"<sub>{n_folds} folds (LOGO by constraint type)</sub>"
        ),
        width=1400,
        height=1600 if n_rows == 4 else 450 * n_rows,
        template="plotly_white",
    )

    _save_fig(fig, save_path)
    return fig


# ── Summary Table ────────────────────────────────────────────────────────────


def build_summary_table(
    probe_results: dict[str, dict[str, pd.DataFrame]],
    token_positions: list[str],
    n_layers: int,
    ctrl_linear: float,
    ctrl_boosted: float | None = None,
) -> pd.DataFrame:
    """Build a summary DataFrame with peak metrics per backend and position.

    Parameters
    ----------
    probe_results
        Maps *backend_label* -> {*pos_name*: DataFrame}.  Each DataFrame
        must contain columns ``layer``, ``roc_auc_mean``, ``roc_auc_std``,
        ``balanced_accuracy_mean``, ``balanced_accuracy_std``.
    token_positions
        Which positions to include.
    n_layers
        Total number of model layers (for computing ``peak_layer_%``).
    ctrl_linear
        Linear metadata control AUC.
    ctrl_boosted
        Boosted metadata control AUC (optional).

    Returns
    -------
    pd.DataFrame
        Summary table sorted by ``(token_position, backend)``.
    """
    rows: list[dict] = []

    for pos in token_positions:
        for backend, pos_dict in probe_results.items():
            if pos not in pos_dict:
                continue
            df_res = pos_dict[pos]

            # Peak ROC AUC
            auc_best_idx = int(df_res["roc_auc_mean"].idxmax())
            auc_best = df_res.loc[auc_best_idx]

            # Peak Balanced Accuracy
            bacc_best_idx = int(df_res["balanced_accuracy_mean"].idxmax())
            bacc_best = df_res.loc[bacc_best_idx]

            row_dict = {
                "backend": backend,
                "token_position": pos,
                "peak_roc_auc": round(float(auc_best["roc_auc_mean"]), 4),
                "roc_auc_std": round(float(auc_best["roc_auc_std"]), 4),
                "roc_auc_peak_layer": int(auc_best["layer"]),
                "roc_auc_peak_layer_%": (
                    f"{int(auc_best['layer']) / n_layers * 100:.0f}%"
                ),
                "peak_balanced_accuracy": round(
                    float(bacc_best["balanced_accuracy_mean"]), 4
                ),
                "balanced_accuracy_std": round(
                    float(bacc_best["balanced_accuracy_std"]), 4
                ),
                "balanced_accuracy_peak_layer": int(bacc_best["layer"]),
                "linear_ctrl": round(ctrl_linear, 4),
                "gap_vs_linear": round(
                    float(auc_best["roc_auc_mean"]) - ctrl_linear, 4
                ),
            }
            if ctrl_boosted is not None:
                row_dict["boosted_ctrl"] = round(ctrl_boosted, 4)
                row_dict["gap_vs_boosted"] = round(
                    float(auc_best["roc_auc_mean"]) - ctrl_boosted, 4
                )

            rows.append(row_dict)

    summary = (
        pd.DataFrame(rows)
        .sort_values(["token_position", "backend"])
        .reset_index(drop=True)
    )
    return summary
