"""Plotly figure builders for the HTML report.

Each function returns a go.Figure (or None when data is insufficient).
Computation notes are provided as HTML text in layout.py, not inside figures.
"""

from __future__ import annotations
from collections import Counter, defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import (
    short_model, compute_scr, compute_cr,
    STRENGTH_ORDER, LABEL_ORDER, LABEL_COLORS,
    DEFAULT_USER_STYLE, DEFAULT_STRENGTH,
    default_conflict_recs, _model_sort_key,
)

# ── Colour palette ────────────────────────────────────────────────────────
MODEL_PALETTE = [
    "#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c",
    "#1abc9c", "#f39c12", "#8e44ad",
]


def _model_color(idx: int) -> str:
    return MODEL_PALETTE[idx % len(MODEL_PALETTE)]


# ── 1. Label distribution by model ───────────────────────────────────────

def fig_label_by_model(records: list[dict]) -> go.Figure:
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        counts[short_model(r["model"])][r["label"]] += 1
    models = sorted(counts, key=_model_sort_key)

    fig = go.Figure()
    for label in LABEL_ORDER:
        vals = []
        for m in models:
            total = sum(counts[m].values()) or 1
            vals.append(counts[m].get(label, 0) / total * 100)
        fig.add_trace(go.Bar(
            name=label.replace("_", " ").title(),
            x=models, y=vals,
            marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
        ))
    fig.update_layout(
        barmode="stack",
        title="Label Distribution by Model (All Conditions)",
        yaxis_title="% of responses", yaxis_range=[0, 105],
        legend_title="Label",
        xaxis_tickangle=-30,
    )
    return fig


# ── 2. SCR by strength (default user style, Condition C) ─────────────────

def fig_scr_by_strength(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["user_style"] == DEFAULT_USER_STYLE]
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    strengths = [s for s in STRENGTH_ORDER if any(r["strength"] == s for r in c)]

    fig = go.Figure()
    for i, model in enumerate(models):
        m_recs = [r for r in c if short_model(r["model"]) == model]
        scrs = [compute_scr([r for r in m_recs if r["strength"] == s]) for s in strengths]
        ns = [len([r for r in m_recs if r["strength"] == s]) for s in strengths]
        fig.add_trace(go.Scatter(
            x=strengths, y=scrs, mode="lines+markers",
            name=model, marker_size=10, marker_color=_model_color(i),
            text=[f"n={n}" for n in ns],
            hovertemplate="%{x}: SCR=%{y:.1%} (%{text})<extra>%{fullData.name}</extra>",
        ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                  annotation_text="0.7")
    fig.update_layout(
        title="SCR vs Prompt Strength (Condition C)",
        xaxis_title="System Prompt Strength", yaxis_title="SCR",
        yaxis_range=[0, 1.1],
    )
    return fig


# ── 3. SCR by constraint type (default filters) ─────────────────────────

def fig_scr_by_constraint(records: list[dict]) -> go.Figure | None:
    c = default_conflict_recs(records)
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))

    fig = go.Figure()
    for i, model in enumerate(models):
        m_recs = [r for r in c if short_model(r["model"]) == model]
        scrs = [compute_scr([r for r in m_recs if r["constraint_type"] == ct]) for ct in cts]
        fig.add_trace(go.Bar(
            name=model, x=[ct.capitalize() for ct in cts], y=scrs,
            marker_color=_model_color(i),
            text=[f"{v:.0%}" for v in scrs], textposition="outside",
        ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="red")
    fig.update_layout(
        barmode="group",
        title="SCR by Constraint Type (Condition C)",
        yaxis_title="SCR", yaxis_range=[0, 1.15],
    )
    return fig


# ── 4. Directional SCR (default filters) ─────────────────────────────────

def fig_directional_scr(records: list[dict]) -> go.Figure | None:
    """2×2 subplot grid: one subplot per experiment pair, showing A→B vs B→A SCR per model."""
    c = default_conflict_recs(records)
    if not c:
        return None

    # Discover experiment pairs from data
    pairs = sorted(set((r["constraint_type"], r["option_a"], r["option_b"]) for r in c))
    if not pairs:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    n_pairs = len(pairs)
    n_cols = min(2, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    titles = [f"{ct}: {oa} vs {ob}" for ct, oa, ob in pairs]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        shared_yaxes=True, horizontal_spacing=0.10,
                        vertical_spacing=0.18)

    for pidx, (ct, oa, ob) in enumerate(pairs):
        row = pidx // n_cols + 1
        col = pidx % n_cols + 1
        p_recs = [r for r in c
                   if r["constraint_type"] == ct
                   and r["option_a"] == oa and r["option_b"] == ob]

        a2b_vals, b2a_vals = [], []
        for m in models:
            m_recs = [r for r in p_recs if short_model(r["model"]) == m]
            a2b_vals.append(compute_scr([r for r in m_recs if r["direction"] == "a_to_b"]))
            b2a_vals.append(compute_scr([r for r in m_recs if r["direction"] == "b_to_a"]))

        fig.add_trace(go.Bar(
            name="A→B", x=models, y=a2b_vals, marker_color="#3498db",
            text=[f"{v:.0%}" if v < 1.0 else "" for v in a2b_vals], textposition="outside",
            showlegend=(pidx == 0),
        ), row=row, col=col)
        fig.add_trace(go.Bar(
            name="B→A", x=models, y=b2a_vals, marker_color="#e67e22",
            text=[f"{v:.0%}" if v < 1.0 else "" for v in b2a_vals], textposition="outside",
            showlegend=(pidx == 0),
        ), row=row, col=col)

        # Annotate asymmetry deltas
        for i, m in enumerate(models):
            d = abs(a2b_vals[i] - b2a_vals[i])
            color = "red" if d > 0.15 else "black"
            fig.add_annotation(
                x=m, y=max(a2b_vals[i], b2a_vals[i]) + 0.18,
                text=f"Δ={d:.2f}", showarrow=False,
                font=dict(color=color, size=10),
                xref=f"x{pidx + 1 if pidx else ''}", yref=f"y{pidx + 1 if pidx else ''}",
            )

        fig.update_yaxes(range=[0, 1.4], row=row, col=col)
        fig.update_xaxes(tickangle=-30, row=row, col=col)

    fig.update_layout(
        barmode="group",
        title="Directional SCR by Experiment Pair (Cond. C)",
        height=380 * n_rows,
    )
    return fig


# ── 5. Heatmap (strength × constraint, default user style) ───────────────

def fig_heatmap(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["user_style"] == DEFAULT_USER_STYLE]
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    strengths = [s for s in STRENGTH_ORDER if any(r["strength"] == s for r in c)]

    # Use 3x3 grid layout to avoid overlapping titles
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=models, shared_yaxes=True,
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    for idx, model in enumerate(models):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        m_recs = [r for r in c if short_model(r["model"]) == model]
        z, text = [], []
        for ct in cts:
            row_z, row_t = [], []
            for s in strengths:
                cell = [r for r in m_recs if r["constraint_type"] == ct and r["strength"] == s]
                scr = compute_scr(cell) if cell else None
                row_z.append(scr)
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "—")
            z.append(row_z)
            text.append(row_t)
        fig.add_trace(go.Heatmap(
            z=z, x=strengths, y=[ct.capitalize() for ct in cts],
            text=text, texttemplate="%{text}", zmin=0, zmax=1,
            colorscale="RdYlGn", showscale=(idx == n_models - 1),
        ), row=row, col=col)
    fig.update_layout(
        title="SCR Heatmap: Strength × Constraint (Cond. C)",
        height=220 * n_rows,
    )
    return fig


# ── 6. Label distribution by condition (per model) ───────────────────────

def fig_by_condition(records: list[dict]) -> go.Figure:
    models = sorted(set(short_model(r["model"]) for r in records), key=_model_sort_key)
    
    # Use 3x3 grid layout to avoid overlapping titles
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Only include labels that actually appear in the data
    present_labels = [l for l in LABEL_ORDER if any(r["label"] == l for r in records)]

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=models, shared_yaxes=True,
                        horizontal_spacing=0.08, vertical_spacing=0.15)

    for idx, model in enumerate(models):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        m_recs = [r for r in records if short_model(r["model"]) == model]
        conds = sorted(set(r["condition"] for r in m_recs))
        counts: dict[str, Counter] = defaultdict(Counter)
        for r in m_recs:
            counts[f"Cond. {r['condition']}"][r["label"]] += 1
        cats = [f"Cond. {c}" for c in conds]
        for label in present_labels:
            vals = []
            for cat in cats:
                total = sum(counts[cat].values()) or 1
                vals.append(counts[cat].get(label, 0) / total * 100)
            fig.add_trace(go.Bar(
                name=label.replace("_", " ").title(), x=cats, y=vals,
                marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
                showlegend=(idx == 0),
            ), row=row, col=col)
    fig.update_layout(
        barmode="stack",
        title="Label Distribution by Condition",
        yaxis_title="% of responses", height=200 * n_rows,
    )
    return fig


# ── 7. User template effect (with_instruction vs jailbreak) ──────────────

def fig_user_template(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["strength"] == DEFAULT_STRENGTH]
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    styles = sorted(set(r["user_style"] for r in c))

    fig = go.Figure()
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i, style in enumerate(styles):
        scrs = [compute_scr([r for r in c if short_model(r["model"]) == m and r["user_style"] == style])
                for m in models]
        fig.add_trace(go.Bar(
            name=style, x=models, y=scrs,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.0%}" for v in scrs], textposition="outside",
        ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="red")
    fig.update_layout(
        barmode="group",
        title="User Template Effect on SCR (Cond. C)",
        yaxis_title="SCR", yaxis_range=[0, 1.15],
        xaxis_tickangle=-30,
    )
    return fig


# ── 8. Cross-model SCR comparison (default filters) ──────────────────────

def fig_cross_model_scr(records: list[dict]) -> go.Figure | None:
    c = default_conflict_recs(records)
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    scrs = [compute_scr([r for r in c if short_model(r["model"]) == m]) for m in models]
    ns = [len([r for r in c if short_model(r["model"]) == m]) for m in models]
    colors = ["#2ecc71" if v >= 0.7 else "#e74c3c" if v < 0.3 else "#f39c12" for v in scrs]

    fig = go.Figure(go.Bar(
        y=models, x=scrs, orientation="h",
        marker_color=colors,
        text=[f"{v:.0%} (n={n})" for v, n in zip(scrs, ns)],
        textposition="outside",
    ))
    fig.add_vline(x=0.7, line_dash="dash", line_color="gray",
                  annotation_text="0.7")
    fig.update_layout(
        title="Overall SCR by Model (Condition C)",
        xaxis_title="SCR", xaxis_range=[0, 1.15],
    )
    return fig


# ── 9. Task effect (default filters) ─────────────────────────────────────

def fig_task_effect(records: list[dict]) -> go.Figure | None:
    c = default_conflict_recs(records)
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    tasks = sorted(set(r["task_id"] for r in c))

    fig = go.Figure()
    for i, model in enumerate(models):
        m_recs = [r for r in c if short_model(r["model"]) == model]
        scrs = [compute_scr([r for r in m_recs if r["task_id"] == t]) for t in tasks]
        fig.add_trace(go.Bar(
            name=model, x=[t.capitalize() for t in tasks], y=scrs,
            marker_color=_model_color(i),
            text=[f"{v:.0%}" for v in scrs], textposition="outside",
        ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="red")
    fig.update_layout(
        barmode="group",
        title="SCR by Task Type (Condition C)",
        yaxis_title="SCR", yaxis_range=[0, 1.15],
    )
    return fig


# ── 10. Confidence distribution ──────────────────────────────────────────

def fig_confidence_distribution(records: list[dict]) -> go.Figure:
    models = sorted(set(short_model(r["model"]) for r in records), key=_model_sort_key)
    fig = make_subplots(rows=1, cols=len(models), subplot_titles=models, shared_yaxes=True)

    for idx, model in enumerate(models, 1):
        m_recs = [r for r in records if short_model(r["model"]) == model]
        for label in LABEL_ORDER:
            vals = [r["confidence"] for r in m_recs if r["label"] == label]
            if vals:
                fig.add_trace(go.Histogram(
                    x=vals, name=label.replace("_", " ").title(),
                    marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
                    opacity=0.7, nbinsx=20,
                    showlegend=(idx == 1),
                ), row=1, col=idx)
    fig.update_layout(
        barmode="overlay",
        title="Classification Confidence Distribution",
        height=350,
    )
    return fig


# ── 11. Jailbreak vs Medium by Constraint Type ───────────────────────────

def fig_jailbreak_vs_medium(records: list[dict]) -> go.Figure | None:
    """Compare SCR under jailbreak user template vs default (with_instruction),
    both at medium strength, broken down by constraint type."""
    c = [r for r in records
         if r["condition"] == "C" and r["strength"] == DEFAULT_STRENGTH]
    if not c:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    styles = [DEFAULT_USER_STYLE, "jailbreak"]
    style_labels = {DEFAULT_USER_STYLE: "Default (with_instruction)", "jailbreak": "Jailbreak"}
    style_colors = {DEFAULT_USER_STYLE: "#3498db", "jailbreak": "#e74c3c"}

    fig = make_subplots(
        rows=1, cols=len(cts),
        subplot_titles=[ct.capitalize() for ct in cts],
        shared_yaxes=True,
    )

    for col, ct in enumerate(cts, 1):
        ct_recs = [r for r in c if r["constraint_type"] == ct]
        for style in styles:
            scrs = []
            ns = []
            for m in models:
                cell = [r for r in ct_recs
                        if short_model(r["model"]) == m and r["user_style"] == style]
                scrs.append(compute_scr(cell))
                ns.append(len(cell))
            fig.add_trace(go.Bar(
                name=style_labels[style], x=models, y=scrs,
                marker_color=style_colors[style],
                text=[f"{v:.0%}<br>n={n}" for v, n in zip(scrs, ns)],
                textposition="outside",
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.add_hline(y=0.7, line_dash="dash", line_color="red")
    for i in range(1, len(cts) + 1):
        fig.update_xaxes(tickangle=-30, row=1, col=i)
        fig.update_yaxes(range=[0, 1.25], row=1, col=i)
    fig.update_layout(
        barmode="group",
        title="Jailbreak vs Default User Template — SCR by Constraint Type (Cond. C)",
        yaxis_title="SCR",
        height=480,
    )
    return fig
