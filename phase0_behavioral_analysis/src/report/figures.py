"""Plotly figure builders for the HTML report.

Each function returns a go.Figure (or None when data is insufficient).
Computation notes are provided as HTML text in layout.py, not inside figures.
"""

from __future__ import annotations
from collections import Counter, defaultdict

import plotly.graph_objects as go
import plotly.colors as pc
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


# ── 5. Constraint type × direction triangle-split matrix ─────────────────

def _scr_to_color(scr: float | None) -> str:
    """Map SCR [0,1] to a RdYlGn hex color."""
    if scr is None:
        return "rgba(210,210,210,0.5)"
    v = max(0.0, min(1.0, scr))
    colors = pc.sample_colorscale("RdYlGn", [v])
    return colors[0]


def fig_constraint_directions(records: list[dict]) -> go.Figure | None:
    """Triangle-split matrix: each cell divided diagonally into two triangles,
    one per counterbalancing direction. Rows = models, cols = constraint types.
    * marks cells where the two directions differ by > 0.15."""
    c = default_conflict_recs(records)
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    n_m, n_c = len(models), len(cts)

    # Precompute SCR for each (model, constraint, direction)
    scr_d1 = {}  # direction "a_to_b"
    scr_d2 = {}  # direction "b_to_a"
    for i, m in enumerate(models):
        for j, ct in enumerate(cts):
            r1 = [r for r in c if short_model(r["model"]) == m
                  and r["constraint_type"] == ct and r["direction"] == "a_to_b"]
            r2 = [r for r in c if short_model(r["model"]) == m
                  and r["constraint_type"] == ct and r["direction"] == "b_to_a"]
            scr_d1[(i, j)] = compute_scr(r1) if r1 else None
            scr_d2[(i, j)] = compute_scr(r2) if r2 else None

    fig = go.Figure()

    for i in range(n_m):
        for j in range(n_c):
            s1 = scr_d1[(i, j)]
            s2 = scr_d2[(i, j)]
            c1 = _scr_to_color(s1)
            c2 = _scr_to_color(s2)
            m_name = models[i]
            ct_name = cts[j]
            asym = abs((s1 or 0) - (s2 or 0))
            flag = " *" if asym > 0.15 else ""

            # Upper-left triangle (direction 1): corners (j-0.5,i-0.5), (j-0.5,i+0.5), (j+0.5,i+0.5)
            t1_text = f"<b>{m_name} × {ct_name}</b><br>Dir 1: {s1:.0%}{flag}" if s1 is not None else f"<b>{m_name} × {ct_name}</b><br>Dir 1: —"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j - 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i + 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c1,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=t1_text + "<extra></extra>",
                showlegend=False,
            ))

            # Lower-right triangle (direction 2): corners (j-0.5,i-0.5), (j+0.5,i-0.5), (j+0.5,i+0.5)
            t2_text = f"<b>{m_name} × {ct_name}</b><br>Dir 2: {s2:.0%}{flag}" if s2 is not None else f"<b>{m_name} × {ct_name}</b><br>Dir 2: —"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j + 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i - 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c2,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=t2_text + "<extra></extra>",
                showlegend=False,
            ))

            # Diagonal divider line
            fig.add_shape(type="line",
                          x0=j - 0.5, y0=i - 0.5, x1=j + 0.5, y1=i + 0.5,
                          line=dict(color="white", width=1))

            # Text annotations (small, inside triangles)
            if s1 is not None:
                fig.add_annotation(x=j - 0.18, y=i + 0.18,
                                   text=f"{s1:.0%}{flag}",
                                   showarrow=False, font=dict(size=8, color="black"),
                                   xref="x", yref="y")
            if s2 is not None:
                fig.add_annotation(x=j + 0.18, y=i - 0.18,
                                   text=f"{s2:.0%}{flag}",
                                   showarrow=False, font=dict(size=8, color="black"),
                                   xref="x", yref="y")

    # Cell grid lines
    for ii in range(n_m + 1):
        fig.add_shape(type="line", x0=-0.5, y0=ii - 0.5, x1=n_c - 0.5, y1=ii - 0.5,
                      line=dict(color="lightgray", width=0.5))
    for jj in range(n_c + 1):
        fig.add_shape(type="line", x0=jj - 0.5, y0=-0.5, x1=jj - 0.5, y1=n_m - 0.5,
                      line=dict(color="lightgray", width=0.5))

    # Dummy heatmap for colorbar only
    fig.add_trace(go.Heatmap(
        z=[[0, 0.5, 1]], x=[-100, -99, -98], y=[-100],
        colorscale="RdYlGn", zmin=0, zmax=1,
        showscale=True, colorbar=dict(title="SCR", x=1.01),
        opacity=0, hoverinfo="skip",
    ))

    fig.update_layout(
        title="SCR by Constraint Type & Direction (Cond. C)<br>"
              "<sup>Each cell split diagonally — upper-left & lower-right = two counterbalancing directions. * = |diff| > 0.15</sup>",
        xaxis=dict(
            tickvals=list(range(n_c)),
            ticktext=[ct.capitalize() for ct in cts],
            tickangle=-30,
            range=[-0.5, n_c - 0.5],
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickvals=list(range(n_m)),
            ticktext=models,
            range=[-0.5, n_m - 0.5],
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="white",
        showlegend=False,
        height=max(300, 55 * n_m + 150),
        margin=dict(l=150, r=80, t=100, b=100),
    )
    return fig


# ── 6. Heatmap (strength × constraint, default user style) ───────────────

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
    # Only show y-axis labels for left-column subplots (Plotly suppresses them with shared_yaxes)
    for idx in range(n_models):
        col = idx % n_cols + 1
        axis_num = idx + 1
        axis_key = f"yaxis{axis_num if axis_num > 1 else ''}"
        fig.update_layout({axis_key: dict(showticklabels=(col == 1))})
    fig.update_layout(
        title="SCR Heatmap: Strength × Constraint (Cond. C)",
        height=max(900, 280 * n_rows),
        width=1100,
        autosize=False,
        margin=dict(l=160, r=40, t=80, b=60),
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
            if r["condition"] == "C" and not (
                r.get("strength") == "weak" and r.get("user_style") == "with_instruction"
            ):
                continue
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


# ── 12. User Style × Constraint Type (interactive, model dropdown) ────────

def fig_user_style_constraint_heatmap(records: list[dict]) -> go.Figure | None:
    """Interactive heatmap: constraint_type × user_style, with model dropdown.
    Fixed: Condition C, strength=medium."""
    c = [r for r in records if r["condition"] == "C" and r["strength"] == DEFAULT_STRENGTH]
    if not c:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    STYLE_ORDER = ["with_instruction", "helpfulness", "authority", "jailbreak"]
    styles = [s for s in STYLE_ORDER if any(r["user_style"] == s for r in c)]

    x_labels = [s.replace("_", " ").title() for s in styles]
    y_labels = [ct.capitalize() for ct in cts]

    def make_z_and_text(model_filter):
        z, text = [], []
        for ct in cts:
            row_z, row_t = [], []
            for style in styles:
                if model_filter is None:
                    cell = [r for r in c if r["constraint_type"] == ct and r["user_style"] == style]
                else:
                    cell = [r for r in c if short_model(r["model"]) == model_filter
                            and r["constraint_type"] == ct and r["user_style"] == style]
                scr = compute_scr(cell) if cell else None
                row_z.append(scr)
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "—")
            z.append(row_z)
            text.append(row_t)
        return z, text

    all_options = ["All models (averaged)"] + models
    traces = []
    for option in all_options:
        model_filter = None if option == "All models (averaged)" else option
        z, text = make_z_and_text(model_filter)
        traces.append(go.Heatmap(
            z=z, x=x_labels, y=y_labels,
            text=text, texttemplate="%{text}",
            zmin=0, zmax=1, colorscale="RdYlGn",
            colorbar=dict(title="SCR"),
            hovertemplate="Constraint: %{y}<br>Style: %{x}<br>%{text}<extra></extra>",
            visible=False,
        ))
    traces[0].visible = True

    buttons = []
    for i, option in enumerate(all_options):
        visibility = [j == i for j in range(len(all_options))]
        buttons.append(dict(
            label=option,
            method="update",
            args=[{"visible": visibility},
                  {"title": f"User Style × Constraint Type — {option} (Cond. C, strength={DEFAULT_STRENGTH})"}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"User Style × Constraint Type — All models (averaged) (Cond. C, strength={DEFAULT_STRENGTH})",
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.15, yanchor="top",
            pad=dict(r=10, t=10),
        )],
        height=max(400, 60 * len(cts) + 150),
        margin=dict(l=140, r=40, t=120, b=80),
    )
    return fig


# ── 13. User Style × System Strength (interactive, two dropdowns) ─────────

def fig_user_style_strength_heatmap(records: list[dict]) -> go.Figure | None:
    """Interactive heatmap: user_style × strength, with model and constraint dropdowns.
    Condition C only."""
    c = [r for r in records if r["condition"] == "C"]
    if not c:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    STYLE_ORDER = ["with_instruction", "helpfulness", "authority", "jailbreak"]
    styles = [s for s in STYLE_ORDER if any(r["user_style"] == s for r in c)]
    strengths = [s for s in STRENGTH_ORDER if any(r["strength"] == s for r in c)]

    x_labels = [s.capitalize() for s in strengths]
    y_labels = [s.replace("_", " ").title() for s in styles]

    model_options = ["All models"] + models
    ct_options = ["All constraints"] + cts

    def make_z_and_text(model_filter, ct_filter):
        z, text = [], []
        for style in styles:
            row_z, row_t = [], []
            for strength in strengths:
                cell = [r for r in c if r["user_style"] == style and r["strength"] == strength]
                if model_filter is not None:
                    cell = [r for r in cell if short_model(r["model"]) == model_filter]
                if ct_filter is not None:
                    cell = [r for r in cell if r["constraint_type"] == ct_filter]
                scr = compute_scr(cell) if cell else None
                row_z.append(scr)
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "—")
            z.append(row_z)
            text.append(row_t)
        return z, text

    traces = []
    trace_index = {}
    for m_opt in model_options:
        for ct_opt in ct_options:
            model_filter = None if m_opt == "All models" else m_opt
            ct_filter = None if ct_opt == "All constraints" else ct_opt
            z, text = make_z_and_text(model_filter, ct_filter)
            idx = len(traces)
            trace_index[(m_opt, ct_opt)] = idx
            traces.append(go.Heatmap(
                z=z, x=x_labels, y=y_labels,
                text=text, texttemplate="%{text}",
                zmin=0, zmax=1, colorscale="RdYlGn",
                colorbar=dict(title="SCR"),
                hovertemplate="Style: %{y}<br>Strength: %{x}<br>%{text}<extra></extra>",
                visible=False,
            ))
    traces[trace_index[("All models", "All constraints")]].visible = True

    def make_buttons(dim, options, other_default):
        buttons = []
        for opt in options:
            visibility = []
            for m_opt in model_options:
                for ct_opt in ct_options:
                    if dim == "model":
                        vis = (m_opt == opt and ct_opt == other_default)
                    else:
                        vis = (ct_opt == opt and m_opt == other_default)
                    visibility.append(vis)
            buttons.append(dict(
                label=opt, method="update",
                args=[{"visible": visibility}],
            ))
        return buttons

    model_buttons = make_buttons("model", model_options, "All constraints")
    ct_buttons = make_buttons("ct", ct_options, "All models")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="User Style × System Strength (Cond. C)",
        updatemenus=[
            dict(
                buttons=model_buttons,
                direction="down", showactive=True,
                x=0.0, xanchor="left",
                y=1.20, yanchor="top",
                pad=dict(r=10, t=10),
            ),
            dict(
                buttons=ct_buttons,
                direction="down", showactive=True,
                x=0.35, xanchor="left",
                y=1.20, yanchor="top",
                pad=dict(r=10, t=10),
            ),
        ],
        annotations=[
            dict(text="Model:", x=0.0, xref="paper", y=1.27, yref="paper",
                 showarrow=False, font=dict(size=12)),
            dict(text="Constraint:", x=0.35, xref="paper", y=1.27, yref="paper",
                 showarrow=False, font=dict(size=12)),
        ],
        height=max(350, 60 * len(styles) + 200),
        margin=dict(l=140, r=40, t=160, b=80),
    )
    return fig
