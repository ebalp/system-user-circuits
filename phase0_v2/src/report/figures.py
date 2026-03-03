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
    SYSTEM_STYLE_ORDER, LABEL_ORDER, LABEL_COLORS,
    DEFAULT_USER_STYLE, DEFAULT_SYSTEM_STYLE,
    default_conflict_recs, _model_sort_key,
)
from phase0_v2.src.metrics import ModelMetrics

# ── Colour palette ────────────────────────────────────────────────────────
MODEL_PALETTE = [
    "#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c",
    "#1abc9c", "#f39c12", "#8e44ad",
]


def _model_color(idx: int) -> str:
    return MODEL_PALETTE[idx % len(MODEL_PALETTE)]


def _val_to_color(val: float | None, colorscale: str = "RdYlGn",
                  vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Map a value to a hex color using a Plotly colorscale."""
    if val is None:
        return "rgba(210,210,210,0.5)"
    if vmax == vmin:
        normalized = 0.5
    else:
        normalized = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    colors = pc.sample_colorscale(colorscale, [normalized])
    return colors[0]


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


# ── 2. SCR by system style (default user style, Condition C) ─────────────

def fig_scr_by_system_style(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["user_style"] == DEFAULT_USER_STYLE]
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    styles = [s for s in SYSTEM_STYLE_ORDER if any(r["system_style"] == s for r in c)]

    fig = go.Figure()
    for i, model in enumerate(models):
        m_recs = [r for r in c if short_model(r["model"]) == model]
        scrs = [compute_scr([r for r in m_recs if r["system_style"] == s]) for s in styles]
        ns = [len([r for r in m_recs if r["system_style"] == s]) for s in styles]
        fig.add_trace(go.Scatter(
            x=styles, y=scrs, mode="lines+markers",
            name=model, marker_size=10, marker_color=_model_color(i),
            text=[f"n={n}" for n in ns],
            hovertemplate="%{x}: SCR=%{y:.1%} (%{text})<extra>%{fullData.name}</extra>",
        ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                  annotation_text="0.7")
    fig.update_layout(
        title="SCR vs System Style (Condition C)",
        xaxis_title="System Style", yaxis_title="SCR",
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
    """2xN subplot grid: one subplot per experiment pair, showing A->B vs B->A SCR per model."""
    c = default_conflict_recs(records)
    if not c:
        return None

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
            name="A->B", x=models, y=a2b_vals, marker_color="#3498db",
            text=[f"{v:.0%}" if v < 1.0 else "" for v in a2b_vals], textposition="outside",
            showlegend=(pidx == 0),
        ), row=row, col=col)
        fig.add_trace(go.Bar(
            name="B->A", x=models, y=b2a_vals, marker_color="#e67e22",
            text=[f"{v:.0%}" if v < 1.0 else "" for v in b2a_vals], textposition="outside",
            showlegend=(pidx == 0),
        ), row=row, col=col)

        for i, m in enumerate(models):
            d = abs(a2b_vals[i] - b2a_vals[i])
            color = "red" if d > 0.15 else "black"
            fig.add_annotation(
                x=m, y=max(a2b_vals[i], b2a_vals[i]) + 0.18,
                text=f"d={d:.2f}", showarrow=False,
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


# ── 5. Constraint type x direction triangle-split matrix ─────────────────

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

    scr_d1 = {}
    scr_d2 = {}
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

            t1_text = f"<b>{m_name} x {ct_name}</b><br>Dir 1: {s1:.0%}{flag}" if s1 is not None else f"<b>{m_name} x {ct_name}</b><br>Dir 1: --"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j - 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i + 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c1,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=t1_text + "<extra></extra>",
                showlegend=False,
            ))

            t2_text = f"<b>{m_name} x {ct_name}</b><br>Dir 2: {s2:.0%}{flag}" if s2 is not None else f"<b>{m_name} x {ct_name}</b><br>Dir 2: --"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j + 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i - 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c2,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=t2_text + "<extra></extra>",
                showlegend=False,
            ))

            fig.add_shape(type="line",
                          x0=j - 0.5, y0=i - 0.5, x1=j + 0.5, y1=i + 0.5,
                          line=dict(color="white", width=1))

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

    for ii in range(n_m + 1):
        fig.add_shape(type="line", x0=-0.5, y0=ii - 0.5, x1=n_c - 0.5, y1=ii - 0.5,
                      line=dict(color="lightgray", width=0.5))
    for jj in range(n_c + 1):
        fig.add_shape(type="line", x0=jj - 0.5, y0=-0.5, x1=jj - 0.5, y1=n_m - 0.5,
                      line=dict(color="lightgray", width=0.5))

    fig.add_trace(go.Heatmap(
        z=[[0, 0.5, 1]], x=[-100, -99, -98], y=[-100],
        colorscale="RdYlGn", zmin=0, zmax=1,
        showscale=True, colorbar=dict(title="SCR", x=1.01),
        opacity=0, hoverinfo="skip",
    ))

    fig.update_layout(
        title="SCR by Constraint Type & Direction (Cond. C)<br>"
              "<sup>Each cell split diagonally -- upper-left & lower-right = two counterbalancing directions. * = |diff| > 0.15</sup>",
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


# ── 6. Heatmap (system_style x constraint, default user style) ───────────

def fig_heatmap(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["user_style"] == DEFAULT_USER_STYLE]
    if not c:
        return None
    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    styles = [s for s in SYSTEM_STYLE_ORDER if any(r["system_style"] == s for r in c)]

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
            for s in styles:
                cell = [r for r in m_recs if r["constraint_type"] == ct and r["system_style"] == s]
                scr = compute_scr(cell) if cell else None
                row_z.append(scr)
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "--")
            z.append(row_z)
            text.append(row_t)
        fig.add_trace(go.Heatmap(
            z=z, x=styles, y=[ct.capitalize() for ct in cts],
            text=text, texttemplate="%{text}", zmin=0, zmax=1,
            colorscale="RdYlGn", showscale=(idx == n_models - 1),
        ), row=row, col=col)
    for idx in range(n_models):
        col = idx % n_cols + 1
        axis_num = idx + 1
        axis_key = f"yaxis{axis_num if axis_num > 1 else ''}"
        fig.update_layout({axis_key: dict(showticklabels=(col == 1))})
    fig.update_layout(
        title="SCR Heatmap: System Style x Constraint (Cond. C)",
        height=max(900, 280 * n_rows),
        width=1100,
        autosize=False,
        margin=dict(l=160, r=40, t=80, b=60),
    )
    return fig


# ── 7. Label distribution by condition (per model) ───────────────────────

def fig_by_condition(records: list[dict]) -> go.Figure:
    models = sorted(set(short_model(r["model"]) for r in records), key=_model_sort_key)

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

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
            # For C, filter to bare+with_instruction to match D format for fair comparison
            if r["condition"] == "C" and not (
                r.get("system_style") == DEFAULT_SYSTEM_STYLE
                and r.get("user_style") == DEFAULT_USER_STYLE
            ):
                continue
            counts[f"Cond. {r['condition']}"][r["label"]] += 1
        cats = [f"Cond. {c}" for c in conds]
        for label in present_labels:
            vals = []
            raw_counts = []
            for cat in cats:
                total = sum(counts[cat].values()) or 1
                count = counts[cat].get(label, 0)
                vals.append(count / total * 100)
                raw_counts.append(count)
            fig.add_trace(go.Bar(
                name=label.replace("_", " ").title(), x=cats, y=vals,
                marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
                showlegend=(idx == 0),
                customdata=raw_counts,
                hovertemplate="%{y:.1f}% (%{customdata} responses)<extra></extra>",
            ), row=row, col=col)
    fig.update_layout(
        barmode="stack",
        title="Label Distribution by Condition",
        yaxis_title="% of responses", height=200 * n_rows,
    )
    return fig


# ── 8. User template effect (with_instruction vs jailbreak) ──────────────

def fig_user_template(records: list[dict]) -> go.Figure | None:
    c = [r for r in records
         if r["condition"] == "C" and r["system_style"] == DEFAULT_SYSTEM_STYLE]
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


# ── 9. Cross-model SCR comparison (default filters) ──────────────────────

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


# ── 10. Task effect (default filters) ─────────────────────────────────────

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


# ── 11. Confidence distribution ──────────────────────────────────────────

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


# ── 12. Jailbreak vs Default by Constraint Type ───────────────────────────

def fig_jailbreak_vs_default(records: list[dict]) -> go.Figure | None:
    """Compare SCR under jailbreak user template vs default (with_instruction),
    both at bare system style, broken down by constraint type."""
    c = [r for r in records
         if r["condition"] == "C" and r["system_style"] == DEFAULT_SYSTEM_STYLE]
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
        title="Jailbreak vs Default User Template -- SCR by Constraint Type (Cond. C)",
        yaxis_title="SCR",
        height=480,
    )
    return fig


# ── 13. User Style x Constraint Type (interactive, model dropdown) ────────

def fig_user_style_constraint_heatmap(records: list[dict]) -> go.Figure | None:
    """Interactive heatmap: constraint_type x user_style, with model dropdown.
    Fixed: Condition C, system_style=bare."""
    c = [r for r in records if r["condition"] == "C" and r["system_style"] == DEFAULT_SYSTEM_STYLE]
    if not c:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    STYLE_ORDER = ["with_instruction", "helpfulness", "authority", "jailbreak", "pleading"]
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
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "--")
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
                  {"title": f"User Style x Constraint Type -- {option} (Cond. C, system_style={DEFAULT_SYSTEM_STYLE})"}],
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"User Style x Constraint Type -- All models (averaged) (Cond. C, system_style={DEFAULT_SYSTEM_STYLE})",
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


# ── 14. User Style x System Style (interactive, two dropdowns) ─────────

def fig_user_style_system_style_heatmap(records: list[dict]) -> go.Figure | None:
    """Interactive heatmap: user_style x system_style, with model and constraint dropdowns.
    Condition C only."""
    c = [r for r in records if r["condition"] == "C"]
    if not c:
        return None

    models = sorted(set(short_model(r["model"]) for r in c), key=_model_sort_key)
    cts = sorted(set(r["constraint_type"] for r in c))
    STYLE_ORDER = ["with_instruction", "helpfulness", "authority", "jailbreak", "pleading"]
    styles = [s for s in STYLE_ORDER if any(r["user_style"] == s for r in c)]
    system_styles = [s for s in SYSTEM_STYLE_ORDER if any(r["system_style"] == s for r in c)]

    x_labels = [s.capitalize() for s in system_styles]
    y_labels = [s.replace("_", " ").title() for s in styles]

    model_options = ["All models"] + models
    ct_options = ["All constraints"] + cts

    def make_z_and_text(model_filter, ct_filter):
        z, text = [], []
        for style in styles:
            row_z, row_t = [], []
            for sys_style in system_styles:
                cell = [r for r in c if r["user_style"] == style and r["system_style"] == sys_style]
                if model_filter is not None:
                    cell = [r for r in cell if short_model(r["model"]) == model_filter]
                if ct_filter is not None:
                    cell = [r for r in cell if r["constraint_type"] == ct_filter]
                scr = compute_scr(cell) if cell else None
                row_z.append(scr)
                row_t.append(f"{scr:.0%} (n={len(cell)})" if scr is not None else "--")
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
                hovertemplate="Style: %{y}<br>System Style: %{x}<br>%{text}<extra></extra>",
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
        title="User Style x System Style (Cond. C)",
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


# ── Triangle-split heatmap helpers (from ModelMetrics) ────────────────────

def make_triangle_heatmap(
    title: str,
    models: list[str],
    conflicts: list[str],
    data: dict[tuple[str, str], tuple[float | None, float | None]],
    colorscale: str = "RdYlGn",
    colorbar_title: str = "Value",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> go.Figure:
    """Build a triangle-split heatmap.

    Args:
        title: Figure title.
        models: List of model names (x-axis / columns).
        conflicts: List of conflict names (y-axis / rows).
        data: Dict mapping (model_name, conflict_name) to (val_a, val_b).
        colorscale: Plotly colorscale name.
        colorbar_title: Label for the colorbar.
        vmin: Minimum value for color mapping.
        vmax: Maximum value for color mapping.

    Returns:
        A Plotly Figure object.
    """
    n_models = len(models)
    n_conflicts = len(conflicts)
    fig = go.Figure()

    for col_idx, model in enumerate(models):
        for row_idx, conflict in enumerate(conflicts):
            val_a, val_b = data.get((model, conflict), (None, None))
            color_a = _val_to_color(val_a, colorscale, vmin, vmax)
            color_b = _val_to_color(val_b, colorscale, vmin, vmax)

            asym = abs((val_a or 0) - (val_b or 0))
            flag = " *" if asym > 0.15 else ""

            x, y = col_idx, row_idx

            hover_a = (
                f"<b>{model} x {conflict}</b><br>Dir a: {val_a:.0%}{flag}"
                if val_a is not None
                else f"<b>{model} x {conflict}</b><br>Dir a: --"
            )
            fig.add_trace(go.Scatter(
                x=[x - 0.5, x - 0.5, x + 0.5, x - 0.5],
                y=[y - 0.5, y + 0.5, y + 0.5, y - 0.5],
                fill="toself", fillcolor=color_a,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=hover_a + "<extra></extra>",
                showlegend=False,
            ))

            hover_b = (
                f"<b>{model} x {conflict}</b><br>Dir b: {val_b:.0%}{flag}"
                if val_b is not None
                else f"<b>{model} x {conflict}</b><br>Dir b: --"
            )
            fig.add_trace(go.Scatter(
                x=[x - 0.5, x + 0.5, x + 0.5, x - 0.5],
                y=[y - 0.5, y - 0.5, y + 0.5, y - 0.5],
                fill="toself", fillcolor=color_b,
                line=dict(color="white", width=1),
                mode="lines",
                hovertemplate=hover_b + "<extra></extra>",
                showlegend=False,
            ))

            fig.add_shape(
                type="line",
                x0=x - 0.5, y0=y - 0.5, x1=x + 0.5, y1=y + 0.5,
                line=dict(color="white", width=1),
            )

            if val_a is not None:
                fmt = f"{val_a:.0%}" if vmin >= 0 else f"{val_a:+.0%}"
                fig.add_annotation(
                    x=x - 0.18, y=y + 0.18,
                    text=f"{fmt}{flag}",
                    showarrow=False, font=dict(size=8, color="black"),
                )
            if val_b is not None:
                fmt = f"{val_b:.0%}" if vmin >= 0 else f"{val_b:+.0%}"
                fig.add_annotation(
                    x=x + 0.18, y=y - 0.18,
                    text=f"{fmt}{flag}",
                    showarrow=False, font=dict(size=8, color="black"),
                )

    for row_idx in range(n_conflicts + 1):
        fig.add_shape(
            type="line",
            x0=-0.5, y0=row_idx - 0.5,
            x1=n_models - 0.5, y1=row_idx - 0.5,
            line=dict(color="lightgray", width=0.5),
        )
    for col_idx in range(n_models + 1):
        fig.add_shape(
            type="line",
            x0=col_idx - 0.5, y0=-0.5,
            x1=col_idx - 0.5, y1=n_conflicts - 0.5,
            line=dict(color="lightgray", width=0.5),
        )

    fig.add_trace(go.Heatmap(
        z=[[vmin, (vmin + vmax) / 2, vmax]],
        x=[-100, -99, -98], y=[-100],
        colorscale=colorscale, zmin=vmin, zmax=vmax,
        showscale=True, colorbar=dict(title=colorbar_title, x=1.01),
        opacity=0, hoverinfo="skip",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=list(range(n_models)),
            ticktext=models,
            tickangle=-30,
            range=[-0.5, n_models - 0.5],
            showgrid=False, zeroline=False,
            side="bottom",
        ),
        yaxis=dict(
            tickvals=list(range(n_conflicts)),
            ticktext=[c.replace("_", " ").capitalize() for c in conflicts],
            range=[-0.5, n_conflicts - 0.5],
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="white",
        showlegend=False,
        height=max(400, 40 * n_conflicts + 150),
        margin=dict(l=180, r=80, t=100, b=100),
    )

    return fig


def _extract_by_constraint(
    all_metrics: dict[str, ModelMetrics],
    attr: str,
) -> tuple[list[str], list[str], dict[tuple[str, str], tuple[float | None, float | None]]]:
    """Extract per-constraint directional data from all_metrics."""
    models = sorted(all_metrics.keys())
    conflict_set: set[str] = set()
    for m in all_metrics.values():
        conflict_set.update(m.by_constraint.keys())
    conflicts = sorted(conflict_set)

    data: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for model in models:
        m = all_metrics[model]
        for ct in conflicts:
            if ct in m.by_constraint:
                dm = getattr(m.by_constraint[ct], attr)
                val_a = dm.a_to_b.value if dm.a_to_b.n > 0 else None
                val_b = dm.b_to_a.value if dm.b_to_a.n > 0 else None
                data[(model, ct)] = (val_a, val_b)
            else:
                data[(model, ct)] = (None, None)

    return models, conflicts, data


def fig_sbr_by_conflict(all_metrics: dict[str, ModelMetrics]) -> go.Figure:
    """SBR triangle-split heatmap by conflict and model."""
    models, conflicts, data = _extract_by_constraint(all_metrics, "sbr")
    return make_triangle_heatmap(
        title=(
            "SBR by Constraint Type & Direction (Cond. A)<br>"
            "<sup>Upper-left = a_to_b, lower-right = b_to_a. * = |diff| > 0.15</sup>"
        ),
        models=models, conflicts=conflicts, data=data,
        colorscale="RdYlGn", colorbar_title="SBR",
    )


def fig_ucr_by_conflict(all_metrics: dict[str, ModelMetrics]) -> go.Figure:
    """UCR triangle-split heatmap by conflict and model."""
    models, conflicts, data = _extract_by_constraint(all_metrics, "ucr")
    return make_triangle_heatmap(
        title=(
            "UCR by Constraint Type & Direction (Cond. B)<br>"
            "<sup>Upper-left = a_to_b, lower-right = b_to_a. * = |diff| > 0.15</sup>"
        ),
        models=models, conflicts=conflicts, data=data,
        colorscale="RdYlGn", colorbar_title="UCR",
    )


def fig_system_authority_delta(all_metrics: dict[str, ModelMetrics]) -> go.Figure:
    """System Authority Delta triangle-split heatmap by conflict and model."""
    models, conflicts, data = _extract_by_constraint(all_metrics, "system_authority_delta")
    return make_triangle_heatmap(
        title=(
            "System Authority Delta by Constraint Type & Direction<br>"
            "<sup>Delta = C_bare_SCR - D_first_rate. Red = system adds authority, "
            "Blue = no special authority. * = |diff| > 0.15</sup>"
        ),
        models=models, conflicts=conflicts, data=data,
        colorscale="RdBu_r", colorbar_title="Delta (system authority)",
        vmin=-1.0, vmax=1.0,
    )
