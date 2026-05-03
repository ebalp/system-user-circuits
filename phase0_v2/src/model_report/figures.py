"""Plotly figure builders for the single-model behavioral analysis report.

Each function takes records: list[dict] (single model's records) and returns
a go.Figure or None when data is insufficient.
"""

from __future__ import annotations
import hashlib
from collections import Counter, defaultdict

import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

from phase0_v2.src.report.data import (
    compute_scr, compute_ucr, compute_cr, wilson_ci,
    SYSTEM_STYLE_ORDER, LABEL_ORDER, LABEL_COLORS,
    DEFAULT_USER_STYLE, DEFAULT_SYSTEM_STYLE,
    default_conflict_recs,
)
from phase0_v2.src.model_report.tables import PREPROCESSING_ICONS
from phase0_v2.src.metrics import MetricsCalculator, ModelMetrics
from phase0_v2.src.model_report.ordering import ordered


# ── Ordering helpers ──────────────────────────────────────────────────────


def _filter_cond_c(records: list[dict],
                   cond_c_filter: dict[str, str] | None = None) -> list[dict]:
    """Filter to condition C records, optionally applying style filters."""
    c = [r for r in records if r["condition"] == "C"]
    if cond_c_filter:
        for field, value in cond_c_filter.items():
            c = [r for r in c if r.get(field) == value]
    return c



# ── Colour helper ────────────────────────────────────────────────────────

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


# ── 1. Label distribution by condition ───────────────────────────────────

def fig_label_distribution(records: list[dict]) -> go.Figure | None:
    """Stacked bar chart showing label proportions for conditions A, B, C, D.

    For condition C, filters to default styles for D-comparability.
    """
    conditions = ["A", "B", "C", "D"]
    counts_by_cond: dict[str, Counter] = {}

    for cond in conditions:
        if cond == "C":
            cond_recs = [
                r for r in records
                if r["condition"] == cond
                and r.get("user_style") == DEFAULT_USER_STYLE
                and r.get("system_style") == DEFAULT_SYSTEM_STYLE
            ]
        else:
            cond_recs = [r for r in records if r["condition"] == cond]
        counts_by_cond[cond] = Counter(r["label"] for r in cond_recs)

    if not any(sum(c.values()) > 0 for c in counts_by_cond.values()):
        return None

    fig = go.Figure()
    for label in LABEL_ORDER:
        vals = []
        texts = []
        for cond in conditions:
            total = sum(counts_by_cond[cond].values()) or 1
            count = counts_by_cond[cond].get(label, 0)
            vals.append(count / total * 100)
            texts.append(str(count))
        fig.add_trace(go.Bar(
            name=label.replace("_", " ").title(),
            x=conditions, y=vals,
            marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
            text=texts, textposition="inside",
            hovertemplate="Condition %{x}: %{y:.1f}% (%{text})<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title="Label Distribution by Condition",
        xaxis_title="Condition",
        yaxis_title="% of responses", yaxis_range=[0, 105],
        legend_title="Label",
        height=400,
    )
    return fig


# ── 2. SCR by system style ──────────────────────────────────────────────

def fig_scr_by_system_style(records: list[dict],
                            cond_c_filter: dict[str, str] | None = None,
                            system_style_order: list[str] | None = None) -> go.Figure | None:
    """Line chart of SCR across system styles (Condition C)."""
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    actual_styles = list({r["system_style"] for r in c})
    styles = ordered(actual_styles, system_style_order) if system_style_order else sorted(actual_styles)
    scrs = []
    ci_lows = []
    ci_highs = []
    ns = []
    for s in styles:
        s_recs = [r for r in c if r["system_style"] == s]
        scr = compute_scr(s_recs)
        n = len(s_recs)
        n_sys = sum(1 for r in s_recs if r["label"] == "followed_system")
        lo, hi = wilson_ci(n_sys, n)
        scrs.append(scr)
        ci_lows.append(scr - lo)
        ci_highs.append(hi - scr)
        ns.append(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=styles, y=scrs, mode="lines+markers+text",
        marker=dict(size=10, color="#3498db"),
        line=dict(color="#3498db", width=2),
        error_y=dict(type="data", symmetric=False,
                     array=ci_highs, arrayminus=ci_lows,
                     color="rgba(52,152,219,0.4)"),
        text=[f"{v:.2f}" for v in scrs],
        textposition="top center",
        hovertemplate="%{x}: SCR=%{y:.3f} (n=%{customdata})<extra></extra>",
        customdata=ns,
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                  annotation_text="0.7 threshold")
    fig.update_layout(
        title="SCR vs System Style (Condition C)",
        xaxis_title="System Style", yaxis_title="SCR",
        yaxis_range=[0, 1.1],
    )
    return fig


# ── 3. SCR by user style ────────────────────────────────────────────────

def fig_scr_by_user_style(records: list[dict],
                          cond_c_filter: dict[str, str] | None = None,
                          user_style_order: list[str] | None = None) -> go.Figure | None:
    """Bar chart of SCR across user styles (Condition C)."""
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    actual_styles = list({r.get("user_style", "unknown") for r in c})
    user_styles = ordered(actual_styles, user_style_order) if user_style_order else sorted(actual_styles)
    scrs = []
    ci_lows = []
    ci_highs = []
    for us in user_styles:
        us_recs = [r for r in c if r.get("user_style") == us]
        scr = compute_scr(us_recs)
        n = len(us_recs)
        n_sys = sum(1 for r in us_recs if r["label"] == "followed_system")
        lo, hi = wilson_ci(n_sys, n)
        scrs.append(scr)
        ci_lows.append(scr - lo)
        ci_highs.append(hi - scr)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=user_styles, y=scrs,
        marker_color="#3498db",
        error_y=dict(type="data", symmetric=False,
                     array=ci_highs, arrayminus=ci_lows),
        text=[f"{v:.2f}" for v in scrs], textposition="outside",
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                  annotation_text="0.7 threshold")
    fig.update_layout(
        title="SCR by User Style (Condition C, bare system)",
        xaxis_title="User Style", yaxis_title="SCR",
        yaxis_range=[0, 1.15],
    )
    return fig


# ── 4. Asymmetry dumbbell chart ─────────────────────────────────────────

def fig_asymmetry_dumbbell(records: list[dict],
                           cond_c_filter: dict[str, str] | None = None,
                           conflict_order: list[str] | None = None) -> go.Figure | None:
    """Dumbbell chart with strip plots showing directional SCR asymmetry.

    For each conflict, shows:
    - 25 jittered dots per direction (one per system_style × user_style combo)
    - A larger marker for each direction's mean SCR
    - A connecting line between the two means (red if asymmetry > 0.15)
    """
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    # Use canonical ordering (reversed so highest SCR at top of y-axis)
    actual_cids = list({r.get("conflict_id", r["constraint_type"]) for r in c})
    ordered_cids = list(reversed(ordered(actual_cids, conflict_order))) if conflict_order else sorted(actual_cids)
    if not ordered_cids:
        return None

    # Compute per-style-combo SCR for each conflict × direction
    data: dict[str, dict] = {}
    for cid in ordered_cids:
        c_recs = [r for r in c if r.get("conflict_id", r["constraint_type"]) == cid]
        entry: dict = {}
        for direction in ("a_to_b", "b_to_a"):
            d_recs = [r for r in c_recs if r["direction"] == direction]
            # Group by (system_style, user_style)
            combos: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for r in d_recs:
                key = (r.get("system_style", "?"), r.get("user_style", "?"))
                combos[key].append(r)
            combo_scrs = []
            combo_labels = []
            for (ss, us), recs in sorted(combos.items()):
                combo_scrs.append(compute_scr(recs))
                combo_labels.append(f"{ss} / {us}")
            mean_scr = compute_scr(d_recs)
            entry[direction] = {
                "mean": mean_scr,
                "combo_scrs": combo_scrs,
                "combo_labels": combo_labels,
            }
        entry["asymmetry"] = abs(entry["a_to_b"]["mean"] - entry["b_to_a"]["mean"])
        data[cid] = entry

    cids = ordered_cids
    fig = go.Figure()

    # Numeric y-axis with tick labels (needed for jitter)
    cid_to_idx = {cid: i for i, cid in enumerate(cids)}

    # Connecting lines between means (drawn first, behind everything)
    for cid in cids:
        e = data[cid]
        color = "red" if e["asymmetry"] > 0.15 else "gray"
        yi = cid_to_idx[cid]
        fig.add_trace(go.Scatter(
            x=[e["a_to_b"]["mean"], e["b_to_a"]["mean"]], y=[yi, yi],
            mode="lines", line=dict(color=color, width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Strip dots — individual style combos (small, semi-transparent, jittered)
    for direction, color in [("a_to_b", "#3498db"), ("b_to_a", "#e67e22")]:
        xs, ys, hovers = [], [], []
        for cid in cids:
            combo_scrs = data[cid][direction]["combo_scrs"]
            combo_labels = data[cid][direction]["combo_labels"]
            idx = cid_to_idx[cid]
            for scr_val, label in zip(combo_scrs, combo_labels):
                xs.append(scr_val)
                # Deterministic vertical jitter from hash of label
                h = int(hashlib.md5(label.encode()).hexdigest()[:8], 16)
                jitter = ((h % 1000) / 1000 - 0.5) * 0.3  # ±0.15 category units
                ys.append(idx + jitter)
                hovers.append(f"{cid}<br>{label}<br>SCR={scr_val:.3f}")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=5, color=color, opacity=0.35),
            name=f"{direction} (combos)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
            legendgroup=direction,
        ))

    # Mean markers (larger, solid)
    for direction, color in [("a_to_b", "#3498db"), ("b_to_a", "#e67e22")]:
        mean_xs = [data[cid][direction]["mean"] for cid in cids]
        mean_ys = [cid_to_idx[cid] for cid in cids]
        fig.add_trace(go.Scatter(
            x=mean_xs, y=mean_ys, mode="markers",
            marker=dict(size=11, color=color,
                        line=dict(width=1.5, color="white")),
            name=f"{direction} (mean)",
            hovertemplate="%{customdata}: mean SCR=%{x:.3f}<extra>" + direction + "</extra>",
            customdata=cids,
            legendgroup=direction,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    n_conflicts = len(cids)
    fig.update_layout(
        title="Direction Asymmetry (Condition C, all style combinations)",
        xaxis_title="SCR",
        xaxis_range=[-0.05, 1.05],
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_conflicts)),
            ticktext=cids,
        ),
        height=max(400, 30 * n_conflicts + 100),
        margin=dict(l=200),
    )
    return fig


# ── 5. System style x constraint heatmap ────────────────────────────────

def _triangle_heatmap(
    records: list[dict],
    style_field: str,
    style_label: str,
    title: str,
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
    style_order: list[str] | None = None,
) -> go.Figure | None:
    """Triangle-split heatmap: each cell divided diagonally into a_to_b (upper-left)
    and b_to_a (lower-right). Rows = conflicts, cols = styles."""
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    actual_cids = list({r.get("conflict_id", r["constraint_type"]) for r in c})
    conflict_ids = list(reversed(
        ordered(actual_cids, conflict_order) if conflict_order else sorted(actual_cids)
    ))
    actual_styles = list({r.get(style_field, "unknown") for r in c})
    styles = ordered(actual_styles, style_order) if style_order else sorted(actual_styles)

    n_rows, n_cols = len(conflict_ids), len(styles)

    # Precompute SCR per (conflict, style, direction)
    scr_a2b: dict[tuple[int, int], float | None] = {}
    scr_b2a: dict[tuple[int, int], float | None] = {}
    for i, cid in enumerate(conflict_ids):
        for j, s in enumerate(styles):
            r_a2b = [r for r in c
                     if r.get("conflict_id", r["constraint_type"]) == cid
                     and r.get(style_field) == s and r["direction"] == "a_to_b"]
            r_b2a = [r for r in c
                     if r.get("conflict_id", r["constraint_type"]) == cid
                     and r.get(style_field) == s and r["direction"] == "b_to_a"]
            scr_a2b[(i, j)] = compute_scr(r_a2b) if r_a2b else None
            scr_b2a[(i, j)] = compute_scr(r_b2a) if r_b2a else None

    fig = go.Figure()

    for i in range(n_rows):
        for j in range(n_cols):
            s1 = scr_a2b[(i, j)]
            s2 = scr_b2a[(i, j)]
            c1 = _val_to_color(s1, "RdYlGn")
            c2 = _val_to_color(s2, "RdYlGn")
            cid = conflict_ids[i]
            style = styles[j]
            asym = abs((s1 or 0) - (s2 or 0))
            flag = " *" if asym > 0.15 else ""

            # Upper-left triangle: a_to_b
            t1 = f"<b>{cid} × {style}</b><br>a→b: {s1:.0%}{flag}" if s1 is not None else f"<b>{cid} × {style}</b><br>a→b: --"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j - 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i + 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c1,
                line=dict(color="white", width=0.5),
                mode="lines",
                hovertemplate=t1 + "<extra></extra>",
                showlegend=False,
            ))

            # Lower-right triangle: b_to_a
            t2 = f"<b>{cid} × {style}</b><br>b→a: {s2:.0%}{flag}" if s2 is not None else f"<b>{cid} × {style}</b><br>b→a: --"
            fig.add_trace(go.Scatter(
                x=[j - 0.5, j + 0.5, j + 0.5, j - 0.5],
                y=[i - 0.5, i - 0.5, i + 0.5, i - 0.5],
                fill="toself", fillcolor=c2,
                line=dict(color="white", width=0.5),
                mode="lines",
                hovertemplate=t2 + "<extra></extra>",
                showlegend=False,
            ))

            # Diagonal line
            fig.add_shape(type="line",
                          x0=j - 0.5, y0=i - 0.5, x1=j + 0.5, y1=i + 0.5,
                          line=dict(color="white", width=0.5))

            # Text annotations
            if s1 is not None:
                fig.add_annotation(x=j - 0.18, y=i + 0.18,
                                   text=f"{s1:.0%}{flag}",
                                   showarrow=False, font=dict(size=7, color="black"))
            if s2 is not None:
                fig.add_annotation(x=j + 0.18, y=i - 0.18,
                                   text=f"{s2:.0%}{flag}",
                                   showarrow=False, font=dict(size=7, color="black"))

    # Grid lines
    for ii in range(n_rows + 1):
        fig.add_shape(type="line", x0=-0.5, y0=ii - 0.5, x1=n_cols - 0.5, y1=ii - 0.5,
                      line=dict(color="lightgray", width=0.5))
    for jj in range(n_cols + 1):
        fig.add_shape(type="line", x0=jj - 0.5, y0=-0.5, x1=jj - 0.5, y1=n_rows - 0.5,
                      line=dict(color="lightgray", width=0.5))

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=list(range(n_cols)), ticktext=styles,
            title=style_label, side="bottom",
        ),
        yaxis=dict(
            tickvals=list(range(n_rows)), ticktext=conflict_ids,
            autorange=True,
        ),
        height=max(500, 30 * n_rows + 150),
        margin=dict(l=200),
        plot_bgcolor="white",
    )
    return fig


def fig_system_style_constraint_heatmap(records: list[dict],
                                        cond_c_filter: dict[str, str] | None = None,
                                        conflict_order: list[str] | None = None,
                                        system_style_order: list[str] | None = None) -> go.Figure | None:
    """Triangle heatmap of SCR by conflict and system style, split by direction."""
    return _triangle_heatmap(
        records, "system_style", "System Style",
        "SCR by Conflict × System Style (a→b ◸ / b→a ◿)",
        cond_c_filter=cond_c_filter, conflict_order=conflict_order,
        style_order=system_style_order,
    )


# ── 6. User style x constraint heatmap ──────────────────────────────────

def fig_user_style_constraint_heatmap(records: list[dict],
                                      cond_c_filter: dict[str, str] | None = None,
                                      conflict_order: list[str] | None = None,
                                      user_style_order: list[str] | None = None) -> go.Figure | None:
    """Triangle heatmap of SCR by conflict and user style, split by direction."""
    return _triangle_heatmap(
        records, "user_style", "User Style",
        "SCR by Conflict × User Style (a→b ◸ / b→a ◿)",
        cond_c_filter=cond_c_filter, conflict_order=conflict_order,
        style_order=user_style_order,
    )


# ── 7. Baselines (SBR and UCR) per conflict ─────────────────────────────

def fig_baselines(records: list[dict],
                   cond_c_filter: dict[str, str] | None = None,
                   conflict_order: list[str] | None = None) -> go.Figure | None:
    """Paired horizontal bar chart showing SBR (cond A) and UCR baseline (cond B)."""
    cond_a = [r for r in records if r["condition"] == "A"]
    cond_b = [r for r in records if r["condition"] == "B"]
    if not cond_a and not cond_b:
        return None

    # Collect conflict IDs from all conditions that have baselines
    c = _filter_cond_c(records, cond_c_filter)
    actual_cids = list({r.get("conflict_id", r["constraint_type"]) for r in c})
    all_cids = list(reversed(ordered(actual_cids, conflict_order))) if conflict_order else sorted(actual_cids)
    if not all_cids:
        return None

    sbrs = []
    ucrs = []
    flagged = []
    for cid in all_cids:
        a_recs = [r for r in cond_a if r.get("conflict_id", r["constraint_type"]) == cid]
        b_recs = [r for r in cond_b if r.get("conflict_id", r["constraint_type"]) == cid]
        sbr = compute_scr(a_recs)  # P(followed_system | A)
        ucr = compute_ucr(b_recs)  # P(followed_user | B)
        sbrs.append(sbr)
        ucrs.append(ucr)
        flagged.append(sbr < 0.95 or ucr < 0.95)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=all_cids, x=sbrs, orientation="h",
        name="SBR (Cond A)", marker_color="#2ecc71",
        text=[f"{v:.2f}" for v in sbrs], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=all_cids, x=ucrs, orientation="h",
        name="UCR baseline (Cond B)", marker_color="#3498db",
        text=[f"{v:.2f}" for v in ucrs], textposition="outside",
    ))

    # Flag conflicts where either < 0.95
    for i, (cid, is_flagged) in enumerate(zip(all_cids, flagged)):
        if is_flagged:
            fig.add_annotation(
                x=1.02, y=cid,
                text="!",
                showarrow=False,
                font=dict(color="red", size=14, family="Arial Black"),
            )

    fig.add_vline(x=0.95, line_dash="dash", line_color="red", line_width=1,
                  annotation_text="0.95")

    n_conflicts = len(all_cids)
    fig.update_layout(
        barmode="group",
        title="Baseline Rates per Conflict (SBR = Cond A, UCR = Cond B)",
        xaxis_title="Rate", xaxis_range=[0, 1.1],
        height=max(400, 30 * n_conflicts + 150),
        margin=dict(l=200),
    )
    return fig


# ── 8. System authority delta ────────────────────────────────────────────

def fig_system_authority_delta(records: list[dict],
                               cond_c_filter: dict[str, str] | None = None,
                               conflict_order: list[str] | None = None) -> go.Figure | None:
    """Grouped horizontal bar chart: delta = C_bare_SCR - D_first_rate per conflict.

    Shows two bars per conflict (a_to_b and b_to_a) to reveal direction-dependent
    authority effects. Always uses bare/with_instruction filter to match
    Condition D's fixed styles.
    """
    c_bare = [
        r for r in records
        if r["condition"] == "C"
        and r.get("system_style") == DEFAULT_SYSTEM_STYLE
        and r.get("user_style") == DEFAULT_USER_STYLE
    ]
    cond_d = [r for r in records if r["condition"] == "D"]
    if not c_bare and not cond_d:
        return None

    c = _filter_cond_c(records, cond_c_filter)
    actual_cids = list({r.get("conflict_id", r["constraint_type"]) for r in c})
    ordered_cids = list(reversed(ordered(actual_cids, conflict_order))) if conflict_order else sorted(actual_cids)
    if not ordered_cids:
        return None

    # Compute delta per conflict × direction
    data: dict[str, dict[str, float]] = {}
    for cid in ordered_cids:
        data[cid] = {}
        for direction in ("a_to_b", "b_to_a"):
            c_recs = [r for r in c_bare
                      if r.get("conflict_id", r["constraint_type"]) == cid
                      and r["direction"] == direction]
            d_recs = [r for r in cond_d
                      if r.get("conflict_id", r["constraint_type"]) == cid
                      and r["direction"] == direction]
            c_scr = compute_scr(c_recs)
            d_first = compute_scr(d_recs)
            data[cid][direction] = c_scr - d_first

    cids = ordered_cids
    all_deltas = [data[cid][d] for cid in cids for d in ("a_to_b", "b_to_a")]
    max_abs = max(abs(d) for d in all_deltas) if all_deltas else 1.0

    fig = go.Figure()
    for direction, color in [("b_to_a", "#e67e22"), ("a_to_b", "#3498db")]:
        deltas = [data[cid][direction] for cid in cids]
        fig.add_trace(go.Bar(
            y=cids, x=deltas, orientation="h",
            name=direction,
            marker_color=color,
            text=[f"{v:+.2f}" for v in deltas], textposition="outside",
            hovertemplate="%{y}: delta=%{x:+.3f}<extra>" + direction + "</extra>",
        ))

    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)

    n_conflicts = len(cids)
    fig.update_layout(
        title="System Authority Delta (C_bare_SCR - D_first_rate)",
        xaxis_title="Delta",
        barmode="group",
        height=max(400, 40 * n_conflicts + 150),
        margin=dict(l=200),
    )
    return fig


# ── 9. Task effect ──────────────────────────────────────────────────────

def fig_task_effect(records: list[dict],
                    cond_c_filter: dict[str, str] | None = None) -> go.Figure | None:
    """Bar chart of SCR per task_id (Condition C)."""
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    task_ids = sorted(set(r.get("task_id", "unknown") for r in c))
    if not task_ids:
        return None

    data = []
    for tid in task_ids:
        t_recs = [r for r in c if r.get("task_id") == tid]
        scr = compute_scr(t_recs)
        data.append((tid, scr))

    # Sort by SCR descending
    data.sort(key=lambda x: x[1], reverse=True)
    tids = [d[0] for d in data]
    scrs = [d[1] for d in data]

    colors = [_val_to_color(s, "RdYlGn") for s in scrs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tids, y=scrs,
        marker_color=colors,
        text=[f"{v:.2f}" for v in scrs], textposition="outside",
        hovertemplate="%{x}: SCR=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="SCR by Task (Condition C, default styles)",
        xaxis_title="Task", yaxis_title="SCR",
        yaxis_range=[0, 1.15],
        xaxis_tickangle=-30,
    )
    return fig


# ── 9b. Task × Conflict interaction ─────────────────────────────────────

def fig_task_conflict_interaction(
    records: list[dict],
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
) -> go.Figure | None:
    """Strip plot showing within-conflict-direction variance of per-task SCR.

    For each conflict-direction, shows the distribution of per-task SCRs
    (one dot per task, styles pooled within each task). Tight clusters mean
    task content doesn't matter; wide spread means task interacts with the
    constraint.
    """
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    actual_cids = list({r.get("conflict_id", r["constraint_type"]) for r in c})
    ordered_cids = ordered(actual_cids, conflict_order) if conflict_order else sorted(actual_cids)
    if not ordered_cids:
        return None

    # Build rows: one per conflict-direction, with per-task SCR values
    rows: list[dict] = []
    for cid in ordered_cids:
        c_recs = [r for r in c if r.get("conflict_id", r["constraint_type"]) == cid]
        for direction in ("a_to_b", "b_to_a"):
            d_recs = [r for r in c_recs if r["direction"] == direction]
            # Group by task
            tasks: dict[str, list[dict]] = defaultdict(list)
            for r in d_recs:
                tasks[r.get("task_id", "?")].append(r)
            task_scrs = []
            task_labels = []
            for tid, recs in sorted(tasks.items()):
                task_scrs.append(compute_scr(recs))
                task_labels.append(tid)
            if task_scrs:
                mean_scr = compute_scr(d_recs)
                obs_var = sum((s - mean_scr) ** 2 for s in task_scrs) / len(task_scrs)
                # Expected binomial variance: p(1-p)/n per task
                # n = median samples per task (should be ~25 = 5 sys × 5 usr styles)
                n_per_task = sorted(len(recs) for recs in tasks.values())
                median_n = n_per_task[len(n_per_task) // 2]
                p = mean_scr
                binom_var = p * (1 - p) / median_n if median_n > 0 else 0
                overdisp = obs_var / binom_var if binom_var > 0 else float("nan")
                rows.append({
                    "cid": cid,
                    "direction": direction,
                    "label": f"{cid} ({direction})",
                    "mean": mean_scr,
                    "task_scrs": task_scrs,
                    "task_labels": task_labels,
                    "std": obs_var ** 0.5,
                    "overdispersion": overdisp,
                    "n_per_task": median_n,
                })

    if not rows:
        return None

    # Order by canonical conflict order, then a_to_b before b_to_a (reversed for y-axis)
    # y=0 is bottom, so reverse: last row in list = top of chart
    cid_rank = {cid: i for i, cid in enumerate(ordered_cids)}
    dir_rank = {"a_to_b": 0, "b_to_a": 1}
    # Sort so that bottom of chart (low y) has last conflicts, top has first
    rows.sort(key=lambda r: (cid_rank.get(r["cid"], 999), dir_rank.get(r["direction"], 0)),
              reverse=True)

    n_rows = len(rows)
    labels = [r["label"] for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.85, 0.15],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    # ── Left panel: strip plot ──
    for direction, color in [("a_to_b", "#3498db"), ("b_to_a", "#e67e22")]:
        xs, ys, hovers = [], [], []
        for i, row in enumerate(rows):
            if row["direction"] != direction:
                continue
            for scr_val, tid in zip(row["task_scrs"], row["task_labels"]):
                xs.append(scr_val)
                h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
                jitter = ((h % 1000) / 1000 - 0.5) * 0.3
                ys.append(i + jitter)
                hovers.append(f"{row['cid']}<br>{tid}<br>SCR={scr_val:.3f}")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=4, color=color, opacity=0.4),
            name=f"{direction}",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
            legendgroup=direction,
        ), row=1, col=1)

    # Mean markers on left panel
    for i, row in enumerate(rows):
        color = "#3498db" if row["direction"] == "a_to_b" else "#e67e22"
        fig.add_trace(go.Scatter(
            x=[row["mean"]], y=[i], mode="markers",
            marker=dict(size=8, color=color, symbol="diamond",
                        line=dict(width=1, color="white")),
            showlegend=False,
            hovertemplate=(
                f"{row['cid']} ({row['direction']})<br>mean={row['mean']:.3f}, "
                f"std={row['std']:.3f}<br>"
                f"overdispersion={row['overdispersion']:.2f} "
                f"(n={row['n_per_task']}/task)<extra></extra>"
            ),
        ), row=1, col=1)

    # ── Right panel: overdispersion markers on log scale ──
    import math
    overdisp_xs = []
    marker_colors = []
    hover_texts = []
    for row in rows:
        d = row["overdispersion"]
        if d != d or d <= 0:  # nan or zero
            overdisp_xs.append(None)
            marker_colors.append("rgba(200,200,200,0.5)")
        else:
            overdisp_xs.append(d)
            if d > 2.0:
                marker_colors.append("#e74c3c")
            elif d < 0.5:
                marker_colors.append("#3498db")
            else:
                marker_colors.append("rgba(150,150,150,0.7)")
        hover_texts.append(
            f"{row['cid']} ({row['direction']})<br>D={d:.2f}"
            if d == d else f"{row['cid']} ({row['direction']})<br>D=nan"
        )

    fig.add_trace(go.Scatter(
        x=overdisp_xs,
        y=list(range(n_rows)),
        mode="markers",
        marker=dict(size=6, color=marker_colors, symbol="square"),
        showlegend=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ), row=1, col=2)

    # Reference line at D=1
    fig.add_vline(x=1, line_dash="dash", line_color="gray", line_width=1,
                  row=1, col=2)

    fig.update_layout(
        title="Task × Conflict Interaction (per-task SCR within each conflict-direction)",
        height=max(400, 18 * n_rows + 100),
        margin=dict(l=280),
    )
    # Left panel axes
    fig.update_xaxes(title_text="SCR", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=labels,
        row=1, col=1,
    )
    # Right panel axes — log scale centered on D=1
    fig.update_xaxes(title_text="D", type="log", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    return fig


# ── 10. Style interaction heatmap ────────────────────────────────────────

def fig_style_interaction_heatmap(records: list[dict],
                                  system_style_order: list[str] | None = None,
                                  user_style_order: list[str] | None = None) -> go.Figure | None:
    """5x5 heatmap of SCR for each (user_style, system_style) combination."""
    c = [r for r in records if r["condition"] == "C"]
    if not c:
        return None

    actual_sys = list({r["system_style"] for r in c})
    system_styles = ordered(actual_sys, system_style_order) if system_style_order else sorted(actual_sys)
    actual_usr = list({r.get("user_style", "unknown") for r in c})
    user_styles = ordered(actual_usr, user_style_order) if user_style_order else sorted(actual_usr)

    z = []
    text = []
    for us in user_styles:
        row_z = []
        row_text = []
        for ss in system_styles:
            cell_recs = [r for r in c
                         if r.get("user_style") == us and r["system_style"] == ss]
            if cell_recs:
                scr = compute_scr(cell_recs)
                row_z.append(scr)
                row_text.append(f"{scr:.2f}")
            else:
                row_z.append(None)
                row_text.append("")
        z.append(row_z)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z, x=system_styles, y=user_styles,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar_title="SCR",
        hovertemplate="User: %{y}<br>System: %{x}<br>SCR: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="SCR: User Style x System Style (Condition C)",
        xaxis_title="System Style",
        yaxis_title="User Style",
    )
    return fig


# ── 11. Score distributions ─────────────────────────────────────────────

def fig_score_distributions(records: list[dict],
                            cond_c_filter: dict[str, str] | None = None) -> go.Figure | None:
    """Box plots of verify_system_score and verify_user_score by label."""
    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    # Check that score fields exist
    if not any("verify_system_score" in r for r in c):
        return None
    if not any("verify_user_score" in r for r in c):
        return None

    labels = [l for l in LABEL_ORDER if any(r["label"] == l for r in c)]
    if not labels:
        return None

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["verify_system_score", "verify_user_score"],
                        horizontal_spacing=0.12)

    for i, score_field in enumerate(["verify_system_score", "verify_user_score"], 1):
        for label in labels:
            label_recs = [r for r in c if r["label"] == label and score_field in r]
            scores = [r[score_field] for r in label_recs
                      if r[score_field] is not None]
            if not scores:
                continue
            fig.add_trace(go.Box(
                y=scores,
                name=label.replace("_", " ").title(),
                marker_color=LABEL_COLORS.get(label, "#bdc3c7"),
                showlegend=(i == 1),
                legendgroup=label,
            ), row=1, col=i)

    fig.update_layout(
        title="Verification Score Distributions by Label (Condition C)",
        height=450,
    )
    fig.update_yaxes(title_text="Score", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[-0.05, 1.05], row=1, col=2)
    return fig


# ── 12. Threshold proximity ──────────────────────────────────────────────

def fig_threshold_proximity(
    records: list[dict],
    calibration_data: dict[str, dict] | None = None,
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
) -> go.Figure | None:
    """Horizontal bar chart showing % of records near threshold per float conflict.

    For each float-scored conflict (those with a threshold in calibration_data):
    - Compute % of condition C records with verify_system_score within +/-0.05 of threshold
    - Color gradient: green (low %, confident) -> red (high %, borderline)

    Bool-scored conflicts are skipped.
    Returns None if no calibration data or no float conflicts.
    """
    if not calibration_data:
        return None

    c = _filter_cond_c(records, cond_c_filter)
    if not c:
        return None

    # Find float conflicts (those with a non-None threshold)
    float_conflicts: dict[str, float] = {}
    for cid, cdata in calibration_data.items():
        thresh = cdata.get("threshold")
        if thresh is not None:
            float_conflicts[cid] = thresh

    if not float_conflicts:
        return None

    # Apply canonical ordering
    actual_cids = [cid for cid in float_conflicts if any(
        r.get("conflict_id", r["constraint_type"]) == cid for r in c
    )]
    ordered_cids = list(reversed(
        ordered(actual_cids, conflict_order) if conflict_order else sorted(actual_cids)
    ))

    if not ordered_cids:
        return None

    pcts = []
    thresholds = []
    for cid in ordered_cids:
        thresh = float_conflicts[cid]
        cid_recs = [r for r in c if r.get("conflict_id", r["constraint_type"]) == cid]
        if not cid_recs:
            pcts.append(0.0)
            thresholds.append(thresh)
            continue
        near_count = sum(
            1 for r in cid_recs
            if abs(r.get("verify_system_score", 0) - thresh) <= 0.05
        )
        pct = near_count / len(cid_recs) * 100
        pcts.append(pct)
        thresholds.append(thresh)

    colors = [_val_to_color(p / 100, "RdYlGn_r") for p in pcts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ordered_cids, x=pcts, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts], textposition="outside",
        customdata=thresholds,
        hovertemplate="%{y}: %{x:.1f}% near threshold (T=%{customdata:.3f})<extra></extra>",
    ))

    n_conflicts = len(ordered_cids)
    fig.update_layout(
        title="Threshold Proximity: % of Records Within +/-0.05 of Threshold",
        xaxis_title="% of records near threshold",
        xaxis_range=[0, max(pcts) * 1.2 + 5] if pcts else [0, 100],
        height=max(400, 30 * n_conflicts + 100),
        margin=dict(l=200),
    )
    return fig


# ── 13. Verifier audit reliability ───────────────────────────────────────

def fig_verifier_audit_reliability(
    audit_data: dict[str, dict] | None = None,
    conflict_order: list[str] | None = None,
) -> go.Figure | None:
    """Two-panel audit reliability chart.

    Left: per-direction error rates with estimated fix improvement.
    Right: meta-commentary prevalence colored by misclassification impact.
    Returns None if no audit data.
    """
    if not audit_data:
        return None

    RATING_COLORS = {
        "GREEN": "#2ecc71",
        "YELLOW": "#f1c40f",
        "AMBER": "#e67e22",
        "RED": "#e74c3c",
    }

    def _action_text(ra) -> str:
        """Normalize recommended_action to a display string (handles dict + str)."""
        if isinstance(ra, dict):
            return ra.get("summary") or ra.get("type") or ""
        return ra if isinstance(ra, str) else ""

    def _wrap(text: str, width: int = 60) -> str:
        """Word-wrap text with <br> for hover readability."""
        words = (text or "").split()
        lines, line = [], ""
        for w in words:
            if line and len(line) + 1 + len(w) > width:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            lines.append(line)
        return "<br>".join(lines)

    actual_cids = list(audit_data.keys())
    ordered_cids = (
        ordered(actual_cids, conflict_order) if conflict_order else sorted(actual_cids)
    )
    if not ordered_cids:
        return None

    # Build rows: one per conflict-direction (a_to_b, b_to_a)
    rows: list[dict] = []
    for cid in ordered_cids:
        entry = audit_data[cid]
        for direction in ("a_to_b", "b_to_a"):
            epct = entry.get(f"error_pct_{direction}", 0.0)
            meta_key = f"meta_commentary_{direction}"
            meta_count = entry.get(meta_key, 0)
            n = entry.get(f"n_{direction}", 0)
            rows.append({
                "cid": cid,
                "direction": direction,
                "label": f"{cid} ({direction})",
                "error_pct": epct,
                "rating": entry.get("rating", "GREEN"),
                "root_cause": entry.get("root_cause", ""),
                "estimated_fix_error_pct": entry.get("estimated_fix_error_pct"),
                "fix_confidence": entry.get("fix_confidence"),
                "fix_complexity": entry.get("fix_complexity"),
                "fix_description": entry.get("fix_description", ""),
                "meta_count": meta_count,
                "meta_pct": meta_count / n * 100 if n > 0 else 0,
                "meta_causes_misclass": entry.get("meta_causes_misclassification", False),
                "recommended_action": _action_text(entry.get("recommended_action", "")),
            })

    if not rows:
        return None

    # Sort: canonical order, a_to_b before b_to_a, reversed for y-axis (bottom=last)
    cid_rank = {cid: i for i, cid in enumerate(ordered_cids)}
    dir_rank = {"a_to_b": 0, "b_to_a": 1}
    rows.sort(
        key=lambda r: (cid_rank.get(r["cid"], 999), dir_rank.get(r["direction"], 0)),
        reverse=True,
    )

    n_rows = len(rows)
    labels = [r["label"] for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.82, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=["Error Rate (%) & Fix Potential", "Meta-commentary"],
    )

    # ── Left panel: error rates ──
    dir_colors = {"a_to_b": "#3498db", "b_to_a": "#e67e22"}
    shown_legend = set()

    for i, row in enumerate(rows):
        direction = row["direction"]
        rating = row["rating"]
        color = RATING_COLORS.get(rating, "#bdc3c7")
        show_dir = direction not in shown_legend
        shown_legend.add(direction)

        # Current error marker
        root_cause_wrapped = _wrap(row["root_cause"])
        action_wrapped = _wrap(row["recommended_action"])
        fig.add_trace(go.Scatter(
            x=[row["error_pct"]], y=[i],
            mode="markers",
            marker=dict(size=9, color=color, symbol="circle",
                        line=dict(width=1, color="white")),
            name=direction if show_dir else None,
            showlegend=False,
            legendgroup=direction,
            hovertemplate=(
                f"{row['label']}<br>"
                f"Error: {row['error_pct']:.2f}%<br>"
                f"Rating: {row['rating']}<br>"
                f"<b>Root cause:</b> {root_cause_wrapped}<br>"
                f"<b>Action:</b> {action_wrapped}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        # Estimated fix marker + connecting line
        est = row["estimated_fix_error_pct"]
        if est is not None:
            # Connecting line (improvement gap)
            fig.add_trace(go.Scatter(
                x=[row["error_pct"], est], y=[i, i],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            # Fix diamond
            conf = row["fix_confidence"] or "?"
            comp = row["fix_complexity"] or "?"
            fix_wrapped = _wrap(row["fix_description"])
            fig.add_trace(go.Scatter(
                x=[est], y=[i],
                mode="markers+text",
                marker=dict(size=8, color=color, symbol="diamond",
                            opacity=0.5, line=dict(width=1, color="white")),
                text=[f" {conf}/{comp}"],
                textposition="middle right",
                textfont=dict(size=8, color="#7f8c8d"),
                showlegend=False,
                hovertemplate=(
                    f"{row['label']}<br>"
                    f"Est. fix error: {est:.2f}%<br>"
                    f"Confidence: {conf}<br>"
                    f"Complexity: {comp}<br>"
                    f"<b>Fix:</b> {fix_wrapped}"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

    # ── Right panel: meta-commentary ──
    for i, row in enumerate(rows):
        meta_color = "#e74c3c" if row["meta_causes_misclass"] else "#95a5a6"
        fig.add_trace(go.Scatter(
            x=[row["meta_count"]], y=[i],
            mode="markers",
            marker=dict(
                size=max(4, min(14, row["meta_pct"] / 5 + 4)),
                color=meta_color,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            showlegend=False,
            hovertemplate=(
                f"{row['label']}<br>"
                f"Meta-commentary: {row['meta_count']} ({row['meta_pct']:.1f}%)<br>"
                f"Causes misclassification: {row['meta_causes_misclass']}"
                "<extra></extra>"
            ),
        ), row=1, col=2)

    # ── Axes & layout ──
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=labels,
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Error %", row=1, col=1)
    fig.update_xaxes(title_text="Count", row=1, col=2)

    # Add legend entries for severity ratings
    for rating, color in RATING_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            name=rating, showlegend=True, legendgroup="ratings",
        ))
    # Legend entry for fix estimate
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=8, color="#bdc3c7", symbol="diamond", opacity=0.5),
        name="Est. post-fix", showlegend=True, legendgroup="fix",
    ))
    # Legend entries for meta-commentary
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=8, color="#e74c3c"),
        name="Meta: causes misclass.", showlegend=True, legendgroup="meta",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=8, color="#95a5a6"),
        name="Meta: benign", showlegend=True, legendgroup="meta",
    ))

    fig.update_layout(
        title="Verifier Audit: Error Rates, Fix Potential & Meta-commentary",
        height=max(400, 28 * n_rows + 140),
        margin=dict(l=250, r=40, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── 14. Score vs threshold ridge plot ────────────────────────────────────

def _normalize_score(score: float, direction: str) -> float:
    """Normalize verify_system_score to the constraint-a scale.

    For a_to_b: score already measures constraint a → return as-is.
    For b_to_a: score measures constraint b (inverted) → return 1 - score
    so it's on the constraint-a scale.
    """
    if direction == "b_to_a":
        return 1.0 - score
    return score


def fig_score_vs_threshold(
    records: list[dict],
    calibration_data: dict[str, dict] | None = None,
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
) -> go.Figure | None:
    """Violin plot of verify_system_score normalized to constraint-a scale,
    centered on the threshold.

    One row per float-scored conflict. X-axis = normalized_score - threshold.
    Right of 0 = response looks like constraint a.
    Left of 0 = response looks like constraint b.

    All b_to_a scores are flipped (1 - score) so both directions are on the
    same scale. This means for a_to_b followed_system (system=a) blue clusters
    right, and for b_to_a followed_system (system=b) blue clusters left.

    Layers:
    - Gray violins: Condition A and B baselines (no conflict).
    - Blue/red violins: Condition C, split by label.

    Returns None if no calibration data or no float conflicts with scores.
    """
    if not calibration_data:
        return None

    c_recs = _filter_cond_c(records, cond_c_filter)
    a_recs = [r for r in records if r["condition"] == "A"]
    b_recs = [r for r in records if r["condition"] == "B"]

    if not c_recs:
        return None

    # Identify float conflicts (those with a threshold)
    float_conflicts: dict[str, float] = {}
    for cid, cdata in calibration_data.items():
        thresh = cdata.get("threshold")
        if thresh is not None:
            float_conflicts[cid] = thresh

    if not float_conflicts:
        return None

    # Filter to conflicts that have score data in condition C
    cids_with_data = []
    for cid in float_conflicts:
        has_scores = any(
            r.get("conflict_id", r.get("constraint_type")) == cid
            and r.get("verify_system_score") is not None
            for r in c_recs
        )
        if has_scores:
            cids_with_data.append(cid)

    if not cids_with_data:
        return None

    # Canonical order (Plotly bottom-to-top, so reverse)
    ordered_cids = list(reversed(
        ordered(cids_with_data, conflict_order) if conflict_order else sorted(cids_with_data)
    ))

    def _get_cid(r: dict) -> str:
        return r.get("conflict_id", r.get("constraint_type", ""))

    def _centered_score(r: dict, thresh: float) -> float:
        """Normalize score to constraint-a scale and center on threshold."""
        raw = r["verify_system_score"]
        normed = _normalize_score(raw, r.get("direction", "a_to_b"))
        return normed - thresh

    import numpy as np

    fig = go.Figure()

    # Collect baseline scores grouped by which constraint was followed.
    # "Baseline a" = Cond A a_to_b (system=a) + Cond B b_to_a (user=a) → right
    # "Baseline b" = Cond A b_to_a (system=b) + Cond B a_to_b (user=b) → left
    ab_recs = a_recs + b_recs

    def _is_baseline_a(r: dict) -> bool:
        return ((r["condition"] == "A" and r.get("direction") == "a_to_b")
                or (r["condition"] == "B" and r.get("direction") == "b_to_a"))

    def _is_baseline_b(r: dict) -> bool:
        return ((r["condition"] == "A" and r.get("direction") == "b_to_a")
                or (r["condition"] == "B" and r.get("direction") == "a_to_b"))

    # Helper to compute median + min/max for a list of scores
    def _median_range(scores: list[float]) -> tuple[float, float, float] | None:
        if not scores:
            return None
        return float(min(scores)), float(np.median(scores)), float(max(scores))

    # Layer 1: Condition C — histogram ridgeline (one row per conflict)
    # Use subplots: one row per conflict, shared x-axis
    from plotly.subplots import make_subplots

    # We'll build a separate figure with subplots for the ridgeline
    n_conflicts = len(ordered_cids)
    # Display order: ordered_cids is reversed for Plotly bottom-to-top,
    # but for subplots we go top-to-bottom, so un-reverse
    display_cids = list(reversed(ordered_cids))

    fig = make_subplots(
        rows=n_conflicts, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.002,
        row_heights=[1] * n_conflicts,
    )

    bin_width = 0.01
    bins_dict = dict(start=-1.2, end=1.2, size=bin_width)

    NEAR_THRESHOLD = 0.05  # highlight zone around x=0

    # Track which conflicts have high threshold proximity for annotation coloring
    high_proximity_cids: set[str] = set()

    for i, cid in enumerate(display_cids, 1):
        thresh = float_conflicts[cid]
        scores = [
            _centered_score(r, thresh)
            for r in c_recs
            if _get_cid(r) == cid and r.get("verify_system_score") is not None
        ]

        if scores:
            # Split into near-threshold and far-from-threshold
            scores_far = [s for s in scores if abs(s) > NEAR_THRESHOLD]
            scores_near = [s for s in scores if abs(s) <= NEAR_THRESHOLD]
            near_pct = len(scores_near) / len(scores) * 100

            if near_pct > 10:
                high_proximity_cids.add(cid)

            # Far-from-threshold bins (gray)
            if scores_far:
                fig.add_trace(go.Histogram(
                    x=scores_far,
                    xbins=bins_dict,
                    marker_color="rgba(52,73,94,0.6)",
                    marker_line_color="rgba(52,73,94,0.8)",
                    marker_line_width=0.5,
                    name="Far from threshold" if i == 1 else None,
                    showlegend=(i == 1),
                    legendgroup="far",
                    hovertemplate=f"{cid}<br>score−T: %{{x:.2f}}<br>count: %{{y}}<extra></extra>",
                ), row=i, col=1)

            # Near-threshold bins (orange/red highlight)
            if scores_near:
                fig.add_trace(go.Histogram(
                    x=scores_near,
                    xbins=bins_dict,
                    marker_color="rgba(230,126,34,0.8)",
                    marker_line_color="rgba(211,84,0,0.9)",
                    marker_line_width=0.5,
                    name=f"Near threshold (±{NEAR_THRESHOLD})" if i == 1 else None,
                    showlegend=(i == 1),
                    legendgroup="near",
                    hovertemplate=f"{cid}<br>score−T: %{{x:.2f}}<br>count: %{{y}}<br>NEAR THRESHOLD<extra></extra>",
                ), row=i, col=1)

        # Compute max histogram count for this subplot (for KDE ribbon scaling)
        hist_max = 0
        if scores:
            counts, _ = np.histogram(scores, bins=np.arange(-1.2, 1.2 + bin_width, bin_width))
            hist_max = int(counts.max()) if len(counts) > 0 else 0

        # Add baseline diamonds (min/max whiskers) + KDE density ribbons
        from scipy.stats import gaussian_kde as _gkde
        for bl_idx, (bl_pred, bl_color, bl_name) in enumerate([
            (_is_baseline_a, "#2980b9", "constraint a"),
            (_is_baseline_b, "#e74c3c", "constraint b"),
        ]):
            bl_scores = [
                _centered_score(r, thresh)
                for r in ab_recs
                if _get_cid(r) == cid
                and r.get("verify_system_score") is not None
                and bl_pred(r)
            ]
            result = _median_range(bl_scores)
            if result:
                smin, med, smax = result
                span = smax - smin
                error_x_cfg = (
                    dict(type="data", symmetric=False,
                         array=[smax - med], arrayminus=[med - smin],
                         color=bl_color, thickness=1.5, width=3)
                    if span > 0.001
                    else dict(type="data", symmetric=True, array=[0], visible=False)
                )
                fig.add_trace(go.Scatter(
                    x=[med],
                    y=[0],
                    mode="markers",
                    marker=dict(color=bl_color, size=8, symbol="diamond",
                                line=dict(width=1, color="white")),
                    error_x=error_x_cfg,
                    name=f"Baseline: {bl_name}" if i == 1 else None,
                    showlegend=(i == 1),
                    legendgroup=f"bl_{bl_name}",
                    hovertemplate=(f"Baseline {bl_name}<br>median: {med:.3f}"
                                  f"<br>range: [{smin:.3f}, {smax:.3f}]<extra></extra>"),
                ), row=i, col=1)

                # KDE density ribbon at y=0, capped to [smin, smax]
                if len(bl_scores) >= 5 and hist_max > 0:
                    bl_arr = np.array(bl_scores)
                    try:
                        kde = _gkde(bl_arr)
                        # Evaluate KDE strictly within [smin, smax]
                        x_kde = np.linspace(smin, smax, 200)
                        y_kde = kde(x_kde)
                        # Scale ribbon height to ~25% of histogram max
                        ribbon_height = hist_max * 0.25
                        y_upper = (y_kde / y_kde.max() * ribbon_height).tolist()
                        y_lower = [0.0] * len(x_kde)
                        x_list = x_kde.tolist()

                        r, g, b = int(bl_color[1:3], 16), int(bl_color[3:5], 16), int(bl_color[5:7], 16)
                        fig.add_trace(go.Scatter(
                            x=x_list + x_list[::-1],
                            y=y_upper + y_lower[::-1],
                            fill="toself",
                            fillcolor=f"rgba({r},{g},{b},0.15)",
                            line=dict(color=f"rgba({r},{g},{b},0.5)", width=1),
                            showlegend=False,
                            legendgroup=f"bl_{bl_name}",
                            hoverinfo="skip",
                        ), row=i, col=1)
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Skip KDE if it fails (e.g., all identical scores)

        # Add threshold line
        fig.add_vline(x=0, line_dash="dash", line_color="#333", line_width=1,
                      row=i, col=1)

        # Hide y-axis ticks, gridlines, and zero line
        fig.update_yaxes(
            showticklabels=False, title_text="",
            showgrid=False, zeroline=False,
            row=i, col=1,
        )

    # Add horizontal conflict name labels + pareto annotations
    for i, cid in enumerate(display_cids, 1):
        yaxis_key = f"yaxis{i}" if i > 1 else "yaxis"
        domain = fig.layout[yaxis_key].domain
        y_mid = (domain[0] + domain[1]) / 2 if domain else (1.0 - i / n_conflicts)
        font_color = "#c0392b" if cid in high_proximity_cids else "#2c3e50"
        fig.add_annotation(
            text=cid,
            xref="paper", yref="paper",
            x=-0.01, y=y_mid,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(size=10, color=font_color),
        )
        # Pareto metrics annotation (right side) — skip if perfect (BA=1, costs=0)
        if calibration_data and cid in calibration_data:
            cd = calibration_data[cid]
            ba = cd.get("balanced_accuracy")
            d_n = cd.get("d_norm")
            c_n = cd.get("c_norm")
            is_feasible = cd.get("feasible", True)
            fallback = cd.get("fallback")

            if not is_feasible and fallback:
                # Infeasible with fallback — show in red
                fallback_label = "valley" if fallback == "valley" else "baseline mid"
                lines = [f"<b>INFEASIBLE</b>", f"T ← {fallback_label}"]
                if ba is not None:
                    lines.append(f"BA={ba:.3f}")
                fig.add_annotation(
                    text="<br>".join(lines),
                    xref="paper", yref="paper",
                    x=1.01, y=y_mid,
                    xanchor="left", yanchor="middle",
                    showarrow=False,
                    font=dict(size=8, color="#e74c3c"),
                )
            elif ba is not None:
                is_perfect = (ba == 1.0
                              and (d_n is None or d_n == 0)
                              and (c_n is None or c_n == 0))
                if not is_perfect:
                    lines = [f"BA={ba:.3f}"]
                    if d_n is not None:
                        lines.append(f"d={d_n:.4f}")
                    if c_n is not None:
                        lines.append(f"c={c_n:.4f}")
                    fig.add_annotation(
                        text="<br>".join(lines),
                        xref="paper", yref="paper",
                        x=1.01, y=y_mid,
                        xanchor="left", yanchor="middle",
                        showarrow=False,
                        font=dict(size=8, color="#7f8c8d"),
                    )

    # Update bottom x-axis label
    fig.update_xaxes(
        title_text="← constraint b · Score − Threshold · constraint a →",
        row=n_conflicts, col=1,
    )

    fig.update_layout(
        title="Score Distribution Relative to Threshold (Float Conflicts)",
        height=max(600, 55 * n_conflicts + 100),
        margin=dict(l=200, r=120, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.05,
        barmode="overlay",
    )

    return fig


# ── Response structure figures ────────────────────────────────────────────


_RESP_TYPE_ORDER = ["clean", "meta_content", "refusal_content", "bare_refusal"]
_RESP_TYPE_LABELS = {
    "clean": "Clean",
    "meta_content": "Meta + Content",
    "refusal_content": "Refusal + Content",
    "bare_refusal": "Bare Refusal",
}
_RESP_TYPE_COLORS = {
    "clean": "#2ecc71",
    "meta_content": "#f39c12",
    "refusal_content": "#e67e22",
    "bare_refusal": "#e74c3c",
}


def _classify_response_type(rec: dict) -> str:
    """Classify a record into one of four response types from precomputed tags."""
    tags = rec.get("refusal_tags")
    if tags is None:
        return "clean"
    s = tags["structure"]
    if s[0] == "refusal":
        bare = s in (["refusal"], ["refusal", "helpfulness_followup"])
        return "bare_refusal" if bare else "refusal_content"
    if "metacommentary" in s:
        return "meta_content"
    return "clean"


def _get_structure_pattern(rec: dict) -> tuple:
    """Get structure pattern as a hashable tuple."""
    tags = rec.get("refusal_tags")
    if tags is None:
        return ("content",)
    return tuple(tags["structure"])


def fig_response_structure_overview(
    records: list[dict],
    type_map: dict[str, dict[str, str]],
    *,
    cond_c_filter: dict[str, str] | None = None,
) -> go.Figure | None:
    """Stacked bar chart: response type proportions by constraint type.

    Left subplot: stacked bars for presence/avoidance/ambiguous + overall.
    Right subplot: top structure patterns table with counts and label breakdown.
    """
    c_recs = _filter_cond_c(records, cond_c_filter)
    if not c_recs:
        return None

    constraint_types = ["presence", "avoidance", "ambiguous"]

    # Classify each record's constraint type and response type
    def _sys_constraint(rec: dict) -> str:
        cid = rec.get("conflict_id")
        direction = rec.get("direction")
        types = type_map.get(cid, {})
        if direction == "a_to_b":
            return types.get("a", "unknown")
        elif direction == "b_to_a":
            return types.get("b", "unknown")
        return "unknown"

    # Build counts per (constraint_type, response_type)
    groups = {ct: Counter() for ct in constraint_types}
    groups["all"] = Counter()
    pattern_counter: Counter = Counter()
    pattern_labels: dict[tuple, Counter] = defaultdict(Counter)

    for rec in c_recs:
        ct = _sys_constraint(rec)
        rt = _classify_response_type(rec)
        if ct in groups:
            groups[ct][rt] += 1
        groups["all"][rt] += 1

        pat = _get_structure_pattern(rec)
        pattern_counter[pat] += 1
        pattern_labels[pat][rec.get("label", "?")] += 1

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.08,
        subplot_titles=["Response Type by Constraint Type", "Top Structure Patterns"],
    )

    # Left panel: stacked bars
    categories = constraint_types + ["all"]
    cat_labels = ["Presence", "Avoidance", "Ambiguous", "All"]

    for rt in _RESP_TYPE_ORDER:
        values = []
        for cat in categories:
            total = sum(groups[cat].values())
            values.append(groups[cat][rt] / total * 100 if total else 0)

        fig.add_trace(
            go.Bar(
                x=cat_labels,
                y=values,
                name=_RESP_TYPE_LABELS[rt],
                marker_color=_RESP_TYPE_COLORS[rt],
                hovertemplate="%{x}: %{y:.1f}%<extra>" + _RESP_TYPE_LABELS[rt] + "</extra>",
                legendgroup=rt,
            ),
            row=1, col=1,
        )

    fig.update_yaxes(title_text="% of Condition C records", row=1, col=1)

    # Right panel: top non-clean patterns as horizontal bars
    # Skip pure "content" — it dominates the scale and isn't interesting
    top_patterns = [
        (pat, count) for pat, count in pattern_counter.most_common(30)
        if pat != ("content",)
    ][:12]
    total_recs = len(c_recs)

    pat_names = []
    pat_pcts = []
    pat_hovers = []
    pat_colors = []

    for pat, count in reversed(top_patterns):  # reversed for bottom-up ordering
        name = " → ".join(pat) if pat else "(empty)"
        # Truncate long names
        if len(name) > 45:
            name = name[:42] + "..."
        pat_names.append(name)
        pct = count / total_recs * 100
        pat_pcts.append(pct)

        labels = pattern_labels[pat]
        n = count
        sys_pct = labels.get("followed_system", 0) / n * 100 if n else 0
        usr_pct = labels.get("followed_user", 0) / n * 100 if n else 0
        hover = (f"N={count:,} ({pct:.1f}%)<br>"
                 f"→sys: {sys_pct:.1f}%<br>"
                 f"→usr: {usr_pct:.1f}%")
        pat_hovers.append(hover)

        # Color by dominant type
        if pat[0] == "refusal":
            pat_colors.append(_RESP_TYPE_COLORS["refusal_content"])
        elif "metacommentary" in pat:
            pat_colors.append(_RESP_TYPE_COLORS["meta_content"])
        else:
            pat_colors.append(_RESP_TYPE_COLORS["clean"])

    fig.add_trace(
        go.Bar(
            y=pat_names,
            x=pat_pcts,
            orientation="h",
            marker_color=pat_colors,
            hovertext=pat_hovers,
            hoverinfo="text",
            showlegend=False,
            text=[f"{p:.1f}%" for p in pat_pcts],
            textposition="outside",
            textfont=dict(size=10),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="% of Condition C records", row=1, col=2)

    fig.update_layout(
        barmode="stack",
        height=500,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=0.45),
    )

    return fig


def fig_response_structure_per_conflict(
    records: list[dict],
    type_map: dict[str, dict[str, str]],
    preprocessing_map: dict[str, list[str]],
    *,
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
) -> go.Figure | None:
    """Per-conflict horizontal stacked bars showing response type proportions.

    Each conflict gets two rows (a_to_b, b_to_a). Right-side annotations show
    system constraint type and preprocessing tags.
    """
    c_recs = _filter_cond_c(records, cond_c_filter)
    if not c_recs:
        return None

    # Group by (conflict_id, direction)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in c_recs:
        key = (rec.get("conflict_id", "?"), rec.get("direction", "?"))
        groups[key].append(rec)

    # Get ordered conflict list
    all_conflicts = sorted({k[0] for k in groups})
    if conflict_order:
        all_conflicts = ordered(all_conflicts, conflict_order)

    # Build row data
    row_labels = []
    row_data = []  # list of {rt: pct}
    row_annotations = []

    for cid in all_conflicts:
        for direction in ("a_to_b", "b_to_a"):
            recs = groups.get((cid, direction), [])
            if not recs:
                continue

            n = len(recs)
            counts = Counter(_classify_response_type(r) for r in recs)

            # System constraint type
            types = type_map.get(cid, {})
            sys_ct = types.get("a", "?") if direction == "a_to_b" else types.get("b", "?")

            # Preprocessing icons
            pp = preprocessing_map.get(cid, [])
            pp_icons = "".join(
                PREPROCESSING_ICONS.get(p, ("❓",))[0] for p in pp
            ) if pp else ""

            dir_short = "a→b" if direction == "a_to_b" else "b→a"
            row_labels.append(f"{cid} ({dir_short})")
            row_data.append({rt: counts.get(rt, 0) / n * 100 for rt in _RESP_TYPE_ORDER})
            row_annotations.append(
                f"{sys_ct[:4]}" + (f" {pp_icons}" if pp_icons else "")
            )

    if not row_labels:
        return None

    n_rows = len(row_labels)

    # Reverse for bottom-up plotly layout
    row_labels_r = list(reversed(row_labels))
    row_data_r = list(reversed(row_data))
    row_annotations_r = list(reversed(row_annotations))

    fig = go.Figure()

    # Plot only non-clean types — clean dominates and obscures the rest.
    # Hover still shows clean % for context.
    non_clean_types = [rt for rt in _RESP_TYPE_ORDER if rt != "clean"]
    for rt in non_clean_types:
        fig.add_trace(go.Bar(
            y=row_labels_r,
            x=[rd[rt] for rd in row_data_r],
            name=_RESP_TYPE_LABELS[rt],
            orientation="h",
            marker_color=_RESP_TYPE_COLORS[rt],
            customdata=[rd["clean"] for rd in row_data_r],
            hovertemplate="%{y}: %{x:.1f}% " + _RESP_TYPE_LABELS[rt]
                         + "<br>Clean: %{customdata:.1f}%<extra></extra>",
        ))

    # Right-side annotations for constraint type + preprocessing
    for i, ann in enumerate(row_annotations_r):
        if ann:
            fig.add_annotation(
                text=ann,
                xref="paper", yref="y",
                x=1.01, y=row_labels_r[i],
                xanchor="left", yanchor="middle",
                showarrow=False,
                font=dict(size=8, color="#7f8c8d"),
            )

    fig.update_layout(
        barmode="stack",
        title="Per-Conflict Response Structure (Condition C)",
        xaxis_title="% of records",
        height=max(600, 22 * n_rows + 120),
        margin=dict(l=280, r=180, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
