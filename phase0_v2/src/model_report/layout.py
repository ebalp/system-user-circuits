"""HTML layout generator for the single-model behavioral analysis report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.io as pio

from phase0_v2.src.report.data import short_model, DEFAULT_USER_STYLE, DEFAULT_SYSTEM_STYLE
from phase0_v2.src.config import load_config, ExperimentConfig
from phase0_v2.src.model_report.ordering import compute_canonical_order
from phase0_v2.src.model_report.tables import (
    CSS,
    load_model_records,
    load_calibration_data,
    load_audit_data,
    html_summary_card,
    html_conflict_table,
    html_failure_table,
    html_constraint_legend,
)
from phase0_v2.src.model_report.figures import (
    fig_label_distribution,
    fig_scr_by_system_style,
    fig_scr_by_user_style,
    fig_asymmetry_dumbbell,
    fig_system_style_constraint_heatmap,
    fig_user_style_constraint_heatmap,
    fig_baselines,
    fig_system_authority_delta,
    fig_task_effect,
    fig_task_conflict_interaction,
    fig_style_interaction_heatmap,
    fig_verifier_audit_reliability,
    fig_score_vs_threshold,
    fig_response_structure_overview,
    fig_response_structure_per_conflict,
)
from phase0_v2.calibration.response_type_analysis import load_constraint_type_map

DUS = DEFAULT_USER_STYLE
DSS = DEFAULT_SYSTEM_STYLE


# ── Helpers ────────────────────────────────────────────────────────────────

def _fig_html(fig) -> str:
    """Convert a Plotly figure to an HTML div string, or return a fallback."""
    if fig is None:
        return "<p><em>Not enough data for this chart.</em></p>"
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _note(text: str) -> str:
    """HTML paragraph for a computation note, placed before the chart."""
    return f'<p class="note">{text}</p>'


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _load_preprocessing_map() -> dict[str, list[str]]:
    """Load {conflict_id: [preprocessing_steps]} from conflicts.yaml."""
    import yaml
    yaml_path = Path(__file__).resolve().parent.parent.parent / "config" / "conflicts.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    result = {}
    for cid, info in data.items():
        if cid.startswith("_"):
            continue
        result[cid] = info.get("preprocessing", []) or []
    return result


# ── Experiment design section ──────────────────────────────────────────────

def _render_experiment_design(config: ExperimentConfig, model_id: str) -> str:
    """Render the Experiment Design section from v2 config (single-model variant)."""

    # Conditions table
    conditions_html = """
    <h3>Conditions</h3>
    <table class="design-tbl">
      <tr><th>Condition</th><th>System Prompt</th><th>User Message</th><th>system_style</th><th>user_style</th><th>Purpose</th></tr>
      <tr><td><strong>A</strong></td><td><code>{instruction}. {negative}.</code></td><td>Task only</td><td><code>bare</code></td><td><code>task_only</code></td><td>Baseline: does model follow a system constraint?</td></tr>
      <tr><td><strong>B</strong></td><td><em>empty</em></td><td><code>Please {instruction}. {task}</code></td><td><code>null</code></td><td><code>with_instruction</code></td><td>Baseline: does model follow a user constraint?</td></tr>
      <tr><td><strong>C</strong></td><td>system_style template</td><td>user_style template</td><td>varies</td><td>varies</td><td>Conflict: system vs user — measures hierarchy compliance</td></tr>
      <tr><td><strong>D</strong></td><td><em>empty</em></td><td><code>{instr1}. {negative}. Please {instr2}. {task}</code></td><td><code>bare</code></td><td><code>with_instruction</code></td><td>Recency: user-user conflict — first vs second instruction</td></tr>
    </table>
    <p class="note">A and B are capability baselines (no conflict). D isolates recency effects: if SCR in C &gt;&gt; first-instruction compliance in D, the hierarchy is real rather than a positional artifact.</p>
    """

    # System templates
    sys_rows = ""
    for name, tpl in config.system_templates.items():
        sys_rows += f"<tr><td>{name}</td><td><code>{_esc(tpl.template)}</code></td></tr>"
    sys_html = f"""
    <h3>System Templates (Styles)</h3>
    <table class="design-tbl">
      <tr><th>Style</th><th>Template</th></tr>
      {sys_rows}
    </table>
    <p class="note">Default system style for general metrics: <strong>{config.default_system_style}</strong></p>
    """

    # User templates
    usr_rows = ""
    for name, tpl in config.user_templates.items():
        usr_rows += f"<tr><td>{name}</td><td><code>{_esc(tpl.template)}</code></td></tr>"
    usr_html = f"""
    <h3>User Templates (Styles)</h3>
    <table class="design-tbl">
      <tr><th>Style</th><th>Template</th></tr>
      {usr_rows}
    </table>
    <p class="note">Default user style for general metrics: <strong>{config.default_user_style}</strong></p>
    """

    # Tasks
    task_rows = ""
    for t in config.tasks:
        task_rows += f"<tr><td>{t.id}</td><td>{_esc(t.prompt)}</td></tr>"
    tasks_html = f"""
    <h3>Tasks</h3>
    <table class="design-tbl">
      <tr><th>ID</th><th>Prompt</th></tr>
      {task_rows}
    </table>
    """

    # Sample counts note
    n_system_styles = len(config.condition_c_system_styles)
    n_user_styles = len(config.user_styles_to_test)
    n_tasks = len(config.tasks)
    instances = config.generation.instances_per_cell if config.generation else 1

    sample_note = f"""
    <h3>Sample Counts</h3>
    <p style="font-size:13px;line-height:1.7">
    Condition C varies across {n_system_styles} system styles × {n_user_styles} user styles × {n_tasks} tasks
    (instances_per_cell={instances}).
    A, B, and D use fixed style configurations.
    </p>
    """

    return f"""
<div class="design-section">
{conditions_html}
{sys_html}
{usr_html}
{tasks_html}
{sample_note}
</div>
"""


# ── Calibration panel ──────────────────────────────────────────────────────

def _render_calibration_panel(calibration_data: dict[str, dict]) -> str:
    """Build a simple HTML table showing per-conflict calibration metrics."""
    if not calibration_data:
        return "<p><em>No calibration data available.</em></p>"

    # Sort by BA descending (None values go last)
    items = sorted(
        calibration_data.items(),
        key=lambda kv: (kv[1].get("balanced_accuracy") or -1.0),
        reverse=True,
    )

    def _tier_badge(ba: float | None) -> str:
        if ba is None:
            return "N/A"
        if ba >= 0.95:
            return '<span class="tier-1">Tier 1</span>'
        if ba >= 0.80:
            return '<span class="tier-2">Tier 2</span>'
        return '<span class="tier-3">Tier 3</span>'

    rows = ""
    for conflict_id, metrics in items:
        ba = metrics.get("balanced_accuracy")
        threshold = metrics.get("threshold")
        opt_low = metrics.get("optimal_threshold_low")
        opt_high = metrics.get("optimal_threshold_high")

        ba_str = f"{ba:.3f}" if ba is not None else "—"
        thr_str = f"{threshold:.3f}" if threshold is not None else "—"
        if opt_low is not None and opt_high is not None:
            opt_str = f"[{opt_low:.3f}, {opt_high:.3f}]"
        else:
            opt_str = "—"

        rows += (
            f"<tr>"
            f'<td style="text-align:left">{conflict_id}</td>'
            f"<td>{ba_str}</td>"
            f"<td>{thr_str}</td>"
            f"<td>{opt_str}</td>"
            f"<td>{_tier_badge(ba)}</td>"
            f"</tr>"
        )

    return (
        '<table class="report">'
        "<thead>"
        "<tr><th>Conflict</th><th>BA</th><th>Threshold</th><th>Optimal Range</th><th>Tier</th></tr>"
        "</thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


# ── Page-level CSS ─────────────────────────────────────────────────────────

_PAGE_CSS = """
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    color: #2c3e50;
  }
  h1 {
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
  }
  h2 {
    color: #2c3e50;
    margin-top: 2em;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 6px;
  }
  h3 {
    color: #34495e;
    margin-top: 1.5em;
    font-size: 1.1em;
  }
  .meta {
    color: #7f8c8d;
    font-size: 14px;
    margin-bottom: 2em;
  }
  .note {
    font-size: 12px;
    color: #7f8c8d;
    font-style: italic;
    margin: 4px 0 8px 0;
    line-height: 1.5;
  }
  section {
    margin-bottom: 2em;
  }
  .def-box {
    background: #f8f9fa;
    border-left: 4px solid #3498db;
    padding: 12px 16px;
    margin: 1em 0;
    font-size: 13px;
    line-height: 1.6;
  }
  .def-box dt { font-weight: bold; color: #2c3e50; }
  .def-box dd { margin: 0 0 8px 0; color: #555; }
  .design-section {
    background: #fafbfc;
    padding: 16px;
    border-radius: 6px;
    margin: 1em 0;
  }
  .design-section h3 { margin-top: 1em; }
  .design-section h3:first-child { margin-top: 0; }
  table.design-tbl {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
    font-size: 13px;
  }
  table.design-tbl th, table.design-tbl td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
  }
  table.design-tbl th { background: #ecf0f1; font-weight: 600; }
  table.design-tbl code { background: #f4f4f4; padding: 2px 4px; font-size: 12px; }
  details > summary h2 { display: inline; cursor: pointer; }
  details > summary { list-style: none; cursor: pointer; }
  details > summary::-webkit-details-marker { display: none; }
</style>
"""


# ── Main entry point ───────────────────────────────────────────────────────

def generate_model_report(
    model_id: str,
    results_dir: str = "phase0_v2/data/results",
    output_path: str | None = None,
    config_path: str = "phase0_v2/config/experiment.yaml",
    cond_c_filter: dict[str, str] | None = None,
) -> str:
    """Generate a self-contained single-model HTML report.

    Args:
        model_id: Model identifier (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
        results_dir: Directory containing JSONL result files.
        output_path: Path to write the HTML report. Auto-generated if None.
        config_path: Path to experiment.yaml configuration file.
        cond_c_filter: Optional style filter for condition C records.
            None = no filter (all styles pooled, default).
            Example: {"system_style": "bare", "user_style": "with_instruction"}

    Returns:
        Path to the written report file.
    """
    # Normalise model_id: accept either slash or underscore form
    safe_model_id = model_id.replace("/", "_")

    # Auto-generate output path if not given
    if output_path is None:
        out = Path("phase0_v2/reports") / f"{safe_model_id}_report.html"
    else:
        out = Path(output_path)

    # 1. Load records
    records = load_model_records(results_dir, model_id)
    n_total = len(records)

    # 2. Optionally load calibration data
    calibration_data = load_calibration_data(model_id)

    # 3. Compute cross-model canonical ordering (once)
    conflict_order, system_style_order, user_style_order = compute_canonical_order(results_dir)

    # 4. Load audit data (optional)
    audit_data = load_audit_data(model_id=model_id)

    # 5. Load experiment config (graceful fallback)
    try:
        config = load_config(config_path)
    except Exception:
        config = None

    # 6. Load constraint type map and preprocessing map from conflicts.yaml
    constraint_type_map = load_constraint_type_map()
    preprocessing_map = _load_preprocessing_map()

    # 7. Derive display names
    model_name = short_model(model_id.replace("_", "/"))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 8. Build all figures
    ccf = cond_c_filter  # shorthand
    figs = {
        "label_distribution": _fig_html(fig_label_distribution(records)),
        "scr_by_system_style": _fig_html(fig_scr_by_system_style(records, cond_c_filter=ccf, system_style_order=system_style_order)),
        "scr_by_user_style": _fig_html(fig_scr_by_user_style(records, cond_c_filter=ccf, user_style_order=user_style_order)),
        "asymmetry_dumbbell": _fig_html(fig_asymmetry_dumbbell(records, cond_c_filter=ccf, conflict_order=conflict_order)),
        "system_style_constraint_heatmap": _fig_html(fig_system_style_constraint_heatmap(records, cond_c_filter=ccf, conflict_order=conflict_order, system_style_order=system_style_order)),
        "user_style_constraint_heatmap": _fig_html(fig_user_style_constraint_heatmap(records, cond_c_filter=ccf, conflict_order=conflict_order, user_style_order=user_style_order)),
        "baselines": _fig_html(fig_baselines(records, cond_c_filter=ccf, conflict_order=conflict_order)),
        "system_authority_delta": _fig_html(fig_system_authority_delta(records, cond_c_filter=ccf, conflict_order=conflict_order)),
        "task_effect": _fig_html(fig_task_effect(records, cond_c_filter=ccf)),
        "task_conflict_interaction": _fig_html(fig_task_conflict_interaction(records, cond_c_filter=ccf, conflict_order=conflict_order)),
        "style_interaction_heatmap": _fig_html(fig_style_interaction_heatmap(records, system_style_order=system_style_order, user_style_order=user_style_order)),
        "verifier_audit": _fig_html(fig_verifier_audit_reliability(audit_data, conflict_order=conflict_order)),
        "score_vs_threshold": _fig_html(fig_score_vs_threshold(records, calibration_data, cond_c_filter=ccf, conflict_order=conflict_order)),
        "response_structure_overview": _fig_html(fig_response_structure_overview(records, constraint_type_map, cond_c_filter=ccf)),
        "response_structure_per_conflict": _fig_html(fig_response_structure_per_conflict(records, constraint_type_map, preprocessing_map, cond_c_filter=ccf, conflict_order=conflict_order)),
    }

    # 8. Build tables
    summary_card = html_summary_card(records, model_name, cond_c_filter=ccf, calibration_data=calibration_data)
    conflict_table = html_conflict_table(records, calibration_data, audit_data=audit_data, cond_c_filter=ccf, conflict_order=conflict_order, preprocessing_map=preprocessing_map)
    failure_table = html_failure_table(records)
    constraint_legend = html_constraint_legend(conflict_order=conflict_order, calibration_data=calibration_data)

    # 9. Experiment design section (if config available)
    design_html = (
        _render_experiment_design(config, model_id)
        if config is not None
        else "<p><em>Config not available.</em></p>"
    )

    calibration_section = ""

    # 11. Computation notes
    if ccf:
        _filter_desc = ", ".join(f"{k}='{v}'" for k, v in ccf.items())
        _filt_note = f"Filtered to {_filter_desc}."
    else:
        _filt_note = "All styles pooled (no filter)."

    notes = dict(
        label_distribution=_note(
            "Stacked bar chart showing label proportions for each condition. "
            f"Condition C is filtered to user_style='{DUS}', system_style='{DSS}' "
            "for comparability with Condition D."
        ),
        scr_by_system_style=_note(
            f"SCR grouped by system style. Condition C. {_filt_note} "
            f"Error bars are Wilson 95% confidence intervals."
        ),
        scr_by_user_style=_note(
            f"SCR grouped by user style. Condition C. {_filt_note} "
            f"Error bars are Wilson 95% confidence intervals."
        ),
        asymmetry_dumbbell=_note(
            f"SCR split by counterbalancing direction (a_to_b vs b_to_a). "
            f"Condition C. {_filt_note} "
            f"Red connecting lines indicate |a→b − b→a| > 0.15 (option preference bias)."
        ),
        system_style_constraint_heatmap=_note(
            f"SCR per (conflict, system_style) cell. Condition C. {_filt_note} "
            f"Upper-left triangle (◸) = a→b direction, lower-right (◿) = b→a. "
            f"* marks asymmetry &gt; 0.15."
        ),
        user_style_constraint_heatmap=_note(
            f"SCR per (conflict, user_style) cell. Condition C. {_filt_note} "
            f"Upper-left triangle (◸) = a→b direction, lower-right (◿) = b→a. "
            f"* marks asymmetry &gt; 0.15."
        ),
        baselines=_note(
            "SBR = P(followed_system | Condition A). "
            "UCR baseline = P(followed_user | Condition B). "
            "Includes both counterbalancing directions (a→b and b→a). "
            "Both should be &ge; 0.95 for a conflict to be a valid test. "
            "Red '!' marks conflicts where either metric is below 0.95."
        ),
        system_authority_delta=_note(
            f"Delta = C_bare_SCR − D_first_rate per conflict. "
            f"C_bare = Condition C with system_style='{DSS}', user_style='{DUS}' (matching D's fixed styles). "
            f"D_first = P(followed_system | Condition D). "
            f"Positive delta indicates genuine system-prompt authority beyond positional advantage."
        ),
        task_effect=_note(
            f"SCR per task_id. Condition C. {_filt_note} "
            f"Sorted by SCR descending."
        ),
        task_conflict_interaction=_note(
            f"Per-task SCR distribution within each conflict-direction. {_filt_note} "
            f"Each dot is one task's SCR (styles pooled, n&asymp;25 per task). Diamond = mean. "
            f"Right panel: overdispersion ratio D = Var(task SCRs) / [p(1&minus;p)/n], "
            f"where p = mean SCR and n = samples per task. "
            f"D &asymp; 1 means task variance matches binomial sampling noise (task is irrelevant). "
            f"D &gt; 2 (red) suggests task content interacts with the conflict. "
            f"D &lt; 0.5 (blue) indicates underdispersion (style effects standardize across tasks). "
            f"Log scale so under- and overdispersion get equal visual weight."
        ),
        style_interaction_heatmap=_note(
            "SCR for each (user_style, system_style) combination. "
            "Condition C. All conflicts and directions pooled."
        ),
        verifier_audit=_note(
            "Left panel: per-direction error rates from condition C audit (Claude-as-judge). "
            "Circles = current error %, diamonds = estimated post-fix error % (best suggested fix). "
            "Labels show fix confidence/complexity. Color by severity: "
            "GREEN = 0%, YELLOW = &lt;3%, AMBER = 3&ndash;10%, RED = &gt;10%. "
            "Right panel: meta-commentary prevalence (model explicitly references conflicting instructions). "
            "Red = causes misclassification, gray = benign."
        ),
        score_vs_threshold=_note(
            "Scores normalized to constraint-a scale (b&rarr;a flipped via 1&minus;score), centered on threshold (x=0). "
            "Right = constraint a, left = constraint b. "
            "Gray bars = score distribution (Condition C). "
            "Orange bars = scores within &plusmn;0.05 of threshold (borderline). "
            "Diamonds = baseline medians with IQR (A/B, no conflict). "
            "Conflict names in red = &gt;10% of records near threshold. "
            "Right-side annotations show Pareto metrics: BA (balanced accuracy), "
            "d (discrimination cost), c (classification cost). "
            "Float-scored conflicts only."
        ),
        response_structure_overview=_note(
            f"Left: response type proportions by system-side constraint type (presence/avoidance/ambiguous). "
            f"Condition C. {_filt_note} "
            f"Clean = no refusal or metacommentary. Meta+Content = metacommentary present, no refusal prefix. "
            f"Refusal+Content = refusal prefix followed by task content. Bare Refusal = refusal only, no content. "
            f"Right: top structure patterns showing the segment sequence (refusal &rarr; metacommentary &rarr; content etc.) "
            f"with frequency. Hover for label breakdown (&rarr;sys, &rarr;usr)."
        ),
        response_structure_per_conflict=_note(
            f"Per-conflict, per-direction response type breakdown. Condition C. {_filt_note} "
            f"Right-side annotations show system constraint type (pres/avoi/ambi) and "
            f"preprocessing steps applied by the verifier (from conflicts.yaml)."
        ),
    )

    # ── Assemble HTML ──────────────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Report: {model_name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{CSS}
{_PAGE_CSS}
</head>
<body>

<h1>Model Deep-Dive: {model_name}</h1>
<p class="meta">Generated {ts} &middot; {n_total} records &middot; {model_id}</p>

<h2>1. Summary</h2>
<section>
{summary_card}
</section>

<h2>2. Label Distribution by Condition</h2>
<section>
{notes["label_distribution"]}
{figs["label_distribution"]}
</section>

<h2>3. Baselines: SBR + UCR per Conflict</h2>
<section>
{notes["baselines"]}
{figs["baselines"]}
</section>

<h2>4. User Style &times; System Style Interaction</h2>
<section>
{notes["style_interaction_heatmap"]}
{figs["style_interaction_heatmap"]}
</section>

<h2>5. Per-Conflict Metrics</h2>
<section>
{_note(
    f"Sortable table of per-conflict metrics. "
    f"SCR, Asymmetry, and Anomaly% computed on Condition C records. {_filt_note} "
    f"Click column headers to sort."
)}
{conflict_table}
</section>

<h2>6. SCR by Direction &mdash; Asymmetry</h2>
<section>
{notes["asymmetry_dumbbell"]}
{figs["asymmetry_dumbbell"]}
</section>

<h2>7. Task &times; Conflict Interaction</h2>
<section>
{notes["task_conflict_interaction"]}
{figs["task_conflict_interaction"]}
</section>

<h2>8. System Style &times; Constraint Heatmap</h2>
<section>
{notes["system_style_constraint_heatmap"]}
{figs["system_style_constraint_heatmap"]}
</section>

<h2>9. User Style &times; Constraint Heatmap</h2>
<section>
{notes["user_style_constraint_heatmap"]}
{figs["user_style_constraint_heatmap"]}
</section>

<h2>10. System Authority Delta</h2>
<section>
{notes["system_authority_delta"]}
{figs["system_authority_delta"]}
</section>

<h2>11. Task Effect</h2>
<section>
{notes["task_effect"]}
{figs["task_effect"]}
</section>

<h2>12. Response Structure Overview</h2>
<section>
{notes["response_structure_overview"]}
{figs["response_structure_overview"]}
</section>

<h2>13. Per-Conflict Response Structure</h2>
<section>
{notes["response_structure_per_conflict"]}
{figs["response_structure_per_conflict"]}
</section>

<h2>14. Score vs Threshold</h2>
<section>
{notes["score_vs_threshold"]}
{figs["score_vs_threshold"]}
</section>

<h2>15. Verifier Reliability</h2>
<section>
{notes["verifier_audit"]}
{figs["verifier_audit"]}
</section>

{calibration_section}

<details>
<summary><h2 style="display:inline">Appendix A. Constraint Legend</h2></summary>
<section>
{_note("Reference table of all conflict definitions: constraint options A and B, scorer type, and threshold.")}
{constraint_legend}
</section>
</details>

<details>
<summary><h2 style="display:inline">Appendix B. Experiment Design</h2></summary>
<section>
{design_html}
</section>
</details>

<details>
<summary><h2 style="display:inline">Appendix C. Failure Cases (Sample)</h2></summary>
<section>
{failure_table}
</section>
</details>

<details>
<summary><h2 style="display:inline">Appendix D. Metric Definitions</h2></summary>
<section>
<div class="def-box">
<dl>
  <dt>SCR &mdash; System Compliance Rate</dt>
  <dd>Fraction of Condition C trials where the model followed the <em>system</em> instruction.
      SCR = #followed_system / #total.</dd>
  <dt>UCR &mdash; User Compliance Rate</dt>
  <dd>Fraction of Condition C trials where the model followed the <em>user</em> instruction.
      UCR = #followed_user / #total.</dd>
  <dt>CR &mdash; Conflict Resolution Rate</dt>
  <dd>Fraction of Condition C trials where the model followed <em>either</em> the system or user
      instruction (i.e. it picked a side). CR = (#followed_system + #followed_user) / #total.
      A low CR means the model often produced ambiguous or unrelated responses.</dd>
  <dt>SBR &mdash; System Baseline Rate</dt>
  <dd>P(followed_system | Condition A). Measures whether the model can follow the system constraint
      in isolation (no conflict).</dd>
  <dt>Asymmetry</dt>
  <dd>|SCR<sub>A&rarr;B</sub> &minus; SCR<sub>B&rarr;A</sub>|. Measures option preference &mdash;
      whether the model favors one option over another regardless of which role (system/user) it is
      assigned to. High asymmetry suggests a built-in preference for certain outputs.</dd>
  <dt>System Authority Delta</dt>
  <dd>C_bare_SCR &minus; D_first_rate. Positive = system prompt confers genuine authority beyond
      position. Near zero = system prompt has no special authority vs. simply appearing first.</dd>
  <dt>BA &mdash; Balanced Accuracy</dt>
  <dd>Mean of sensitivity and specificity for the verifier at the configured threshold.
      Used to assess verifier quality (Tier 1: BA &ge; 0.95, Tier 2: BA &ge; 0.80).</dd>
</dl>
<p style="margin:8px 0 0;font-size:12px;color:#7f8c8d">
  <strong>Default filter for general metrics:</strong> {_filt_note}
  Style-specific charts (SCR by System Style, heatmaps) apply their own filters as noted.
  System Authority Delta always uses bare/with_instruction to match Condition D.
</p>
</div>
</section>
</details>

</body>
</html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)
