"""Assemble the full HTML report from figures and tables."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

import plotly.io as pio

from .data import load_all_records, short_model, DEFAULT_USER_STYLE, DEFAULT_SYSTEM_STYLE, _model_sort_key
from .figures import (
    fig_scr_by_system_style,
    fig_constraint_directions,
    fig_heatmap,
    fig_by_condition,
    fig_cross_model_scr,
    fig_task_effect,
    fig_user_style_constraint_heatmap,
    fig_user_style_system_style_heatmap,
    fig_sbr_by_conflict,
    fig_ucr_by_conflict,
    fig_system_authority_delta,
)
from .tables import CSS, html_summary_table, html_failure_table
from phase0_v2.src.config import load_config, ExperimentConfig
from phase0_v2.src.metrics import MetricsCalculator

DUS = DEFAULT_USER_STYLE
DSS = DEFAULT_SYSTEM_STYLE


def _fig_html(fig, full_html: bool = False) -> str:
    if fig is None:
        return "<p><em>Not enough data for this chart.</em></p>"
    return pio.to_html(fig, full_html=full_html, include_plotlyjs=False)


def _note(text: str) -> str:
    """HTML paragraph for a computation note, placed before the chart."""
    return f'<p class="note">{text}</p>'


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_experiment_design(config: ExperimentConfig, models: list[str]) -> str:
    """Render the Experiment Design section from v2 config."""

    # Conditions
    conditions_html = """
    <h3>Conditions</h3>
    <table class="design-tbl">
      <tr><th>Condition</th><th>System Prompt</th><th>User Message</th><th>system_style</th><th>user_style</th><th>Purpose</th></tr>
      <tr><td><strong>A</strong></td><td><code>{instruction}. {negative}.</code></td><td>Task only</td><td><code>bare</code></td><td><code>task_only</code></td><td>Baseline: does model follow a system constraint?</td></tr>
      <tr><td><strong>B</strong></td><td><em>empty</em></td><td><code>Please {instruction}. {task}</code></td><td><code>null</code></td><td><code>with_instruction</code></td><td>Baseline: does model follow a user constraint?</td></tr>
      <tr><td><strong>C</strong></td><td>system_style template</td><td>user_style template</td><td>varies</td><td>varies</td><td>Conflict: system vs user -- measures hierarchy compliance</td></tr>
      <tr><td><strong>D</strong></td><td><em>empty</em></td><td><code>{instr1}. {negative}. Please {instr2}. {task}</code></td><td><code>bare</code></td><td><code>with_instruction</code></td><td>Recency: user-user conflict -- first vs second instruction</td></tr>
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

    # Models
    models_html = f"""
    <h3>Models Evaluated</h3>
    <ul>{"".join(f"<li>{m}</li>" for m in models)}</ul>
    """

    # Sample counts note
    n_system_styles = len(config.condition_c_system_styles)
    n_user_styles = len(config.user_styles_to_test)
    n_tasks = len(config.tasks)

    # We don't have n_pairs directly in v2 config, estimate from data or show formula generically
    sample_note = f"""
    <h3>Sample Counts (per model)</h3>
    <p style="font-size:13px;line-height:1.7">
    Condition C varies across {n_system_styles} system styles x {n_user_styles} user styles x {n_tasks} tasks
    (instances_per_cell={config.generation.instances_per_cell if config.generation else 1}).
    A, B, and D use fixed style configurations.
    </p>
    <p class="note">
    The summary metrics (Section 1) filter Condition C to user_style='{config.default_user_style}' and
    system_style='{config.default_system_style}'.
    </p>
    """

    return f"""
<div class="design-section">
{conditions_html}
{sys_html}
{usr_html}
{tasks_html}
{models_html}
{sample_note}
</div>
"""


def generate_report(
    results_dir: str = "data/results",
    output_path: str = "reports/report.html",
    config_path: str = "config/experiment.yaml",
) -> str:
    """Generate a self-contained interactive HTML report."""
    records = load_all_records(results_dir)
    if not records:
        raise RuntimeError(f"No JSONL records found in {results_dir}")

    # Load config for experiment design section
    try:
        config = load_config(config_path)
    except Exception:
        config = None

    models = sorted(set(short_model(r["model"]) for r in records), key=_model_sort_key)
    n_total = len(records)
    n_c = sum(1 for r in records if r["condition"] == "C")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    figs = {
        "scr_by_system_style": _fig_html(fig_scr_by_system_style(records)),
        "constraint_directions": _fig_html(fig_constraint_directions(records)),
        "heatmap": _fig_html(fig_heatmap(records)),
        "by_condition": _fig_html(fig_by_condition(records)),
        "cross_model_scr": _fig_html(fig_cross_model_scr(records)),
        "task_effect": _fig_html(fig_task_effect(records)),
        "user_style_constraint": _fig_html(fig_user_style_constraint_heatmap(records)),
        "user_style_system_style": _fig_html(fig_user_style_system_style_heatmap(records)),
    }

    # Compute ModelMetrics for triangle heatmaps
    calculator = MetricsCalculator(records)
    all_metrics = calculator.compute_all()

    figs["sbr_by_conflict"] = _fig_html(fig_sbr_by_conflict(all_metrics))
    figs["ucr_by_conflict"] = _fig_html(fig_ucr_by_conflict(all_metrics))
    figs["system_authority_delta"] = _fig_html(fig_system_authority_delta(all_metrics))

    summary_tbl = html_summary_table(records)
    failure_tbl = html_failure_table(records)

    # Experiment design section (if config available)
    design_html = ""
    if config:
        design_html = _render_experiment_design(config, models)

    # ── Computation notes (HTML, placed before each chart) ────────────
    notes = dict(
        cross_model_scr=_note(
            f"SCR = #followed_system / #total across Condition C records "
            f"with user_style='{DUS}', system_style='{DSS}'. "
            f"Green >= 0.7, orange = marginal, red < 0.3."
        ),
        by_condition=_note(
            "A and B are capability controls (no conflict): if A ~ 100% followed_system and B ~ 100% followed_user, "
            "the model can follow individual constraints correctly. "
            f"Condition C is filtered to system_style='{DSS}' + user_style='{DUS}' to match D's format for a fair comparison. "
            "If C shows a higher proportion of followed_system (green) than D, it suggests a genuine system/user hierarchy "
            "circuit rather than a positional recency artifact."
        ),
        scr_by_system_style=_note(
            f"SCR = #followed_system / #total for Condition C records. "
            f"Filtered to user_style='{DUS}'. "
            f"Averaged across all constraint types and tasks."
        ),
        constraint_directions=_note(
            f"SCR per (model, constraint_type) split by counterbalancing direction. "
            f"Filtered to Condition C, user_style='{DUS}', system_style='{DSS}'. "
            f"* marks cells where |A->B - B->A| > 0.15 (option preference bias)."
        ),
        heatmap=_note(
            f"SCR per (system_style, constraint_type) cell. "
            f"Filtered to Condition C, user_style='{DUS}'. "
            f"Both directions pooled. n = sample count per cell."
        ),
        user_style_constraint=_note(
            f"SCR per (constraint_type, user_style) cell. "
            f"Filtered to Condition C, system_style='{DSS}'. "
            f"Use the dropdown to select a model or view all models averaged."
        ),
        user_style_system_style=_note(
            f"SCR per (user_style, system_style) cell. "
            f"Filtered to Condition C. "
            f"Use the dropdowns to select a model and/or constraint type."
        ),
        task_effect=_note(
            f"SCR per task_id per model. "
            f"Filtered to Condition C, user_style='{DUS}', system_style='{DSS}'."
        ),
        sbr_by_conflict=_note(
            "System Baseline Rate (SBR) per (model, constraint_type), split by counterbalancing direction. "
            "SBR = P(followed_system | Condition A). Shows whether the model can follow each constraint "
            "in isolation (no conflict). * marks cells where |dir_a - dir_b| > 0.15."
        ),
        ucr_by_conflict=_note(
            "User Compliance Rate (UCR) per (model, constraint_type), split by counterbalancing direction. "
            "UCR = P(followed_user | Condition B). Shows whether the model can follow each user constraint "
            "in isolation (no conflict). * marks cells where |dir_a - dir_b| > 0.15."
        ),
        system_authority_delta=_note(
            "System Authority Delta = C_bare_SCR - D_first_rate per (model, constraint_type). "
            "Positive values (red) mean the system prompt adds genuine authority beyond positional advantage. "
            "Near-zero (white) means the system prompt confers no special authority. "
            "* marks cells where |dir_a - dir_b| > 0.15."
        ),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 0 v2 -- Behavioral Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{CSS}
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; color: #2c3e50; }}
  h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 2em; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; }}
  h3 {{ color: #34495e; margin-top: 1.5em; font-size: 1.1em; }}
  .meta {{ color: #7f8c8d; font-size: 14px; margin-bottom: 2em; }}
  .note {{ font-size: 12px; color: #7f8c8d; font-style: italic; margin: 4px 0 8px 0;
           line-height: 1.5; }}
  section {{ margin-bottom: 2em; }}
  .def-box {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 12px 16px;
              margin: 1em 0; font-size: 13px; line-height: 1.6; }}
  .def-box dt {{ font-weight: bold; color: #2c3e50; }}
  .def-box dd {{ margin: 0 0 8px 0; color: #555; }}
  .design-section {{ background: #fafbfc; padding: 16px; border-radius: 6px; margin: 1em 0; }}
  .design-section h3 {{ margin-top: 1em; }}
  .design-section h3:first-child {{ margin-top: 0; }}
  table.design-tbl {{ border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 13px; }}
  table.design-tbl th, table.design-tbl td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  table.design-tbl th {{ background: #ecf0f1; font-weight: 600; }}
  table.design-tbl code {{ background: #f4f4f4; padding: 2px 4px; font-size: 12px; }}
  .design-section ul {{ margin: 0.5em 0; padding-left: 1.5em; }}
  .design-section li {{ margin: 2px 0; }}
</style>
</head>
<body>

<h1>Phase 0 v2 -- Behavioral Analysis Report</h1>
<p class="meta">Generated {ts} &middot; {n_total} records &middot; {len(models)} models &middot; {n_c} conflict trials (Cond. C)</p>

<h2>Experiment Design</h2>
{design_html if design_html else "<p><em>Config not available.</em></p>"}

<h2>Metric Definitions</h2>
<div class="def-box">
<dl>
  <dt>SCR -- System Compliance Rate</dt>
  <dd>Fraction of Condition C trials where the model followed the <em>system</em> instruction.
      SCR = #followed_system / #total.</dd>
  <dt>UCR -- User Compliance Rate</dt>
  <dd>Fraction of Condition C trials where the model followed the <em>user</em> instruction.
      UCR = #followed_user / #total.</dd>
  <dt>CR -- Conflict Resolution Rate</dt>
  <dd>Fraction of Condition C trials where the model followed <em>either</em> the system or user
      instruction (i.e. it picked a side). CR = (#followed_system + #followed_user) / #total.
      A low CR means the model often produced ambiguous or unrelated responses ('followed_neither').</dd>
  <dt>Asymmetry</dt>
  <dd>|SCR<sub>A->B</sub> - SCR<sub>B->A</sub>|. Measures option preference -- whether the model
      favors one option over another regardless of which role (system/user) it's assigned to.
      High asymmetry suggests the model has a built-in preference for certain outputs (e.g., always preferring English over Spanish).</dd>
  <dt>SBR -- System Baseline Rate</dt>
  <dd>P(followed_system | Condition A). Measures whether the model can follow the system constraint
      in isolation (no conflict).</dd>
  <dt>System Authority Delta</dt>
  <dd>C_bare_SCR - D_first_rate. Positive = system prompt confers genuine authority beyond position.
      Near zero = system prompt has no special authority vs. simply appearing first in user message.</dd>
</dl>
<p style="margin:8px 0 0;font-size:12px;color:#7f8c8d">
  <strong>Default filters for general metrics:</strong> user_style='{DUS}',
  system_style='{DSS}'. This avoids mixing effects from different user templates
  or system styles. Plots that deviate from these defaults note it explicitly.
</p>
</div>

<h2>1. Summary Metrics</h2>
<section>{summary_tbl}</section>

<h2>2. Overall SCR by Model</h2>
<section>
{notes["cross_model_scr"]}
{figs["cross_model_scr"]}
</section>

<h2>3. Label Distribution by Condition</h2>
<section>
{notes["by_condition"]}
{figs["by_condition"]}
</section>

<h2>4. SCR vs System Style</h2>
<section>
{notes["scr_by_system_style"]}
{figs["scr_by_system_style"]}
</section>

<h2>5. SCR by Constraint Type &amp; Direction</h2>
<section>
{notes["constraint_directions"]}
{figs["constraint_directions"]}
</section>

<h2>6. SCR Heatmap: System Style x Constraint</h2>
<section>
{notes["heatmap"]}
{figs["heatmap"]}
</section>

<h2>7. User Style x Constraint Type</h2>
<section>
{notes["user_style_constraint"]}
{figs["user_style_constraint"]}
</section>

<h2>8. User Style x System Style</h2>
<section>
{notes["user_style_system_style"]}
{figs["user_style_system_style"]}
</section>

<h2>9. Task Effect</h2>
<section>
{notes["task_effect"]}
{figs["task_effect"]}
</section>

<h2>10. Failure Cases (Condition C) -- Sample</h2>
<section>{failure_tbl}</section>

<h2>11. SBR by Constraint &amp; Direction</h2>
<section>
{notes["sbr_by_conflict"]}
{figs["sbr_by_conflict"]}
</section>

<h2>12. UCR by Constraint &amp; Direction</h2>
<section>
{notes["ucr_by_conflict"]}
{figs["ucr_by_conflict"]}
</section>

<h2>13. System Authority Delta</h2>
<section>
{notes["system_authority_delta"]}
{figs["system_authority_delta"]}
</section>

</body>
</html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)
