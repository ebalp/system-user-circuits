"""Generate a self-contained HTML report from condition C audit JSONs.

Usage:
    uv run python -m phase0_v2.calibration.audit_html_report google_gemma-4-31B-it
    uv run python -m phase0_v2.calibration.audit_html_report google_gemma-4-31B-it --timestamp 0404_0105
"""

from __future__ import annotations

import argparse
import json
import html as html_mod
import sys
from pathlib import Path

_AUDIT_BASE = Path("phase0_v2/calibration/output/condition_c_audit")


def _esc(s: str) -> str:
    return html_mod.escape(str(s))


def _load_latest_json(conflict_dir: Path) -> dict | None:
    audits = sorted(conflict_dir.glob("audit_*.json"))
    return json.loads(audits[-1].read_text()) if audits else None


def _severity_color(sev: str) -> str:
    return {
        "GREEN": "#27ae60",
        "YELLOW": "#f39c12",
        "AMBER": "#e67e22",
        "RED": "#e74c3c",
    }.get(sev, "#95a5a6")


def _severity_bg(sev: str) -> str:
    return {
        "GREEN": "#eafaf1",
        "YELLOW": "#fef9e7",
        "AMBER": "#fdebd0",
        "RED": "#fdedec",
    }.get(sev, "#f5f5f5")


def _fmt(val, width: int = 3) -> str:
    if val is None:
        return "---"
    if width == 6:
        return f"{val:.6f}"
    return f"{val:.3f}"


def _badge(sev: str) -> str:
    color = _severity_color(sev)
    return f'<span class="badge" style="background:{color}">{sev}</span>'


def _load_all(model_label: str) -> list[dict]:
    model_dir = _AUDIT_BASE / model_label
    if not model_dir.is_dir():
        return []
    results = []
    for cdir in sorted(model_dir.iterdir()):
        if not cdir.is_dir():
            continue
        data = _load_latest_json(cdir)
        if data is None:
            continue
        results.append(data)
    return results


def _extract(data: dict) -> dict:
    """Extract display fields from raw audit JSON."""
    verifier = data.get("verifier", {})
    severity_obj = data.get("severity", {})
    diag = data.get("diagnosis", {})
    pareto = data.get("pareto") or {}
    fixes = data.get("suggested_fixes", [])
    notes = data.get("notes", [])
    cond_c = diag.get("condition_C", {})

    sev = severity_obj.get("rating", "---") if isinstance(severity_obj, dict) else str(severity_obj)

    return {
        "id": data.get("conflict_id", "???"),
        "type": verifier.get("type", "---"),
        "severity": sev,
        "error_pct": diag.get("overall_error_pct", 0),
        "error_count": diag.get("overall_error_count", 0),
        "n": diag.get("overall_n", 0),
        "ba": pareto.get("ba"),
        "d_norm": pareto.get("d_norm"),
        "c_norm": pareto.get("c_norm"),
        "dist": pareto.get("distribution", "---"),
        "integrity": pareto.get("baseline_integrity", "---"),
        "feasible": pareto.get("feasible", True),
        "threshold": pareto.get("threshold"),
        "optimal_range": pareto.get("baseline_optimal_range"),
        "fixes": fixes,
        "notes": notes,
        "open_questions": data.get("open_questions", []),
        "recommended_action": data.get("recommended_action", "None"),
        "cond_c": cond_c,
        "meta_commentary": data.get("meta_commentary", {}),
        "response_structure": data.get("response_structure", {}),
    }


def _read_summary_md(model_label: str) -> str | None:
    """Read the latest summary markdown for cross-cutting/recommendations."""
    model_dir = _AUDIT_BASE / model_label
    summaries = sorted(model_dir.glob("summary_*.md"))
    if not summaries:
        return None
    return summaries[-1].read_text()


def _parse_cross_cutting(md_text: str) -> tuple[str, str]:
    """Extract cross-cutting findings and recommendations from summary MD."""
    cross_cutting = ""
    recommendations = ""
    current = None
    lines = md_text.split("\n")
    for line in lines:
        if line.startswith("## Cross-cutting findings"):
            current = "cross"
            continue
        elif line.startswith("## Recommendations"):
            current = "rec"
            continue
        elif line.startswith("## ") and current is not None:
            current = None
            continue

        if current == "cross" and not line.startswith("<!--"):
            cross_cutting += line + "\n"
        elif current == "rec" and not line.startswith("<!--"):
            recommendations += line + "\n"

    return cross_cutting.strip(), recommendations.strip()


def _md_to_html_simple(text: str) -> str:
    """Very basic markdown-to-HTML for paragraphs, bold, lists, headers."""
    import re
    lines = text.split("\n")
    html_lines = []
    in_list = False
    in_ol = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            continue

        # Headers
        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h4>{_esc(stripped[4:])}</h4>")
            continue

        # Numbered list items (e.g., "1. **foo** -- bar")
        m = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            content = m.group(2)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
            continue

        # Unordered list items
        if stripped.startswith("- "):
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = stripped[2:]
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
            continue

        # Regular paragraph
        if in_list:
            html_lines.append("</ul>")
            in_list = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False
        content = stripped
        content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
        content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
        html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    return "\n".join(html_lines)


def generate_html(model_label: str, timestamp: str | None = None) -> Path:
    all_data = _load_all(model_label)
    if not all_data:
        print(f"No audit JSONs found for {model_label}", file=sys.stderr)
        sys.exit(1)

    records = [_extract(d) for d in all_data]
    records.sort(key=lambda r: (-r["error_pct"], r["id"]))

    green = [r for r in records if r["severity"] == "GREEN"]
    yellow = [r for r in records if r["severity"] == "YELLOW"]
    amber = [r for r in records if r["severity"] == "AMBER"]
    red = [r for r in records if r["severity"] == "RED"]

    # Read cross-cutting / recommendations from summary markdown
    summary_md = _read_summary_md(model_label)
    cross_cutting_html = ""
    recommendations_html = ""
    if summary_md:
        cc_text, rec_text = _parse_cross_cutting(summary_md)
        if cc_text:
            cross_cutting_html = _md_to_html_simple(cc_text)
        if rec_text:
            recommendations_html = _md_to_html_simple(rec_text)

    # --- Build HTML ---
    parts: list[str] = []

    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Condition C Audit — {_esc(model_label)}</title>
<style>
:root {{
    --bg: #fafbfc;
    --card-bg: #ffffff;
    --border: #e1e4e8;
    --text: #24292e;
    --text-muted: #6a737d;
    --green: #27ae60;
    --yellow: #f39c12;
    --amber: #e67e22;
    --red: #e74c3c;
    --blue: #2980b9;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}}
h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
h2 {{ font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }}
h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
h4 {{ font-size: 1rem; margin: 1.2rem 0 0.4rem; color: var(--text-muted); }}
.subtitle {{ color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem; }}
.overview-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1rem 0 2rem;
}}
.card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
}}
.card .count {{ font-size: 2.2rem; font-weight: 700; }}
.card .label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.88rem;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}}
th {{
    background: #34495e;
    color: #fff;
    padding: 10px 12px;
    text-align: center;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
td {{
    padding: 8px 12px;
    text-align: center;
    border-bottom: 1px solid var(--border);
}}
tr:hover {{ background: #f6f8fa; }}
tr.severity-GREEN {{ border-left: 3px solid var(--green); }}
tr.severity-YELLOW {{ border-left: 3px solid var(--yellow); }}
tr.severity-AMBER {{ border-left: 3px solid var(--amber); }}
tr.severity-RED {{ border-left: 3px solid var(--red); }}
td.conflict-name {{ text-align: left; font-family: 'SFMono-Regular', Consolas, monospace; font-size: 0.84rem; }}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    color: #fff;
    font-size: 0.78rem;
    font-weight: 600;
}}
.fix-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}}
.fix-card h4 {{ margin: 0 0 0.3rem; color: var(--text); font-size: 0.95rem; }}
.fix-card .meta {{ font-size: 0.82rem; color: var(--text-muted); }}
.fix-card .desc {{ margin-top: 0.5rem; font-size: 0.88rem; }}
.section-box {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}}
.section-box p {{ margin: 0.4rem 0; }}
.section-box ul, .section-box ol {{ margin: 0.5rem 0 0.5rem 1.5rem; }}
.section-box li {{ margin: 0.3rem 0; }}
code {{
    background: #f1f3f5;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.85em;
}}
.detail-row {{ display: none; }}
.detail-row.active {{ display: table-row; }}
.detail-cell {{ text-align: left; padding: 12px 20px; background: #f6f8fa; }}
.detail-cell .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.detail-cell .note-item {{ font-size: 0.85rem; margin: 0.3rem 0; padding-left: 0.8rem; border-left: 2px solid var(--border); }}
.toggle-btn {{
    cursor: pointer;
    color: var(--blue);
    font-size: 0.82rem;
    background: none;
    border: none;
    text-decoration: underline;
}}
.bar-chart {{
    display: flex;
    height: 20px;
    border-radius: 4px;
    overflow: hidden;
    margin: 0.3rem 0;
}}
.bar-segment {{ height: 100%; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.82rem; }}
@media (max-width: 768px) {{
    body {{ padding: 1rem; }}
    .overview-cards {{ grid-template-columns: repeat(2, 1fr); }}
    .detail-cell .detail-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
""")

    # --- Header ---
    parts.append(f"""
<h1>Condition C Verifier Audit</h1>
<div class="subtitle">{_esc(model_label)} &mdash; {len(records)} conflicts audited</div>
""")

    # --- Overview Cards ---
    total_errors = sum(r["error_count"] for r in records)
    total_n = sum(r["n"] for r in records)
    overall_pct = (total_errors / total_n * 100) if total_n else 0

    parts.append("""<h2>Overview</h2>
<div class="overview-cards">""")
    for label, group, color in [
        ("GREEN", green, "var(--green)"),
        ("YELLOW", yellow, "var(--yellow)"),
        ("AMBER", amber, "var(--amber)"),
        ("RED", red, "var(--red)"),
    ]:
        parts.append(f"""  <div class="card">
    <div class="count" style="color:{color}">{len(group)}</div>
    <div class="label">{label}</div>
  </div>""")
    parts.append(f"""  <div class="card">
    <div class="count">{len(records)}</div>
    <div class="label">Total</div>
  </div>
  <div class="card">
    <div class="count">{overall_pct:.2f}%</div>
    <div class="label">Overall Error</div>
  </div>
</div>""")

    # --- Severity bar ---
    if records:
        g_pct = len(green) / len(records) * 100
        y_pct = len(yellow) / len(records) * 100
        a_pct = len(amber) / len(records) * 100
        r_pct = len(red) / len(records) * 100
        parts.append(f"""<div class="bar-chart" title="{len(green)} GREEN, {len(yellow)} YELLOW, {len(amber)} AMBER, {len(red)} RED">
  <div class="bar-segment" style="width:{g_pct:.1f}%;background:var(--green)"></div>
  <div class="bar-segment" style="width:{y_pct:.1f}%;background:var(--yellow)"></div>
  <div class="bar-segment" style="width:{a_pct:.1f}%;background:var(--amber)"></div>
  <div class="bar-segment" style="width:{r_pct:.1f}%;background:var(--red)"></div>
</div>""")

    # --- Conflict Health Table ---
    parts.append("""<h2>Conflict Health</h2>
<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:0.5rem">Click a row to expand details.</p>
<table id="health-table">
<thead>
<tr>
  <th>Conflict</th><th>Type</th><th>Rating</th><th>Error%</th><th>Errors</th>
  <th>BA</th><th>d_norm</th><th>c_norm</th><th>Dist</th><th>Integrity</th><th>Action</th>
</tr>
</thead>
<tbody>""")

    for i, r in enumerate(records):
        sev = r["severity"]
        action = "None"
        if r["fixes"]:
            action = r["fixes"][0].get("action_type", "Adjust verifier")
        elif r["recommended_action"] and not r["recommended_action"].lower().startswith("none"):
            action = r["recommended_action"][:30]

        err_display = f"{r['error_count']}/{r['n']}" if r["n"] else "---"

        parts.append(f"""<tr class="severity-{sev}" onclick="toggleDetail({i})" style="cursor:pointer">
  <td class="conflict-name">{_esc(r['id'])}</td>
  <td>{_esc(r['type'])}</td>
  <td>{_badge(sev)}</td>
  <td>{r['error_pct']}%</td>
  <td>{err_display}</td>
  <td>{_fmt(r['ba'])}</td>
  <td>{_fmt(r['d_norm'], 6)}</td>
  <td>{_fmt(r['c_norm'], 6)}</td>
  <td>{_esc(str(r['dist']))}</td>
  <td>{_esc(str(r['integrity']))}</td>
  <td>{_esc(action)}</td>
</tr>""")

        # Detail row (hidden by default)
        notes_html = ""
        for note in r.get("notes", []):
            notes_html += f'<div class="note-item">{_esc(str(note))}</div>\n'

        # Condition C breakdown
        cc = r.get("cond_c", {})
        cc_html = ""
        for direction in ("a_to_b", "b_to_a"):
            dir_data = cc.get(direction, {})
            if dir_data:
                labels = dir_data.get("labels", {})
                label_parts = []
                for lbl in ("followed_system", "followed_user", "followed_both", "followed_neither"):
                    cnt = labels.get(lbl, 0)
                    if cnt:
                        label_parts.append(f"{lbl}: {cnt}")
                cc_html += f'<div class="note-item"><strong>{direction}</strong>: {", ".join(label_parts) if label_parts else "no data"}</div>\n'

        # Meta-commentary
        meta = r.get("meta_commentary", {})
        meta_html = ""
        if meta:
            ma = meta.get("prevalence_a_to_b", "?")
            mb = meta.get("prevalence_b_to_a", "?")
            meta_html = f'<div class="note-item">Meta-commentary: a_to_b={ma}, b_to_a={mb}</div>'

        # Threshold info
        thresh_html = ""
        if r.get("threshold") is not None:
            opt = r.get("optimal_range")
            opt_str = f" (optimal: [{opt[0]}, {opt[1]}])" if opt and isinstance(opt, (list, tuple)) else ""
            thresh_html = f'<div class="note-item">Threshold: {r["threshold"]}{opt_str}</div>'

        parts.append(f"""<tr class="detail-row" id="detail-{i}">
  <td colspan="11" class="detail-cell">
    <div class="detail-grid">
      <div>
        <strong>Condition C Breakdown</strong>
        {cc_html}
        {meta_html}
        {thresh_html}
      </div>
      <div>
        <strong>Notes</strong>
        {notes_html if notes_html else '<div class="note-item">No notes.</div>'}
      </div>
    </div>
  </td>
</tr>""")

    parts.append("</tbody></table>")

    # --- Suggested Fixes ---
    fix_records = [r for r in records if r["fixes"]]
    if fix_records:
        parts.append("<h2>Suggested Fixes</h2>")
        for r in fix_records:
            for fix in r["fixes"]:
                desc = fix.get("description", "No description")
                complexity = fix.get("complexity", "---")
                risk = fix.get("risk_to_other_models", fix.get("risk", "---"))
                est = fix.get("estimated_error_pct", "?")
                confidence = fix.get("confidence", "---")

                parts.append(f"""<div class="fix-card">
  <h4>{_esc(r['id'])} &mdash; {_badge(r['severity'])} {r['error_pct']}% &rarr; ~{est}%</h4>
  <div class="meta">Complexity: {_esc(str(complexity))} &middot; Risk: {_esc(str(risk))} &middot; Confidence: {_esc(str(confidence))}</div>
  <div class="desc">{_esc(desc)}</div>
</div>""")

    # --- Cross-cutting findings ---
    if cross_cutting_html:
        parts.append(f"""<h2>Cross-cutting Findings</h2>
<div class="section-box">
{cross_cutting_html}
</div>""")

    # --- Recommendations ---
    if recommendations_html:
        parts.append(f"""<h2>Recommendations</h2>
<div class="section-box">
{recommendations_html}
</div>""")

    # --- Footer ---
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f"""
<footer>
  Generated {now} &middot; Phase 0 v2 Condition C Audit &middot; {_esc(model_label)}
</footer>

<script>
function toggleDetail(idx) {{
  const row = document.getElementById('detail-' + idx);
  if (row) row.classList.toggle('active');
}}
</script>
</body>
</html>""")

    html_content = "\n".join(parts)

    # Write
    ts = timestamp or "latest"
    out_path = _AUDIT_BASE / model_label / f"audit_report_{ts}.html"
    out_path.write_text(html_content)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML audit report")
    parser.add_argument("model_label", help="Model label (directory name under condition_c_audit/)")
    parser.add_argument("--timestamp", default=None, help="Timestamp for filename (default: 'latest')")
    args = parser.parse_args()
    path = generate_html(args.model_label, args.timestamp)
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()
