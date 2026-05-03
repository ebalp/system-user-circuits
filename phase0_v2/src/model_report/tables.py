"""Data loading and HTML table builders for the single-model behavioral analysis report."""

from __future__ import annotations

import json
import random
from pathlib import Path

from phase0_v2.src.report.data import (
    compute_scr,
    compute_ucr,
    wilson_ci,
    DEFAULT_USER_STYLE,
    DEFAULT_SYSTEM_STYLE,
    LABEL_ORDER,
    short_model,
)
from phase0_v2.src.report.tables import CSS as _BASE_CSS

# ── Preprocessing icons ──────────────────────────────────────────────────

PREPROCESSING_ICONS: dict[str, tuple[str, str]] = {
    # tag → (icon, description)
    "extract_content":          ("🧹", "Strip refusal prefixes, metacommentary segments, and helpfulness followups via shared extract_content() pipeline"),
    "refusal_prefix_only":      ("🚫", "Strip only the refusal prefix using a custom per-conflict pattern (not the shared pipeline)"),
    "use_mention_stripping":    ("🔤", "Strip quoted/emphasized constraint words where the model discusses them rather than using them (use-mention distinction)"),
    "code_fence_unwrap":        ("📦", "Extract content from markdown code blocks (```json ... ```)"),
    "markdown_prefix_stripping":("✂️", "Strip leading markdown formatting characters (#, *, _) before checking content"),
    "parenthetical_stripping":  ("🔗", "Strip parenthetical ASCII annotations (e.g. pinyin romanizations, English translations in parens)"),
}


def preprocessing_icons_html(tags: list[str]) -> str:
    """Convert preprocessing tags to a compact icon string with CSS tooltips."""
    if not tags:
        return ""
    icons = []
    for tag in tags:
        icon, desc = PREPROCESSING_ICONS.get(tag, ("❓", tag))
        tip = f"{tag}: {desc}"
        icons.append(
            f'<span class="pp-tip">{icon}'
            f'<span class="pp-tiptext">{tip}</span></span>'
        )
    return " ".join(icons)

# ── Additional CSS ─────────────────────────────────────────────────────────

CSS = _BASE_CSS + """
<style>
/* Summary card grid */
.summary-card {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 1em 0 1.5em 0;
}
.summary-card .card-item {
    background: #f4f6f9;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}
.summary-card .card-item .card-label {
    font-size: 11px;
    color: #57606a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.summary-card .card-item .card-value {
    font-size: 22px;
    font-weight: bold;
    color: #24292f;
}
.summary-card .card-item .card-sub {
    font-size: 11px;
    color: #57606a;
    margin-top: 2px;
}
.model-title {
    font-size: 20px;
    font-weight: bold;
    color: #24292f;
    margin: 0.5em 0 0.2em 0;
}
/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}
.badge-green  { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red    { background: #fee2e2; color: #991b1b; }
.badge-blue   { background: #dbeafe; color: #1e40af; }
.badge-gray   { background: #f3f4f6; color: #374151; }

/* Preprocessing icon tooltips */
.pp-tip {
    position: relative;
    cursor: help;
    font-size: 14px;
}
.pp-tip .pp-tiptext {
    visibility: hidden;
    background: #2c3e50;
    color: #fff;
    font-size: 11px;
    line-height: 1.5;
    padding: 8px 12px;
    border-radius: 4px;
    position: absolute;
    z-index: 10;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    white-space: nowrap;
    width: max-content;
    max-width: 550px;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s;
}
.pp-tip .pp-tiptext[style*="white-space:normal"] {
    white-space: normal !important;
}
.pp-tip:hover .pp-tiptext {
    visibility: visible;
    opacity: 1;
}

/* Sticky header for conflict table */
table.sortable-report {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 13px;
}
table.sortable-report th, table.sortable-report td {
    border: 1px solid #ddd;
    padding: 7px 10px;
    text-align: center;
    white-space: nowrap;
}
table.sortable-report thead th {
    position: sticky;
    top: 0;
    background: #34495e;
    color: #fff;
    cursor: pointer;
    user-select: none;
    z-index: 2;
}
table.sortable-report thead th:hover {
    background: #2c3e50;
}
table.sortable-report thead th::after {
    content: " ⇅";
    font-size: 10px;
    opacity: 0.6;
}
table.sortable-report thead th.sort-asc::after  { content: " ▲"; opacity: 1; }
table.sortable-report thead th.sort-desc::after { content: " ▼"; opacity: 1; }
table.sortable-report tr:nth-child(even) { background: #f9f9f9; }
table.sortable-report tr:hover { background: #eaf2f8; }

/* Color-coded SCR cells */
.scr-high   { background: #d1fae5 !important; color: #065f46; font-weight: bold; }
.scr-medium { background: #fef3c7 !important; color: #92400e; }
.scr-low    { background: #fee2e2 !important; color: #991b1b; }

/* Tier badges */
.tier-1 { color: #065f46; font-weight: bold; }
.tier-2 { color: #92400e; font-weight: bold; }
.tier-3 { color: #991b1b; font-weight: bold; }

/* Collapsible response details */
details.response-detail summary {
    cursor: pointer;
    font-size: 12px;
    color: #57606a;
    font-style: italic;
}
details.response-detail pre {
    margin: 4px 0 0 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 11px;
    background: #f6f8fa;
    padding: 6px 8px;
    border-radius: 4px;
    border: 1px solid #d0d7de;
    max-height: 200px;
    overflow-y: auto;
}
</style>
"""

# ── JavaScript for sortable table ─────────────────────────────────────────

_SORT_JS = """
<script>
(function() {
  function makeTableSortable(table) {
    var headers = table.querySelectorAll('thead th');
    var sortCol = -1, sortAsc = false;

    headers.forEach(function(th, colIdx) {
      th.addEventListener('click', function() {
        if (sortCol === colIdx) {
          sortAsc = !sortAsc;
        } else {
          sortCol = colIdx;
          sortAsc = false; // default descending
        }
        headers.forEach(function(h) { h.classList.remove('sort-asc', 'sort-desc'); });
        th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');

        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function(a, b) {
          var aVal = a.cells[colIdx] ? a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].innerText : '';
          var bVal = b.cells[colIdx] ? b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].innerText : '';
          var aNum = parseFloat(aVal), bNum = parseFloat(bVal);
          if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortAsc ? aNum - bNum : bNum - aNum;
          }
          return sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        });
        rows.forEach(function(row) { tbody.appendChild(row); });
      });
    });

    // Default sort: Q column descending (last column)
    if (headers.length > 1) { headers[headers.length - 1].click(); }
  }

  document.querySelectorAll('table.sortable-report').forEach(makeTableSortable);
})();
</script>
"""


# ── Loading ────────────────────────────────────────────────────────────────

def load_model_records(results_dir: str, model_id: str) -> list[dict]:
    """Load JSONL records for a single model.

    Args:
        results_dir: Directory containing JSONL result files.
        model_id: Model identifier (e.g. 'meta-llama/Llama-3.1-8B-Instruct').

    Returns:
        List of record dicts with defaults applied.
    """
    safe_model_id = model_id.replace("/", "_")
    path = Path(results_dir) / f"{safe_model_id}_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    records = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rec.setdefault("condition", "")
        rec.setdefault("constraint_type", "")
        rec.setdefault("system_style", "unknown")
        rec.setdefault("user_style", rec.get("user_template", "unknown"))
        rec.setdefault("task_id", "unknown")
        rec.setdefault("direction", "unknown")
        records.append(rec)
    return records


def load_calibration_data(model_id: str) -> dict[str, dict] | None:
    """Load threshold data from thresholds.yaml for a single model.

    Reads the per-model section if available, otherwise falls back to defaults.
    Per-model sections (from the Pareto optimizer) include BA and distribution
    metadata; the default section only has plain threshold values.

    Args:
        model_id: Model identifier (e.g. 'meta-llama/Llama-3.1-8B-Instruct').

    Returns:
        Dict keyed by conflict_id with threshold metrics, or None if no
        thresholds found.
    """
    thresholds_path = Path(__file__).resolve().parent.parent.parent / "config" / "thresholds.yaml"
    if not thresholds_path.exists():
        return None

    import yaml
    with open(thresholds_path) as f:
        data = yaml.safe_load(f)

    safe_id = model_id.replace("/", "_")
    model_section = data.get(safe_id, {})
    default_section = data.get("default", {})

    by_conflict: dict[str, dict] = {}

    # Collect all conflict IDs from both sections
    all_ids = set(default_section.keys()) | set(model_section.keys())
    all_ids.discard("_meta")

    for cid in all_ids:
        entry = model_section.get(cid)
        if isinstance(entry, dict):
            by_conflict[cid] = {
                "threshold": entry.get("threshold"),
                "balanced_accuracy": entry.get("ba"),
                "d_norm": entry.get("d_norm"),
                "c_norm": entry.get("c_norm"),
                "feasible": entry.get("feasible", True),
                "fallback": entry.get("fallback"),
                "optimal_threshold_low": None,
                "optimal_threshold_high": None,
            }
        else:
            # Fall back to default (plain float)
            default_val = default_section.get(cid)
            if default_val is not None:
                by_conflict[cid] = {
                    "threshold": float(default_val),
                    "balanced_accuracy": None,
                    "d_norm": None,
                    "c_norm": None,
                    "optimal_threshold_low": None,
                    "optimal_threshold_high": None,
                }

    return by_conflict if by_conflict else None


def _compute_bool_ba(records: list[dict]) -> dict[str, float]:
    """Compute balanced accuracy for boolean conflicts from baseline conditions.

    For condition A: correct label is followed_system (sensitivity).
    For condition B: correct label is followed_user (specificity).
    BA = (sensitivity + specificity) / 2.

    Returns dict of {conflict_id: ba} for conflicts with sufficient data.
    """
    from collections import defaultdict

    # Group by conflict_id and condition
    cond_a: dict[str, list[str]] = defaultdict(list)
    cond_b: dict[str, list[str]] = defaultdict(list)

    for r in records:
        cid = r.get("conflict_id", r.get("constraint_type", ""))
        cond = r.get("condition")
        label = r.get("label", "")
        if cond == "A":
            cond_a[cid].append(label)
        elif cond == "B":
            cond_b[cid].append(label)

    result = {}
    for cid in set(cond_a) & set(cond_b):
        a_labels = cond_a[cid]
        b_labels = cond_b[cid]
        if not a_labels or not b_labels:
            continue
        sensitivity = sum(1 for l in a_labels if l == "followed_system") / len(a_labels)
        specificity = sum(1 for l in b_labels if l == "followed_user") / len(b_labels)
        result[cid] = (sensitivity + specificity) / 2

    return result


def _float_or_none(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── Metric helpers ─────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:.0%}"


def _pct2(v: float) -> str:
    return f"{v:.1%}"


def _ci_str(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _note_p(text: str) -> str:
    return f'<p style="font-size:11px;color:#7f8c8d;margin-top:4px">{text}</p>'


def _order_sort_key(
    conflict_id: str, conflict_order: list[str] | None
) -> tuple[int, str]:
    """Sort key using conflict_order position, unknowns at end alphabetically."""
    if conflict_order is not None:
        try:
            return (conflict_order.index(conflict_id), conflict_id)
        except ValueError:
            return (len(conflict_order), conflict_id)
    return (0, conflict_id)


def _scr_class(scr: float) -> str:
    if scr > 0.20:
        return "scr-high"
    if scr > 0.05:
        return "scr-medium"
    return "scr-low"




# ── A. Summary card ────────────────────────────────────────────────────────

def _filter_cond_c(records: list[dict],
                   cond_c_filter: dict[str, str] | None = None) -> list[dict]:
    """Filter to condition C records, optionally applying style filters."""
    c = [r for r in records if r["condition"] == "C"]
    if cond_c_filter:
        for field, value in cond_c_filter.items():
            c = [r for r in c if r.get(field) == value]
    return c


def html_summary_card(records: list[dict], model_name: str,
                      cond_c_filter: dict[str, str] | None = None,
                      calibration_data: list[dict] | None = None) -> str:
    """Build a CSS Grid summary card for the single model.

    Row 1: sample counts (Total, A, B, C, D).
    Row 2: metrics (SCR, UCR, Avg Asymmetry, Anomaly%, Conflicts).
    """
    # Counts by condition
    cond_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in records:
        cond = r.get("condition", "")
        if cond in cond_counts:
            cond_counts[cond] += 1

    # Condition C with optional filter
    c_recs = _filter_cond_c(records, cond_c_filter)

    scr = compute_scr(c_recs)
    ucr = compute_ucr(c_recs)

    n_sys = sum(1 for r in c_recs if r["label"] == "followed_system")
    n_usr = sum(1 for r in c_recs if r["label"] == "followed_user")
    n_c = len(c_recs)
    ci_lo, ci_hi = wilson_ci(n_sys, n_c)
    ucr_ci_lo, ucr_ci_hi = wilson_ci(n_usr, n_c)

    # Mean per-conflict |SCR_a→b - SCR_b→a|
    conflicts_in_c = set(r["conflict_id"] for r in c_recs)
    asym_values = []
    for cid in conflicts_in_c:
        c_a2b = [r for r in c_recs if r["conflict_id"] == cid and r["direction"] == "a_to_b"]
        c_b2a = [r for r in c_recs if r["conflict_id"] == cid and r["direction"] == "b_to_a"]
        if c_a2b and c_b2a:
            asym_values.append(abs(compute_scr(c_a2b) - compute_scr(c_b2a)))
    avg_asymmetry = sum(asym_values) / len(asym_values) if asym_values else 0.0

    # Anomaly%: (followed_both + followed_neither) / n
    n_both = sum(1 for r in c_recs if r["label"] == "followed_both")
    n_neither = sum(1 for r in c_recs if r["label"] == "followed_neither")
    anomaly_pct = (n_both + n_neither) / n_c if n_c else 0.0

    # Conflict counts with tier breakdown (from calibration_data dict)
    n_conflicts = len(conflicts_in_c)
    tier_counts = {"1": 0, "2": 0, "3": 0}
    if calibration_data:
        for cid, cdata in calibration_data.items():
            ba = cdata.get("balanced_accuracy") or 0
            if ba >= 0.95:
                tier_counts["1"] += 1
            elif ba >= 0.80:
                tier_counts["2"] += 1
            else:
                tier_counts["3"] += 1
    tier_sub = f"{tier_counts['1']} T1, {tier_counts['2']} T2, {tier_counts['3']} T3" if calibration_data else ""

    # Badge colors
    scr_badge_cls = "badge-green" if scr >= 0.3 else ("badge-yellow" if scr >= 0.1 else "badge-red")
    ucr_badge_cls = "badge-green" if ucr >= 0.7 else ("badge-yellow" if ucr >= 0.5 else "badge-red")
    asym_badge_cls = "badge-red" if avg_asymmetry > 0.15 else ("badge-yellow" if avg_asymmetry > 0.08 else "badge-green")
    anom_badge_cls = "badge-green" if anomaly_pct < 0.02 else ("badge-yellow" if anomaly_pct < 0.05 else "badge-red")

    # Row 1: counts
    count_cards = [
        ("Total records", str(len(records)), "all conditions"),
        ("Condition A", str(cond_counts["A"]), "system baseline"),
        ("Condition B", str(cond_counts["B"]), "user baseline"),
        ("Condition C", str(cond_counts["C"]), "conflict trials"),
        ("Condition D", str(cond_counts["D"]), "recency control"),
    ]

    # Row 2: metrics
    metric_cards = [
        (
            "SCR",
            f'<span class="badge {scr_badge_cls}">{_pct2(scr)}</span>',
            f"95% CI {_ci_str((ci_lo, ci_hi))}",
        ),
        (
            "UCR",
            f'<span class="badge {ucr_badge_cls}">{_pct2(ucr)}</span>',
            f"95% CI {_ci_str((ucr_ci_lo, ucr_ci_hi))}",
        ),
        (
            "Avg Asymmetry",
            f'<span class="badge {asym_badge_cls}">{_pct2(avg_asymmetry)}</span>',
            f"mean |&Delta;| over {len(asym_values)} conflicts",
        ),
        (
            "Anomaly%",
            f'<span class="badge {anom_badge_cls}">{_pct2(anomaly_pct)}</span>',
            f"{n_both} both + {n_neither} neither",
        ),
        (
            "Conflicts",
            str(n_conflicts),
            tier_sub if tier_sub else "no calibration data",
        ),
    ]

    def _render_cards(cards):
        html = ""
        for label, value, sub in cards:
            html += (
                f'<div class="card-item">'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value">{value}</div>'
                f'<div class="card-sub">{sub}</div>'
                f"</div>"
            )
        return html

    filter_note = ""
    if cond_c_filter:
        filter_desc = ", ".join(f"{k}='{v}'" for k, v in cond_c_filter.items())
        filter_note = f"Condition C filtered to {filter_desc}."
    else:
        filter_note = "Condition C: all styles pooled."

    note = _note_p(
        f"{filter_note} "
        f"SCR = System Compliance Rate. UCR = User Compliance Rate. "
        f"Avg Asymmetry = mean per-conflict |SCR a→b − SCR b→a|. "
        f"Anomaly% = (followed_both + followed_neither) / n."
    )

    return (
        f'<div class="model-title">{model_name}</div>'
        f'<div class="summary-card">{_render_cards(count_cards)}</div>'
        f'<div class="summary-card">{_render_cards(metric_cards)}</div>'
        f"{note}"
    )


# ── B. Conflict table ──────────────────────────────────────────────────────

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


_RATING_COLORS = {
    "GREEN": "#2ecc71",
    "YELLOW": "#f1c40f",
    "AMBER": "#e67e22",
    "RED": "#e74c3c",
}


def _err_cell(err_pct: float | None, tooltip: str = "") -> str:
    """Build a colored table cell for an audit error percentage."""
    if err_pct is None:
        return '<td data-val="-1">\u2014</td>'
    if err_pct < 1.0:
        color = "#2ecc71"
    elif err_pct > 3.0:
        color = "#e74c3c"
    else:
        color = "#f1c40f"
    if tooltip:
        # Escape quotes for HTML attribute safety
        safe_tip = tooltip.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        return (
            f'<td data-val="{err_pct:.4f}" style="color:{color};font-weight:bold">'
            f'<span class="pp-tip">{err_pct:.1f}%'
            f'<span class="pp-tiptext" style="white-space:normal;text-align:left">{safe_tip}</span>'
            f'</span></td>'
        )
    return (
        f'<td data-val="{err_pct:.4f}" style="color:{color};font-weight:bold">'
        f"{err_pct:.1f}%</td>"
    )


def _cost_cell(calib: dict) -> str:
    """Build a colored table cell for threshold cost (d_norm + c_norm)."""
    threshold = calib.get("threshold")
    if threshold is None:
        # Boolean conflict — no cost
        return '<td data-val="-1"></td>'
    feasible = calib.get("feasible", True)
    if not feasible:
        return '<td data-val="999" style="color:#e74c3c;font-weight:bold">infeasible</td>'
    d_norm = calib.get("d_norm") or 0.0
    c_norm = calib.get("c_norm") or 0.0
    cost = d_norm + c_norm
    color = "#f1c40f" if cost > 0.02 else "#2ecc71"
    return (
        f'<td data-val="{cost:.4f}" style="color:{color};font-weight:bold">'
        f"{cost:.3f}</td>"
    )


def html_conflict_table(
    records: list[dict],
    calibration_data: dict[str, dict] | None = None,
    audit_data: dict[str, dict] | None = None,
    cond_c_filter: dict[str, str] | None = None,
    conflict_order: list[str] | None = None,
    preprocessing_map: dict[str, list[str]] | None = None,
) -> str:
    """Build a sortable per-conflict metrics table.

    Columns: Conflict | Prep | SCR | SCR a→b | SCR b→a | Asymmetry | BA |
             Δ(mean) | Cost(d+c) | Err a→b | Err b→a | Meta% | Neither% | n

    Args:
        records: All records for the model (all conditions).
        calibration_data: Optional dict from load_calibration_data().
        audit_data: Optional dict from load_audit_data().
        cond_c_filter: Optional style filter for condition C records.
        conflict_order: Optional canonical ordering for row order.
        preprocessing_map: Optional {conflict_id: [tags]} from conflicts.yaml.

    Returns:
        HTML string with a sortable table and inline JavaScript.
    """
    c_recs = _filter_cond_c(records, cond_c_filter)

    # Group by conflict_id
    by_conflict: dict[str, list[dict]] = {}
    for r in c_recs:
        cid = r.get("conflict_id", r.get("constraint_type", "unknown"))
        by_conflict.setdefault(cid, []).append(r)

    if not by_conflict:
        return "<p>No conflict data found.</p>"

    # Sort conflicts by canonical order
    conflict_ids = sorted(
        by_conflict.keys(),
        key=lambda c: _order_sort_key(c, conflict_order),
    )

    has_calib = calibration_data is not None

    # Fill in BA for boolean conflicts that aren't in thresholds.yaml
    if has_calib:
        bool_ba = _compute_bool_ba(records)
        for cid, ba in bool_ba.items():
            if cid not in calibration_data:
                calibration_data[cid] = {
                    "threshold": None,
                    "balanced_accuracy": ba,
                    "d_norm": None,
                    "c_norm": None,
                    "optimal_threshold_low": None,
                    "optimal_threshold_high": None,
                }
            elif calibration_data[cid].get("balanced_accuracy") is None:
                calibration_data[cid]["balanced_accuracy"] = ba

    # Pre-compute System Authority Delta per conflict (mean of both directions)
    c_bare = [
        r for r in records
        if r["condition"] == "C"
        and r.get("system_style") == DEFAULT_SYSTEM_STYLE
        and r.get("user_style") == DEFAULT_USER_STYLE
    ]
    cond_d = [r for r in records if r["condition"] == "D"]
    delta_by_conflict: dict[str, float | None] = {}
    for cid in conflict_ids:
        deltas = []
        for direction in ("a_to_b", "b_to_a"):
            c_dir = [r for r in c_bare
                     if r.get("conflict_id", r.get("constraint_type")) == cid
                     and r["direction"] == direction]
            d_dir = [r for r in cond_d
                     if r.get("conflict_id", r.get("constraint_type")) == cid
                     and r["direction"] == direction]
            if c_dir and d_dir:
                deltas.append(compute_scr(c_dir) - compute_scr(d_dir))
        delta_by_conflict[cid] = sum(deltas) / len(deltas) if deltas else None

    # Pre-compute 1−Clean% per conflict (condition C)
    dirty_by_conflict: dict[str, float] = {}
    for cid in conflict_ids:
        c_sub = by_conflict[cid]
        n_dirty = sum(1 for r in c_sub if _classify_response_type(r) != "clean")
        dirty_by_conflict[cid] = n_dirty / len(c_sub) if c_sub else 0.0

    # Build header
    header = (
        '<tr><th style="text-align:left">Conflict</th>'
        '<th title="Preprocessing steps applied by the verifier">Prep</th>'
        "<th>SCR</th><th>SCR a\u2192b</th><th>SCR b\u2192a</th>"
        "<th>Asym</th>"
    )
    if has_calib:
        header += "<th>BA</th>"
    header += (
        "<th>\u0394(mean)</th>"
        "<th>Cost</th>"
        "<th>Err a\u2192b</th><th>Err b\u2192a</th>"
        "<th>Meta%</th>"
        "<th>Neither%</th><th>n</th><th>Q</th>"
        "</tr>"
    )

    # Build body
    body = ""
    for cid in conflict_ids:
        c_sub = by_conflict[cid]
        n_c = len(c_sub)

        scr = compute_scr(c_sub)
        scr_cls = _scr_class(scr)

        # Directional SCR for asymmetry
        a2b = [r for r in c_sub if r["direction"] == "a_to_b"]
        b2a = [r for r in c_sub if r["direction"] == "b_to_a"]
        scr_a2b = compute_scr(a2b)
        scr_b2a = compute_scr(b2a)
        asymmetry = abs(scr_a2b - scr_b2a)

        # Neither% = followed_neither / n
        n_neither = sum(1 for r in c_sub if r["label"] == "followed_neither")
        neither_pct = n_neither / n_c if n_c else 0.0

        asym_style = ' style="color:#991b1b;font-weight:bold"' if asymmetry > 0.15 else ""

        scr_a2b_cls = _scr_class(scr_a2b)
        scr_b2a_cls = _scr_class(scr_b2a)

        pp_tags = (preprocessing_map or {}).get(cid, [])
        pp_icons = preprocessing_icons_html(pp_tags)

        body += (
            f"<tr>"
            f'<td style="text-align:left">{cid}</td>'
            f'<td data-val="{len(pp_tags)}">{pp_icons}</td>'
            f'<td class="{scr_cls}" data-val="{scr:.4f}">{_pct2(scr)}</td>'
            f'<td class="{scr_a2b_cls}" data-val="{scr_a2b:.4f}">{_pct2(scr_a2b)}</td>'
            f'<td class="{scr_b2a_cls}" data-val="{scr_b2a:.4f}">{_pct2(scr_b2a)}</td>'
            f'<td{asym_style} data-val="{asymmetry:.4f}">{asymmetry:.3f}</td>'
        )

        # BA (red if <0.95)
        if has_calib:
            calib = calibration_data.get(cid, {})
            ba = calib.get("balanced_accuracy")
            if ba is not None:
                ba_style = ' style="color:#e74c3c;font-weight:bold"' if ba < 0.95 else ""
                body += f'<td{ba_style} data-val="{ba}">{ba:.3f}</td>'
            else:
                body += '<td data-val="-1">\u2014</td>'

        # Δ(mean) — green for large positive, red for large negative
        delta = delta_by_conflict.get(cid)
        if delta is not None:
            if delta > 0.10:
                d_style = ' style="color:#2ecc71;font-weight:bold"'
            elif delta < -0.10:
                d_style = ' style="color:#e74c3c;font-weight:bold"'
            else:
                d_style = ""
            body += f'<td{d_style} data-val="{delta:.4f}">{delta:+.3f}</td>'
        else:
            body += '<td data-val="-999">\u2014</td>'

        # Cost(d+c)
        calib = calibration_data.get(cid, {}) if has_calib else {}
        body += _cost_cell(calib)

        # Err a→b, Err b→a (from audit) — with root-cause tooltips
        audit = audit_data.get(cid, {}) if audit_data else {}
        if audit:
            err_a2b_tip = audit.get("root_cause_a_to_b", "") or audit.get("root_cause", "")
            n_err_a2b = audit.get("error_count_a_to_b", 0)
            if err_a2b_tip and n_err_a2b:
                err_a2b_tip = f"{n_err_a2b} errors: {err_a2b_tip}"
            err_b2a_tip = audit.get("root_cause_b_to_a", "")
            n_err_b2a = audit.get("error_count_b_to_a", 0)
            if err_b2a_tip and n_err_b2a:
                err_b2a_tip = f"{n_err_b2a} errors: {err_b2a_tip}"
            body += _err_cell(audit.get("error_pct_a_to_b"), err_a2b_tip)
            body += _err_cell(audit.get("error_pct_b_to_a"), err_b2a_tip)
        else:
            body += '<td data-val="-1">\u2014</td><td data-val="-1">\u2014</td>'

        # Meta% — with response structure breakdown tooltip
        dirty = dirty_by_conflict.get(cid, 0.0)
        dirty_style = ' style="color:#e67e22;font-weight:bold"' if dirty > 0.05 else ""
        if audit:
            br_a = audit.get("resp_bare_refusal_a_to_b", 0)
            br_b = audit.get("resp_bare_refusal_b_to_a", 0)
            rc_a = audit.get("resp_refusal_content_a_to_b", 0)
            rc_b = audit.get("resp_refusal_content_b_to_a", 0)
            mc_a = audit.get("resp_meta_content_a_to_b", 0)
            mc_b = audit.get("resp_meta_content_b_to_a", 0)
            meta_tip = (
                f"a\u2192b: {br_a} bare refusal, {rc_a} refusal+content, {mc_a} meta+content | "
                f"b\u2192a: {br_b} bare refusal, {rc_b} refusal+content, {mc_b} meta+content"
            )
            meta_misclass = audit.get("meta_causes_misclassification", False)
            if meta_misclass:
                meta_tip += " | meta causes misclassification"
            body += (
                f'<td{dirty_style} data-val="{dirty:.4f}">'
                f'<span class="pp-tip">{_pct2(dirty)}'
                f'<span class="pp-tiptext" style="white-space:normal;text-align:left">{meta_tip}</span>'
                f'</span></td>'
            )
        else:
            body += f'<td{dirty_style} data-val="{dirty:.4f}">{_pct2(dirty)}</td>'

        # Neither%, n
        body += (
            f'<td data-val="{neither_pct:.4f}">{_pct2(neither_pct)}</td>'
            f'<td data-val="{n_c}">{n_c}</td>'
        )

        # Quality score (0–1): multiplicative penalty model
        # 1) minSCR term: linear ramp, full credit at 10%+
        min_scr = min(scr_a2b, scr_b2a)
        q_scr = min(min_scr / 0.1, 1.0)

        # 2) Error term: 1 - max(err/SCR)/0.25, clamped to [0, 1]
        #    err/SCR=0 → 1.0, err/SCR=0.25 → 0.0 (max penalty)
        err_a2b_val = audit.get("error_pct_a_to_b") if audit else None
        err_b2a_val = audit.get("error_pct_b_to_a") if audit else None
        if err_a2b_val is not None and err_b2a_val is not None:
            # err values are percentages (0-100), SCR is 0-1
            ratios = []
            if scr_a2b > 0:
                ratios.append((err_a2b_val / 100) / scr_a2b)
            if scr_b2a > 0:
                ratios.append((err_b2a_val / 100) / scr_b2a)
            q_err = max(0.0, 1.0 - max(ratios) / 0.25) if ratios else 0.0
        else:
            q_err = 0.5  # unknown audit → moderate penalty

        # 3) Threshold term: bool=1.0, feasible=f(cost), infeasible=0.2
        calib_q = calibration_data.get(cid, {}) if has_calib else {}
        is_bool = calib_q.get("threshold") is None
        if is_bool:
            q_thresh = 1.0
        elif not calib_q.get("feasible", True):
            q_thresh = 0.2
        else:
            d_n = calib_q.get("d_norm") or 0.0
            c_n = calib_q.get("c_norm") or 0.0
            q_thresh = max(0.0, 1.0 - (d_n + c_n) * 5)

        # 4) BA term: rescaled so BA=0.5 → 0, BA=1.0 → 1
        ba_q = calib_q.get("balanced_accuracy")
        q_ba = max(0.0, (ba_q - 0.5) * 2) if ba_q is not None else 0.5

        q_score = q_scr * q_err * q_thresh * q_ba

        if q_score >= 0.8:
            q_color = "#2ecc71"
        elif q_score >= 0.5:
            q_color = "#f1c40f"
        else:
            q_color = "#e74c3c"
        q_tip = (f"minSCR={q_scr:.2f} \u00d7 err={q_err:.2f}"
                 f" \u00d7 thresh={q_thresh:.2f} \u00d7 BA={q_ba:.2f}")
        body += (
            f'<td data-val="{q_score:.4f}"'
            f' style="color:{q_color};font-weight:bold;text-align:center">'
            f'<span class="pp-tip">{q_score:.2f}'
            f'<span class="pp-tiptext">{q_tip}</span></span></td>'
            "</tr>"
        )

    if cond_c_filter:
        filter_desc = ", ".join(f"{k}='{v}'" for k, v in cond_c_filter.items())
        filter_note = f"Cond C filtered to {filter_desc}"
    else:
        filter_note = "Cond C, all styles pooled"

    prep_legend = " ".join(
        f"{icon}={tag}" for tag, (icon, _) in PREPROCESSING_ICONS.items()
    )
    note = _note_p(
        f"SCR = System Compliance Rate ({filter_note}). "
        f"Prep = verifier preprocessing (hover for details): {prep_legend}. "
        f"Asym = |SCR(a\u2192b) \u2212 SCR(b\u2192a)|, red if > 0.15. "
        + (
            "BA = Balanced Accuracy. "
            if has_calib
            else ""
        )
        + "\u0394(mean) = mean System Authority Delta (C\u2212D); green >+0.10, red <\u22120.10. "
        f"Cost = d_norm+c_norm (green \u22640.02, yellow >0.02, red infeasible). "
        f"Err = audit error % (green <1%, yellow 1\u20133%, red >3%). "
        f"Meta% = metacommentary/refusal fraction. "
        f"Neither% = followed_neither fraction. "
        f"Q = quality score (0\u20131): minSCR \u00d7 error \u00d7 threshold \u00d7 BA "
        f"(green \u22650.8, yellow 0.5\u20130.8, red <0.5; hover for breakdown). "
        f"Click headers to sort."
    )

    table_id = "conflict-table"
    html = (
        f'<table class="sortable-report" id="{table_id}">'
        f"<thead>{header}</thead>"
        f"<tbody>{body}</tbody>"
        f"</table>"
        f"{note}"
        f"{_SORT_JS}"
    )
    return html


# ── C. Failure table ───────────────────────────────────────────────────────

def html_failure_table(records: list[dict], sample_size: int = 50) -> str:
    """Build a table of sampled failure cases with expandable response text.

    Args:
        records: All records for the model (all conditions).
        sample_size: Number of failures to sample (default 50).

    Returns:
        HTML string with a failure cases table.
    """
    failures = []
    for r in records:
        if r["condition"] == "C" and r["label"] != "followed_system":
            failures.append({
                "prompt_id": r["prompt_id"],
                "conflict_id": r.get("conflict_id", r.get("constraint_type", "")),
                "constraint_type": r.get("constraint_type", ""),
                "direction": r["direction"],
                "label": r["label"],
                "confidence": r["confidence"],
                "system_style": r.get("system_style", "unknown"),
                "user_style": r.get("user_style", "unknown"),
                "response": r.get("response", ""),
            })

    total_failures = len(failures)
    if not failures:
        return (
            "<p>No failure cases — all Condition C prompts followed system instructions.</p>"
        )

    if len(failures) > sample_size:
        rng = random.Random(42)
        failures = rng.sample(failures, sample_size)

    header = (
        "<tr>"
        "<th>Conflict</th>"
        "<th>Prompt ID</th>"
        "<th>Direction</th>"
        "<th>Label</th>"
        "<th>Sys Style</th>"
        "<th>User Style</th>"
        "<th>Conf.</th>"
        "<th>Response</th>"
        "</tr>"
    )

    body = ""
    for f in failures:
        resp_full = f["response"].replace("<", "&lt;").replace(">", "&gt;")
        resp_preview = resp_full[:80]
        resp_rest = resp_full[80:300]

        response_cell = (
            f'<details class="response-detail">'
            f"<summary>{resp_preview}…</summary>"
            f"<pre>{resp_full[:300]}</pre>"
            f"</details>"
            if len(resp_full) > 80
            else resp_preview
        )

        body += (
            f"<tr>"
            f'<td style="text-align:left;font-size:12px">{f["conflict_id"]}</td>'
            f'<td style="font-size:10px">{f["prompt_id"]}</td>'
            f"<td>{f['direction']}</td>"
            f"<td>{f['label']}</td>"
            f"<td>{f['system_style']}</td>"
            f"<td>{f['user_style']}</td>"
            f"<td>{f['confidence']:.2f}</td>"
            f'<td style="text-align:left;font-size:12px;min-width:220px">{response_cell}</td>'
            f"</tr>"
        )

    note = _note_p(
        f"Random sample of {len(failures)} out of {total_failures} total failure cases "
        f"(seed=42). Failure = Condition C record where model did NOT follow the system "
        f"instruction (label is 'followed_user' or 'followed_neither'). "
        f"All user/system styles included. Click response summary to expand full text (up to 300 chars)."
    )

    return (
        f'<table class="report">{header}{body}</table>'
        f"{note}"
    )


# ── D. Audit data loading ─────────────────────────────────────────────────

def load_audit_data(
    audit_dir: str = "phase0_v2/calibration/output/condition_c_audit",
    model_id: str | None = None,
) -> dict[str, dict] | None:
    """Load per-conflict audit data from JSON files.

    Looks for {audit_dir}/{safe_model_id}/{conflict_id}/audit_*.json files.
    For each conflict, picks the latest JSON by filename timestamp.

    Returns dict keyed by conflict_id with rich audit metrics,
    or None if no audit data exists.
    """
    audit_path = Path(audit_dir)
    if not audit_path.exists() or model_id is None:
        return None

    safe_id = model_id.replace("/", "_")
    model_dir = audit_path / safe_id
    if not model_dir.exists():
        return None

    result: dict[str, dict] = {}
    for conflict_dir in sorted(model_dir.iterdir()):
        if not conflict_dir.is_dir():
            continue
        # Collect both audit_*.json and merged_*.json, pick latest by timestamp
        # Sort by timestamp suffix (MMDD_HHMM) not prefix, so audit vs merged
        # doesn't affect ordering
        jsons = (
            list(conflict_dir.glob("audit_*.json"))
            + list(conflict_dir.glob("merged_*.json"))
        )
        if not jsons:
            continue
        def _ts_key(p: Path) -> tuple[str, int]:
            # Sort by timestamp, then prefer merged over audit at same timestamp
            # "audit_0323_0121.json" -> ("0323_0121", 0)
            # "merged_0323_0200.json" -> ("0323_0200", 1)
            ts = p.stem.split("_", 1)[1]
            is_merged = 1 if p.stem.startswith("merged") else 0
            return (ts, is_merged)
        latest = max(jsons, key=_ts_key)
        entry = _parse_audit_json(latest)
        if entry is not None:
            result[conflict_dir.name] = entry

    return result if result else None


def _action_text(ra) -> str:
    """Normalize recommended_action to a display string (handles dict + str)."""
    if isinstance(ra, dict):
        return ra.get("summary") or ra.get("type") or ""
    return ra if isinstance(ra, str) else ""


def _parse_audit_json(path: Path) -> dict | None:
    """Parse a single audit JSON file into the report data structure."""
    import json
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    diag = data.get("diagnosis", {})
    cond_c = diag.get("condition_C", {})
    a2b = cond_c.get("a_to_b", {})
    b2a = cond_c.get("b_to_a", {})
    meta = data.get("meta_commentary", {})

    # Find the best suggested fix (lowest estimated_error_pct)
    fixes = data.get("suggested_fixes", [])
    best_fix = None
    best_error = None
    for fix in fixes:
        est = fix.get("estimated_error_pct")
        if est is not None and (best_error is None or est < best_error):
            best_error = est
            best_fix = fix

    # Extract per-direction root cause descriptions (first cause per direction)
    root_cause_a2b = ""
    for cause in a2b.get("root_causes", []):
        root_cause_a2b = cause.get("description", "")
        break
    root_cause_b2a = ""
    for cause in b2a.get("root_causes", []):
        root_cause_b2a = cause.get("description", "")
        break

    # Response structure breakdown
    resp = data.get("response_structure", {})
    resp_a2b = resp.get("a_to_b", {})
    resp_b2a = resp.get("b_to_a", {})

    return {
        "error_pct": diag.get("overall_error_pct", 0.0),
        "rating": data.get("severity", {}).get("rating", "GREEN"),
        "root_cause_a_to_b": root_cause_a2b,
        "root_cause_b_to_a": root_cause_b2a,
        "error_pct_a_to_b": a2b.get("error_pct", 0.0),
        "error_pct_b_to_a": b2a.get("error_pct", 0.0),
        "error_count_a_to_b": a2b.get("misclassified", 0),
        "error_count_b_to_a": b2a.get("misclassified", 0),
        "n_a_to_b": a2b.get("n", 0),
        "n_b_to_a": b2a.get("n", 0),
        "estimated_fix_error_pct": best_error,
        "estimated_fix_pareto": best_fix.get("estimated_pareto") if best_fix else None,
        "fix_confidence": best_fix.get("confidence") if best_fix else None,
        "fix_complexity": best_fix.get("complexity") if best_fix else None,
        "fix_description": best_fix.get("description") if best_fix else None,
        "meta_commentary_a_to_b": meta.get("prevalence_a_to_b", 0),
        "meta_commentary_b_to_a": meta.get("prevalence_b_to_a", 0),
        "meta_causes_misclassification": meta.get("causes_misclassification", False),
        "recommended_action": _action_text(data.get("recommended_action", "")),
        "resp_bare_refusal_a_to_b": resp_a2b.get("bare_refusal", 0),
        "resp_bare_refusal_b_to_a": resp_b2a.get("bare_refusal", 0),
        "resp_refusal_content_a_to_b": resp_a2b.get("refusal_content", 0),
        "resp_refusal_content_b_to_a": resp_b2a.get("refusal_content", 0),
        "resp_meta_content_a_to_b": resp_a2b.get("meta_content", 0),
        "resp_meta_content_b_to_a": resp_b2a.get("meta_content", 0),
        "stripping_needed": resp.get("stripping_needed", False),
    }


# ── E. Constraint legend ──────────────────────────────────────────────────

def html_constraint_legend(
    conflict_order: list[str] | None = None,
    calibration_data: dict[str, dict] | None = None,
) -> str:
    """Build a constraint legend table from conflicts.yaml metadata.

    Returns HTML table: Conflict, Constraint A, Constraint B, Type, Scorer, Threshold.
    """
    from phase0_v2.config.conflict_config import get_conflict_meta, get_all_conflict_ids

    all_ids = get_all_conflict_ids()
    if not all_ids:
        return "<p>No conflict metadata found.</p>"

    entries: list[dict] = []
    for cid in all_ids:
        meta = get_conflict_meta(cid)
        if meta is None:
            continue
        entries.append({
            "conflict_id": cid,
            "constraint_a": meta.get("constraint_a", "\u2014"),
            "constraint_b": meta.get("constraint_b", "\u2014"),
            "type": meta.get("type", "\u2014"),
            "scorer": meta.get("verifier_logic", "\u2014"),
        })

    if not entries:
        return "<p>No conflict metadata found.</p>"

    entries.sort(key=lambda e: _order_sort_key(e["conflict_id"], conflict_order))

    header = (
        '<tr><th style="text-align:left">Conflict</th>'
        '<th style="text-align:left">Constraint A</th>'
        '<th style="text-align:left">Constraint B</th>'
        "<th>Type</th>"
        '<th style="text-align:left">Scorer</th>'
        "<th>Threshold</th></tr>"
    )
    body = ""
    for e in entries:
        cid = e["conflict_id"]
        thresh_str = "\u2014"
        if calibration_data and cid in calibration_data:
            t = calibration_data[cid].get("threshold")
            if t is not None:
                thresh_str = f"{t:.3f}"
        body += (
            f"<tr>"
            f'<td style="text-align:left">{cid}</td>'
            f'<td style="text-align:left">{e["constraint_a"]}</td>'
            f'<td style="text-align:left">{e["constraint_b"]}</td>'
            f'<td>{e["type"]}</td>'
            f'<td style="text-align:left">{e["scorer"]}</td>'
            f"<td>{thresh_str}</td>"
            f"</tr>"
        )

    note = _note_p(
        "Constraint metadata from phase0_v2/config/conflicts.yaml."
    )
    return f'<table class="report">{header}{body}</table>{note}'
