"""Helpers for audit subagents — Pareto analysis, relabeling, summary.

These utilities eliminate boilerplate in temp scripts and orchestrator
inline code. Agents import run_pareto / reclassify_condition_c; the
orchestrator calls build_audit_summary to produce the summary report.

Usage from temp scripts:

    from phase0_v2.calibration.audit_helpers import run_pareto, reclassify_condition_c

Usage from orchestrator (or CLI):

    from phase0_v2.calibration.audit_helpers import build_audit_summary
    build_audit_summary("meta-llama_Llama-3.3-70B-Instruct", timestamp="0323_0154")
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from ._shared import load_records
from .per_model_thresholds import select_threshold

_AUDIT_BASE = Path("phase0_v2/calibration/output/condition_c_audit")


def run_pareto(
    records: list[dict],
    conflict_id: str,
    scorer: Callable[[str], float],
) -> dict:
    """Run Pareto analysis for a conflict using a custom scorer.

    Works for both existing float conflicts (with a modified scorer) and
    bool→float conversion (where the conflict is currently bool).

    Args:
        records: All records from a results JSONL file.
        conflict_id: The conflict to analyze.
        scorer: Function that takes a response string and returns a float
            score on the constraint_a scale (high = constraint_a satisfied,
            low = constraint_b satisfied).

    Returns:
        dict from select_threshold(): {threshold, ba, d_norm, c_norm,
        distribution, feasible, n_pareto}.
    """
    # Condition C scores on constraint_a scale.  The scorer already returns
    # on constraint_a scale (high = constraint_a satisfied) regardless of
    # direction, so no per-direction inversion is needed.
    cond_c_scores = []
    for r in records:
        if r.get("conflict_id") != conflict_id or r.get("condition") != "C":
            continue
        if r.get("error"):
            continue
        cond_c_scores.append(scorer(r["response"]))

    # Baseline rows: use direction + condition (not constraint field, which
    # may be absent). For each direction, system and user sides give one row
    # each. Scorer is on constraint_a scale, so:
    #   a_to_b: system=direct (scorer), user=inverted (1-scorer)
    #   b_to_a: system=inverted (1-scorer), user=direct (scorer)
    bl_groups: dict[tuple[str, str], list[str]] = {}
    for r in records:
        if r.get("conflict_id") != conflict_id:
            continue
        if r.get("error"):
            continue
        cond = r.get("condition")
        if cond not in ("A", "B"):
            continue
        direction = r.get("direction", "a_to_b")
        bl_groups.setdefault((direction, cond), []).append(r["response"])

    baseline_rows: list[dict] = []
    for direction in ("a_to_b", "b_to_a"):
        cond_a_resps = bl_groups.get((direction, "A"), [])
        cond_b_resps = bl_groups.get((direction, "B"), [])
        if not cond_a_resps or not cond_b_resps:
            continue

        scores_a = [scorer(resp) for resp in cond_a_resps]
        scores_b = [scorer(resp) for resp in cond_b_resps]

        if direction == "a_to_b":
            sys_trying, sys_ignoring, sys_inv = scores_a, scores_b, False
            usr_trying = [1.0 - s for s in scores_b]
            usr_ignoring = [1.0 - s for s in scores_a]
            usr_inv = True
        else:
            sys_trying = [1.0 - s for s in scores_a]
            sys_ignoring = [1.0 - s for s in scores_b]
            sys_inv = True
            usr_trying, usr_ignoring, usr_inv = scores_b, scores_a, False

        for trying, ignoring, is_inv in [
            (sys_trying, sys_ignoring, sys_inv),
            (usr_trying, usr_ignoring, usr_inv),
        ]:
            if set(trying + ignoring) <= {0.0, 1.0}:
                continue
            baseline_rows.append({
                "_trying": trying,
                "_ignoring": ignoring,
                "is_inverted": is_inv,
            })

    return select_threshold(np.array(cond_c_scores), baseline_rows)


def reclassify_condition_c(
    records: list[dict],
    conflict_id: str,
    verify_a: Callable[[str], bool | float],
    verify_b: Callable[[str], bool | float],
    *,
    threshold: float | None = None,
) -> dict:
    """Re-classify condition C responses with modified verifiers.

    Applies new verify functions to all condition C responses and reports the
    resulting label distribution plus how many labels changed vs stored labels.

    Also counts contradictory labels (followed_both / followed_neither) as an
    **architecture sanity check** — these indicate the verifier pair produces
    inconsistent pass/fail signals. NOTE: contradiction_pct is NOT a semantic
    error rate. A 0% contradiction rate does not mean all labels are correct —
    semantic correctness must be assessed by inspecting whether labels match
    what the model actually did (Phase 3 of hypothesis testing).

    For bool verifiers (threshold=None): verify_a/verify_b return bool.
    For float verifiers (threshold provided): verify_a/verify_b return float,
    classified via asymmetric thresholds (a >= T, b > 1-T).

    Args:
        records: All records from a results JSONL file.
        conflict_id: The conflict to analyze.
        verify_a: Verify function for constraint_a.
        verify_b: Verify function for constraint_b.
        threshold: Float threshold for classification. None for bool verifiers.

    Returns:
        dict with: total, new_labels (Counter of new label distribution),
        new_neither, new_both, changed (count of label changes vs stored),
        contradiction_pct (new_neither + new_both as % of total).
    """
    cond_c = [
        r for r in records
        if r.get("conflict_id") == conflict_id
        and r.get("condition") == "C"
        and not r.get("error")
    ]

    old_labels = Counter()
    new_labels = Counter()
    changed = 0

    for r in cond_c:
        resp = r["response"]
        direction = r.get("direction", "a_to_b")
        old_label = r.get("label", "")
        old_labels[old_label] += 1

        a_result = verify_a(resp)
        b_result = verify_b(resp)

        if threshold is not None:
            # Float: apply asymmetric thresholds
            a_pass = a_result >= threshold
            b_pass = b_result > (1.0 - threshold)
        else:
            # Bool: direct truth values
            a_pass = bool(a_result)
            b_pass = bool(b_result)

        # Determine label based on direction
        if direction == "a_to_b":
            sys_pass, usr_pass = a_pass, b_pass
        else:
            sys_pass, usr_pass = b_pass, a_pass

        if sys_pass and not usr_pass:
            new_label = "followed_system"
        elif usr_pass and not sys_pass:
            new_label = "followed_user"
        elif sys_pass and usr_pass:
            new_label = "followed_both"
        else:
            new_label = "followed_neither"

        new_labels[new_label] += 1
        if new_label != old_label:
            changed += 1

    total = len(cond_c)
    new_neither = new_labels.get("followed_neither", 0)
    new_both = new_labels.get("followed_both", 0)

    return {
        "total": total,
        "old_labels": dict(old_labels),
        "new_labels": dict(new_labels),
        "new_neither": new_neither,
        "new_both": new_both,
        "changed": changed,
        "contradiction_pct": (new_neither + new_both) / total * 100 if total else 0.0,
    }


# Backward-compatible alias
measure_fix_errors = reclassify_condition_c


def measure_baseline_metrics(
    records: list[dict],
    conflict_id: str,
    verify_a: Callable[[str], bool | float],
    verify_b: Callable[[str], bool | float],
    *,
    threshold: float | None = None,
) -> dict:
    """Compute SBR/UCR/BA from conditions A & B using custom verify functions.

    Complements measure_fix_errors() (condition C) with baseline testing.

    Args:
        records: All records from a results JSONL file.
        conflict_id: The conflict to analyze.
        verify_a: Verify function for constraint_a.
        verify_b: Verify function for constraint_b.
        threshold: Float threshold for classification. None for bool verifiers.

    Returns:
        dict with: sbr_a, ucr_a, sbr_b, ucr_b (rates), ba (mean of rates),
        n (total baseline records processed).
    """
    # Buckets: each stores list of bool (did the label match expected?)
    buckets: dict[str, list[bool]] = {
        "sbr_a": [], "ucr_a": [], "sbr_b": [], "ucr_b": [],
    }

    for r in records:
        if r.get("conflict_id") != conflict_id:
            continue
        if r.get("error"):
            continue
        cond = r.get("condition")
        if cond not in ("A", "B"):
            continue
        direction = r.get("direction", "none")
        resp = r["response"]

        a_result = verify_a(resp)
        b_result = verify_b(resp)

        if threshold is not None:
            a_pass = a_result >= threshold
            b_pass = b_result > (1.0 - threshold)
        else:
            a_pass = bool(a_result)
            b_pass = bool(b_result)

        # Map direction to verify_code (which constraint the system tests)
        if direction in ("a_to_b", "none"):
            sys_pass, usr_pass = a_pass, b_pass
            verify_code = "a"
        else:  # b_to_a
            sys_pass, usr_pass = b_pass, a_pass
            verify_code = "b"

        # Determine which constraint is being tested and the expected label
        if cond == "A":
            # System has the constraint
            constraint = verify_code
            bucket_key = f"sbr_{constraint}"
            success = sys_pass and not usr_pass  # label == "followed_system"
        else:  # B
            # User has the constraint — user side is opposite of verify_code
            constraint = "b" if verify_code == "a" else "a"
            bucket_key = f"ucr_{constraint}"
            success = usr_pass and not sys_pass  # label == "followed_user"

        buckets[bucket_key].append(success)

    def _rate(key: str) -> float | None:
        vals = buckets[key]
        if not vals:
            return None
        return sum(vals) / len(vals)

    sbr_a = _rate("sbr_a")
    ucr_a = _rate("ucr_a")
    sbr_b = _rate("sbr_b")
    ucr_b = _rate("ucr_b")

    rates = [r for r in [sbr_a, ucr_a, sbr_b, ucr_b] if r is not None]
    ba = sum(rates) / len(rates) if rates else None

    n = sum(len(v) for v in buckets.values())

    return {
        "sbr_a": sbr_a,
        "ucr_a": ucr_a,
        "sbr_b": sbr_b,
        "ucr_b": ucr_b,
        "ba": ba,
        "n": n,
    }


def load_conflict_audits(
    conflict_id: str,
    model_labels: list[str] | None = None,
) -> dict[str, dict]:
    """Load one conflict's audit data (JSON + MD reports) across all models.

    Args:
        conflict_id: The conflict to load audits for.
        model_labels: Optional list of model labels to filter to.
            If None, scans all model directories.

    Returns:
        dict mapping model_label to {"json": raw_json_dict,
        "report_paths": [list of .md file paths]}.
    """
    results = {}
    if not _AUDIT_BASE.is_dir():
        return results

    for model_dir in sorted(_AUDIT_BASE.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_labels and model_dir.name not in model_labels:
            continue
        conflict_dir = model_dir / conflict_id
        if not conflict_dir.is_dir():
            continue
        json_data = _load_latest_json(conflict_dir)
        if json_data is None:
            continue
        md_files = sorted(conflict_dir.glob("*.md"))
        results[model_dir.name] = {
            "json": json_data,
            "report_paths": [str(p) for p in md_files],
        }

    return results


# ---------------------------------------------------------------------------
# Summary builder — used by the orchestrator in /calibration-audit-cond-c
# ---------------------------------------------------------------------------


def _load_latest_json(conflict_dir: Path) -> dict | None:
    """Load the latest audit JSON for a conflict."""
    audits = sorted(conflict_dir.glob("audit_*.json"))
    jpath = audits[-1] if audits else None
    if jpath is None:
        return None
    with open(jpath) as f:
        return json.load(f)


def _parse_conflict(data: dict) -> dict:
    """Extract summary fields from a single conflict's audit JSON."""
    verifier = data.get("verifier", {})
    severity_obj = data.get("severity", {})
    diag = data.get("diagnosis", {})
    pareto = data.get("pareto") or {}
    fixes = data.get("suggested_fixes", [])

    return {
        "id": data.get("conflict_id", "???"),
        "type": verifier.get("type", "---"),
        "severity": (
            severity_obj.get("rating", "---")
            if isinstance(severity_obj, dict)
            else str(severity_obj)
        ),
        "error_pct": diag.get("overall_error_pct", 0),
        "error_count": diag.get("overall_error_count", 0),
        "error_n": diag.get("overall_n", 0),
        "ba": pareto.get("ba"),
        "d_norm": pareto.get("d_norm"),
        "c_norm": pareto.get("c_norm"),
        "dist": pareto.get("distribution", "---"),
        "integrity": pareto.get("baseline_integrity", "---"),
        "feasible": pareto.get("feasible", True),
        "fallback": pareto.get("fallback"),
        "recommended_action": data.get("recommended_action", "None"),
        "fixes": fixes,
        "open_questions": data.get("open_questions", []),
    }


def _fmt(val: float | None, width: int = 3) -> str:
    """Format a float or return '---'."""
    if val is None:
        return "---"
    if width == 3:
        return f"{val:.3f}"
    return f"{val:.6f}"


def _short_action(rec: dict) -> str:
    """Return a short action label for the health table."""
    if rec["fixes"]:
        return rec["fixes"][0].get("action_type", "Adjust verifier")
    ra = rec["recommended_action"]
    if not ra or ra.lower().startswith("none"):
        return "None"
    # Extract the verb phrase before the first colon or period
    for sep in (":", "."):
        if sep in ra:
            prefix = ra.split(sep)[0].strip()
            if len(prefix) <= 30:
                return prefix
    return ra[:30]


def load_all_audits(model_label: str) -> list[dict]:
    """Load and parse all audit JSONs for a model.

    Returns a list of parsed conflict dicts (see _parse_conflict),
    sorted by conflict id.
    """
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
        results.append(_parse_conflict(data))
    return results


def build_audit_summary(
    model_label: str,
    timestamp: str,
    *,
    human_timestamp: str | None = None,
) -> Path:
    """Build the audit summary markdown from per-conflict JSON files.

    Args:
        model_label: e.g. "meta-llama_Llama-3.3-70B-Instruct"
        timestamp: MMDD_HHMM format for the output filename.
        human_timestamp: "YYYY-MM-DD HH:MM" for the report header.
            Derived from timestamp if not provided.

    Returns:
        Path to the written summary file.
    """
    if human_timestamp is None:
        # Convert MMDD_HHMM → YYYY-MM-DD HH:MM (assume current year)
        from datetime import datetime

        now = datetime.now()
        human_timestamp = (
            f"{now.year}-{timestamp[:2]}-{timestamp[2:4]} "
            f"{timestamp[5:7]}:{timestamp[7:9]}"
        )

    results = load_all_audits(model_label)
    if not results:
        raise ValueError(f"No audit JSONs found for {model_label}")

    green = [r for r in results if r["severity"] == "GREEN"]
    yellow = [r for r in results if r["severity"] == "YELLOW"]
    amber = [r for r in results if r["severity"] == "AMBER"]
    red = [r for r in results if r["severity"] == "RED"]

    lines: list[str] = []

    # --- Header ---
    lines.append("# Condition C Verifier Audit Summary")
    lines.append("")
    lines.append(f"**Date:** {human_timestamp}")
    lines.append(f"**Model audited:** {model_label}")
    lines.append(f"**Conflicts audited:** {len(results)}")
    lines.append("**Accuracy target:** 98%")
    lines.append("")

    # --- Overview ---
    lines.append("## Overview")
    lines.append("")
    lines.append("| Rating | Count | Conflicts |")
    lines.append("|--------|-------|-----------|")
    for label, group in [
        ("GREEN (0% error)", green),
        ("YELLOW (>0% and <3%)", yellow),
        ("AMBER (>=3% and <10%)", amber),
        ("RED (>=10%)", red),
    ]:
        ids = ", ".join(r["id"] for r in group)
        lines.append(f"| {label} | {len(group)} | {ids} |")
    lines.append("")

    # --- Infeasible thresholds ---
    infeasible = [r for r in results if not r.get("feasible", True)]
    if infeasible:
        lines.append("## Infeasible Thresholds")
        lines.append("")
        lines.append(
            "These float conflicts have no Pareto-optimal threshold "
            "meeting quality caps. The threshold is a fallback — "
            "classification accuracy may be imperfect. Root cause could be "
            "overlapping model distributions, a flawed scorer, or both."
        )
        lines.append("")
        lines.append("| Conflict | Fallback | BA | Integrity |")
        lines.append("|----------|----------|----|-----------|")
        for r in infeasible:
            fb = r.get("fallback") or "---"
            lines.append(
                f"| {r['id']} | {fb} | {_fmt(r['ba'])} "
                f"| {r['integrity']} |"
            )
        lines.append("")

    # --- Conflict Health ---
    lines.append("## Conflict Health")
    lines.append("")
    lines.append(
        "| Conflict | Type | Feas | BA | d_norm | c_norm | Dist "
        "| Integrity | Rating | Error% | Action |"
    )
    lines.append(
        "|----------|------|------|----|--------|--------|------"
        "|-----------|--------|--------|--------|"
    )
    for r in sorted(results, key=lambda x: -x["error_pct"]):
        feas = "Y" if r.get("feasible", True) else "N"
        lines.append(
            f"| {r['id']} | {r['type']} | {feas} | {_fmt(r['ba'])} "
            f"| {_fmt(r['d_norm'], 6)} | {_fmt(r['c_norm'], 6)} "
            f"| {r['dist']} | {r['integrity']} | {r['severity']} "
            f"| {r['error_pct']}% | {_short_action(r)} |"
        )
    lines.append("")

    # --- Suggested Fixes ---
    fix_rows = []
    for r in results:
        for fix in r["fixes"]:
            desc = fix.get("description", "---")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            fix_rows.append(
                {
                    "id": r["id"],
                    "desc": desc,
                    "complexity": fix.get("complexity", "---"),
                    "risk": fix.get("risk_to_other_models", fix.get("risk", "---")),
                    "cur_err": r["error_pct"],
                    "est_err": fix.get("estimated_error_pct", "---"),
                    "confidence": fix.get("confidence", "---"),
                }
            )

    if fix_rows:
        fix_rows.sort(key=lambda x: -x["cur_err"])
        lines.append("## Suggested Fixes Prioritization")
        lines.append("")
        lines.append(
            "| Conflict | Fix | Complexity | Risk "
            "| Cur Error% | Est Error% | Confidence |"
        )
        lines.append(
            "|----------|-----|------------|------"
            "|------------|------------|------------|"
        )
        for f in fix_rows:
            est = f["est_err"]
            est_s = f"{est}%" if isinstance(est, (int, float)) else f"{est}"
            lines.append(
                f"| {f['id']} | {f['desc']} | {f['complexity']} "
                f"| {f['risk']} | {f['cur_err']}% | {est_s} "
                f"| {f['confidence']} |"
            )
        lines.append("")

    # --- Placeholders for human-written sections ---
    lines.append("## Cross-cutting findings")
    lines.append("")
    lines.append("<!-- Synthesize from notes across all conflict JSONs. -->")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("<!-- Prioritized actions grouped by shared root cause. -->")
    lines.append("")

    # --- Write ---
    out_path = _AUDIT_BASE / model_label / f"summary_{timestamp}.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
