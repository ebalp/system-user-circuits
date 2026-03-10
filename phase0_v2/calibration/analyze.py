"""Calibration analysis: extract calibration insights from experiment results.

Analyzes baseline conditions (A & B) to compute SBR/UCR rates and
float-score trying/ignoring distributions. Identifies anomalous records.

Usage:
    uv run python -m phase0_v2.calibration.analyze <results_file> [options]

Examples:
    uv run python -m phase0_v2.calibration.analyze phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl
    uv run python -m phase0_v2.calibration.analyze results.jsonl --output-dir cal_out
    uv run python -m phase0_v2.calibration.analyze results.jsonl --conflict sentence_chaining
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from ._shared import (
    load_records,
    build_conflict_threshold_map,
    direction_to_verify_code,
    apply_threshold,
    ConflictThresholdInfo,
)
from ..conflicts.registry import get_all_conflicts


def _build_constraint_labels() -> dict[str, tuple[str, str]]:
    """Build a map of conflict_id -> (label_a, label_b) from the registry."""
    labels = {}
    for conflict in get_all_conflicts():
        labels[conflict.conflict_id] = conflict.get_constraint_labels()
    return labels


def _check_completeness(
    records: list[dict],
    config_path: str | None,
    conflict_filter: str | None = None,
) -> None:
    """Check dataset completeness and print warnings for gaps.

    Computes expected record counts per (conflict, condition) from the config
    and conflict registry, then compares against actual counts in the data.
    """
    # Load config for style counts and task counts
    n_sys_styles = 5  # defaults
    n_usr_styles = 5
    n_tasks = 50
    if config_path and Path(config_path).exists():
        try:
            from ..src.config import load_config
            config = load_config(config_path)
            n_sys_styles = len(config.condition_c_system_styles)
            n_usr_styles = len(config.user_styles_to_test)
            n_tasks = len(config.tasks)
        except Exception:
            pass  # fall back to defaults

    # Build registry info: which conflicts are invertible
    registry_conflicts: dict[str, bool] = {}  # conflict_id -> is_invertible
    for conflict in get_all_conflicts():
        registry_conflicts[conflict.conflict_id] = conflict.supports_counterbalancing()

    # Count actual records per (conflict_id, condition)
    actual: dict[tuple[str, str], int] = defaultdict(int)
    error_count = 0
    for rec in records:
        if rec.get("error") is not None:
            error_count += 1
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue
        actual[(cid, rec["condition"])] += 1

    # Compute expected counts per (conflict_id, condition)
    data_conflict_ids = {cid for (cid, _) in actual}
    gaps = []

    for cid in sorted(data_conflict_ids):
        invertible = registry_conflicts.get(cid, True)
        n_dir = 2 if invertible else 1
        expected = {
            "A": n_dir * n_tasks,
            "B": n_dir * n_tasks,
            "C": n_dir * n_sys_styles * n_usr_styles * n_tasks,
            "D": (n_dir * n_tasks) if invertible else 0,
        }
        for cond in ("A", "B", "C", "D"):
            exp = expected[cond]
            act = actual.get((cid, cond), 0)
            if act != exp:
                gaps.append((cid, cond, exp, act))

    # Print results
    print(f"\n{'=' * 70}")
    print("DATASET COMPLETENESS")
    print(f"{'=' * 70}")
    print(f"  Records: {len(records)} total, {error_count} errors")
    print(f"  Conflicts in data: {len(data_conflict_ids)}")
    print(f"  Expected per conflict: {n_tasks} tasks, {n_sys_styles}x{n_usr_styles} styles (Cond C)")

    if not gaps:
        print("  Status: COMPLETE — all conflicts have expected record counts")
    else:
        print(f"  Status: INCOMPLETE — {len(gaps)} gaps found:")
        print(f"  {'conflict_id':<35} {'cond':>4} {'expected':>8} {'actual':>8} {'diff':>8}")
        print(f"  {'-' * 67}")
        for cid, cond, exp, act in gaps:
            diff = act - exp
            sign = "+" if diff > 0 else ""
            print(f"  {cid:<35} {cond:>4} {exp:>8} {act:>8} {sign}{diff:>7}")


def _check_threshold_consistency(
    records: list[dict],
    threshold_map: dict[str, ConflictThresholdInfo],
    conflict_filter: str | None = None,
) -> None:
    """Check if stored verify results are consistent with current thresholds.

    For float-scored records, re-applies the current threshold to stored scores
    and checks if the result matches the stored verify_system_result /
    verify_user_result. Mismatches mean the results file was scored with a
    different threshold than what's currently configured.
    """
    mismatches: dict[str, int] = defaultdict(int)
    checked: dict[str, int] = defaultdict(int)

    for rec in records:
        if rec.get("error") is not None:
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue

        info = threshold_map.get(cid)
        if info is None:
            continue

        sys_score = rec.get("verify_system_score")
        usr_score = rec.get("verify_user_score")
        if sys_score is None or usr_score is None:
            continue

        direction = rec["direction"]
        verify_code = direction_to_verify_code(direction)
        if verify_code not in info.sides:
            continue

        # Check if scores are genuinely float (not just 0.0/1.0 from bool)
        if {sys_score, usr_score} <= {0.0, 1.0}:
            continue

        checked[cid] += 1
        sys_info = info.sides[verify_code]["system"]
        usr_info = info.sides[verify_code]["user"]

        expected_sys = apply_threshold(sys_score, info.threshold, sys_info.is_inverted)
        expected_usr = apply_threshold(usr_score, info.threshold, usr_info.is_inverted)

        stored_sys = rec.get("verify_system_result")
        stored_usr = rec.get("verify_user_result")

        if expected_sys != stored_sys or expected_usr != stored_usr:
            mismatches[cid] += 1

    if not checked:
        return

    print(f"\n{'=' * 70}")
    print("THRESHOLD CONSISTENCY")
    print(f"{'=' * 70}")
    if not mismatches:
        print(f"  OK — all {sum(checked.values())} float-scored records match current thresholds")
    else:
        print(f"  WARNING — stored results were scored with different thresholds:")
        print(f"  {'conflict_id':<35} {'mismatched':>10} {'checked':>10} {'%':>7}")
        print(f"  {'-' * 65}")
        for cid in sorted(mismatches):
            n_mis = mismatches[cid]
            n_chk = checked[cid]
            pct = 100 * n_mis / n_chk
            print(f"  {cid:<35} {n_mis:>10} {n_chk:>10} {pct:>6.1f}%")
        print("  Run rescore to re-apply current thresholds to stored scores.")


def _print_constraint_legend(
    labels: dict[str, tuple[str, str]],
    conflict_ids: set[str],
) -> None:
    """Print a legend showing what constraint a and b mean for each conflict."""
    print("\n" + "=" * 90)
    print("CONSTRAINT LEGEND — what (a) and (b) mean for each conflict")
    print("=" * 90)
    for cid in sorted(conflict_ids):
        if cid not in labels:
            continue
        label_a, label_b = labels[cid]
        print(f"  {cid:<35} a = {label_a}")
        print(f"  {'':<35} b = {label_b}")


def _is_float_scored(record: dict) -> bool:
    """Check if a record has genuine float scores (not just 0.0/1.0 from bool)."""
    sys_score = record.get("verify_system_score")
    usr_score = record.get("verify_user_score")
    if sys_score is None or usr_score is None:
        return False
    return True


def _group_records(
    records: list[dict], conflict_filter: str | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Group records by (conflict_id, direction) with optional conflict filter."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("error") is not None:
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue
        direction = rec["direction"]
        groups[(cid, direction)].append(rec)
    return groups


def _constraint_from_direction(direction: str, condition: str) -> str:
    """Which constraint (a/b) is being tested, given the condition.

    In Condition A (system only), the system side of a_to_b is constraint a.
    In Condition B (user only), the user side of a_to_b is constraint b.
    """
    verify_code = direction_to_verify_code(direction)
    if condition == "B":
        return "b" if verify_code == "a" else "a"  # user side = opposite
    return verify_code  # system side for A, both for C/D


def _compute_baseline_rates(
    records: list[dict], conflict_filter: str | None = None
) -> list[dict]:
    """Compute SBR and UCR per (conflict_id), grouped by constraint.

    SBR(x) = fraction of condition A where constraint x is tested and label == 'followed_system'
    UCR(x) = fraction of condition B where constraint x is tested and label == 'followed_user'

    Derivation:
      SBR(a) = Cond A + direction a_to_b  (system tests constraint a)
      UCR(a) = Cond B + direction b_to_a  (user tests constraint a)
      SBR(b) = Cond A + direction b_to_a  (system tests constraint b)
      UCR(b) = Cond B + direction a_to_b  (user tests constraint b)
    """
    groups = _group_records(records, conflict_filter)

    # Group by (conflict_id, constraint, metric_type)
    # metric_type is "sbr" (from cond A) or "ucr" (from cond B)
    by_conflict: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (cid, direction), recs in groups.items():
        for rec in recs:
            cond = rec["condition"]
            if cond not in ("A", "B"):
                continue
            constraint = _constraint_from_direction(direction, cond)
            key = f"{'sbr' if cond == 'A' else 'ucr'}_{constraint}"
            by_conflict[cid][key].append(rec)

    results = []
    for cid in sorted(by_conflict.keys()):
        buckets = by_conflict[cid]

        def _rate(bucket_key: str, success_label: str) -> tuple[float | None, int]:
            recs = buckets.get(bucket_key, [])
            n = len(recs)
            if n == 0:
                return None, 0
            return sum(1 for r in recs if r["label"] == success_label) / n, n

        sbr_a, n_sbr_a = _rate("sbr_a", "followed_system")
        ucr_a, n_ucr_a = _rate("ucr_a", "followed_user")
        sbr_b, n_sbr_b = _rate("sbr_b", "followed_system")
        ucr_b, n_ucr_b = _rate("ucr_b", "followed_user")

        results.append(
            {
                "conflict_id": cid,
                "sbr_a": sbr_a,
                "ucr_a": ucr_a,
                "sbr_b": sbr_b,
                "ucr_b": ucr_b,
                "n_sbr_a": n_sbr_a,
                "n_ucr_a": n_ucr_a,
                "n_sbr_b": n_sbr_b,
                "n_ucr_b": n_ucr_b,
            }
        )
    return results


def _compute_all_balanced_accuracy(
    records: list[dict], conflict_filter: str | None = None
) -> dict[str, float]:
    """Compute per-conflict balanced accuracy from baseline verify results.

    Uses the final verify_system_result / verify_user_result booleans.
    For boolean conflicts these are direct True/False from the verifier;
    for float conflicts they incorporate the verify_threshold.

    For each conflict, averages balanced accuracy across all 4
    (constraint, role) combinations:
      TPR = P(verify=True | trying), TNR = P(verify=False | ignoring)
      BA_row = (TPR + TNR) / 2
      BA_conflict = mean(BA_row for all 4 rows)
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("error") is not None:
            continue
        cond = rec["condition"]
        if cond not in ("A", "B"):
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue
        direction = rec["direction"]
        verify_code = direction_to_verify_code(direction)
        groups[(cid, verify_code, cond)].append(rec)

    conflict_bas: dict[str, list[float]] = defaultdict(list)

    seen: set[tuple[str, str]] = set()
    for (cid, verify_code, _) in sorted(groups.keys()):
        if (cid, verify_code) in seen:
            continue
        seen.add((cid, verify_code))

        cond_a = groups.get((cid, verify_code, "A"), [])
        cond_b = groups.get((cid, verify_code, "B"), [])

        if not cond_a or not cond_b:
            continue

        # System side: trying=CondA, ignoring=CondB
        sys_try = [r.get("verify_system_result", False) for r in cond_a]
        sys_ign = [r.get("verify_system_result", False) for r in cond_b]
        if sys_try and sys_ign:
            tpr = sum(1 for v in sys_try if v) / len(sys_try)
            tnr = sum(1 for v in sys_ign if not v) / len(sys_ign)
            conflict_bas[cid].append((tpr + tnr) / 2)

        # User side: trying=CondB, ignoring=CondA
        usr_try = [r.get("verify_user_result", False) for r in cond_b]
        usr_ign = [r.get("verify_user_result", False) for r in cond_a]
        if usr_try and usr_ign:
            tpr = sum(1 for v in usr_try if v) / len(usr_try)
            tnr = sum(1 for v in usr_ign if not v) / len(usr_ign)
            conflict_bas[cid].append((tpr + tnr) / 2)

    return {cid: statistics.mean(bas) for cid, bas in conflict_bas.items() if bas}


def _find_conflict_optimal_threshold(
    rows: list[dict],
) -> tuple[float, float, float]:
    """Find the threshold range [T_low, T_high] that maximizes mean balanced
    accuracy across all (constraint, role) rows of a conflict.

    Each conflict has one verify_threshold applied to all 4 rows, but the
    comparison depends on is_inverted:
      - Direct (not inverted): trying should pass score >= T, ignoring should fail score < T
      - Inverted (1-score):    trying should pass score > (1-T), ignoring should fail score <= (1-T)

    Strategy: collect all unique scores as candidates, sweep T, compute
    balanced accuracy per row with the correct operator, average across rows.
    Then collect all T values achieving the same max BA.

    Returns (t_low, t_high, best_mean_balanced_accuracy).
    """
    if not rows:
        return 0.0, 0.0, 0.0

    # Collect all unique scores as candidate thresholds.
    # Include 1-s for each score s: inverted rows use boundary 1-T,
    # so T=1-s tests a boundary exactly at score s for inverted rows.
    # Also include midpoints between consecutive values to cover gaps
    # where the optimal threshold lies between two data points.
    raw_scores: set[float] = set()
    for row in rows:
        raw_scores.update(row["_trying"])
        raw_scores.update(row["_ignoring"])
    all_candidates: set[float] = set(raw_scores)
    for s in raw_scores:
        complement = 1.0 - s
        if 0.0 <= complement <= 1.0:
            all_candidates.add(complement)
    sorted_vals = sorted(all_candidates)
    # Add midpoints between consecutive candidates
    midpoints: list[float] = []
    for i in range(len(sorted_vals) - 1):
        midpoints.append((sorted_vals[i] + sorted_vals[i + 1]) / 2)
    all_candidates.update(midpoints)
    candidates = sorted(all_candidates)
    if not candidates:
        return 0.0, 0.0, 0.0

    best_ba = 0.0
    threshold_bas: list[tuple[float, float]] = []

    for t in candidates:
        ba_sum = 0.0
        n_rows = 0
        for row in rows:
            trying = row["_trying"]
            ignoring = row["_ignoring"]
            if not trying or not ignoring:
                continue
            n_try = len(trying)
            n_ign = len(ignoring)

            if row["is_inverted"]:
                # Inverted: pass when score > (1-T)
                boundary = 1.0 - t
                tpr = sum(1 for s in trying if s > boundary) / n_try
                tnr = sum(1 for s in ignoring if s <= boundary) / n_ign
            else:
                # Direct: pass when score >= T
                tpr = sum(1 for s in trying if s >= t) / n_try
                tnr = sum(1 for s in ignoring if s < t) / n_ign

            ba_sum += (tpr + tnr) / 2
            n_rows += 1

        if n_rows > 0:
            mean_ba = ba_sum / n_rows
            threshold_bas.append((t, mean_ba))
            if mean_ba > best_ba:
                best_ba = mean_ba

    # Collect all thresholds achieving max BA (within floating point tolerance)
    matching = [t for t, ba in threshold_bas if abs(ba - best_ba) < 1e-9]
    if not matching:
        return candidates[0], candidates[0], best_ba

    return min(matching), max(matching), best_ba


def _compute_float_calibration(
    records: list[dict],
    threshold_map: dict[str, ConflictThresholdInfo],
    conflict_filter: str | None = None,
) -> list[dict]:
    """Compute trying/ignoring score distributions for float-scored conflicts.

    Returns rows keyed by (conflict_id, constraint, role) where:
    - constraint = "a" or "b" (which constraint is being measured)
    - role = "system" or "user" (which prompt slot carried the constraint)

    Mapping from (verify_code, score_field) to (constraint, role):
      verify_code "a" (direction a_to_b): system=constraint_a, user=constraint_b
      verify_code "b" (direction b_to_a): system=constraint_b, user=constraint_a

    Score sources:
      verify_system_score: trying from Cond A (system active), ignoring from Cond B (no system)
      verify_user_score:   trying from Cond B (user active),   ignoring from Cond A (no user)
    """
    # Group baseline records by (conflict_id, verify_code, condition)
    baseline_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("error") is not None:
            continue
        cond = rec["condition"]
        if cond not in ("A", "B"):
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue
        direction = rec["direction"]
        verify_code = direction_to_verify_code(direction)
        baseline_groups[(cid, verify_code, cond)].append(rec)

    results = []
    seen_keys = set()

    for (cid, verify_code, _), _ in sorted(baseline_groups.items()):
        if (cid, verify_code) in seen_keys:
            continue
        seen_keys.add((cid, verify_code))

        info = threshold_map.get(cid)
        if info is None:
            continue
        if verify_code not in info.sides:
            continue

        cond_a_recs = baseline_groups.get((cid, verify_code, "A"), [])
        cond_b_recs = baseline_groups.get((cid, verify_code, "B"), [])

        if not cond_a_recs and not cond_b_recs:
            continue

        # System side: trying from Cond A, ignoring from Cond B
        sys_trying = [r["verify_system_score"] for r in cond_a_recs if r.get("verify_system_score") is not None]
        sys_ignoring = [r["verify_system_score"] for r in cond_b_recs if r.get("verify_system_score") is not None]

        # User side: trying from Cond B, ignoring from Cond A
        usr_trying = [r["verify_user_score"] for r in cond_b_recs if r.get("verify_user_score") is not None]
        usr_ignoring = [r["verify_user_score"] for r in cond_a_recs if r.get("verify_user_score") is not None]

        sys_info = info.sides[verify_code]["system"]
        usr_info = info.sides[verify_code]["user"]

        # Map (verify_code, side) -> (constraint, role)
        # verify_code "a": system tests constraint a, user tests constraint b
        # verify_code "b": system tests constraint b, user tests constraint a
        side_mappings = [
            ("system", sys_trying, sys_ignoring, sys_info,
             verify_code, "system"),  # system side always matches verify_code
            ("user", usr_trying, usr_ignoring, usr_info,
             "b" if verify_code == "a" else "a", "user"),  # user side is opposite
        ]

        for _side, trying, ignoring, side_info, constraint, role in side_mappings:
            # Skip sides where all scores are 0.0 or 1.0 (bool-like)
            all_scores = trying + ignoring
            if not all_scores:
                continue
            unique_vals = set(all_scores)
            if unique_vals <= {0.0, 1.0}:
                continue

            trying_mean = statistics.mean(trying) if trying else None
            trying_min = min(trying) if trying else None
            trying_max = max(trying) if trying else None
            trying_std = statistics.stdev(trying) if len(trying) > 1 else 0.0
            ignoring_mean = statistics.mean(ignoring) if ignoring else None
            ignoring_min = min(ignoring) if ignoring else None
            ignoring_max = max(ignoring) if ignoring else None
            ignoring_std = statistics.stdev(ignoring) if len(ignoring) > 1 else 0.0

            gap = (trying_mean - ignoring_mean) if trying_mean is not None and ignoring_mean is not None else None

            results.append(
                {
                    "conflict_id": cid,
                    "constraint": constraint,
                    "role": role,
                    "trying_mean": trying_mean,
                    "trying_min": trying_min,
                    "trying_max": trying_max,
                    "trying_std": trying_std,
                    "ignoring_mean": ignoring_mean,
                    "ignoring_min": ignoring_min,
                    "ignoring_max": ignoring_max,
                    "ignoring_std": ignoring_std,
                    "gap": gap,
                    "current_threshold": info.threshold,
                    "is_inverted": side_info.is_inverted,
                    # _trying/_ignoring kept for optimal threshold computation
                    "_trying": trying,
                    "_ignoring": ignoring,
                    "n_trying": len(trying),
                    "n_ignoring": len(ignoring),
                }
            )

    # Phase 2: compute per-conflict optimal threshold across all rows
    by_conflict: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_conflict[row["conflict_id"]].append(row)

    for cid, conflict_rows in by_conflict.items():
        t_low, t_high, best_ba = _find_conflict_optimal_threshold(conflict_rows)
        for row in conflict_rows:
            row["optimal_threshold_low"] = t_low
            row["optimal_threshold_high"] = t_high
            row["optimal_threshold"] = (t_low + t_high) / 2
            row["balanced_accuracy"] = best_ba

    # Remove internal fields
    for row in results:
        del row["_trying"]
        del row["_ignoring"]

    return results


def _find_anomalies(
    records: list[dict], conflict_filter: str | None = None
) -> list[dict]:
    """Find anomalous records across all conditions."""
    anomalies = []
    for rec in records:
        if rec.get("error") is not None:
            continue
        cid = rec["conflict_id"]
        if conflict_filter and cid != conflict_filter:
            continue

        cond = rec["condition"]
        label = rec["label"]
        reasons = []

        if label == "followed_both":
            reasons.append("followed_both")
        if cond == "A" and label == "followed_user":
            reasons.append("cond_A_followed_user")
        if cond == "B" and label == "followed_system":
            reasons.append("cond_B_followed_system")

        for reason in reasons:
            anomalies.append(
                {
                    "conflict_id": cid,
                    "condition": cond,
                    "direction": rec["direction"],
                    "label": label,
                    "expected_label": rec.get("expected_label"),
                    "anomaly_reason": reason,
                    "verify_system_score": rec.get("verify_system_score"),
                    "verify_user_score": rec.get("verify_user_score"),
                    "verify_system_result": rec.get("verify_system_result"),
                    "verify_user_result": rec.get("verify_user_result"),
                    "response": rec.get("response", "")[:500],
                    "prompt_id": rec.get("prompt_id"),
                }
            )
    return anomalies


def _print_baseline_table(baseline_rates: list[dict]) -> None:
    """Print SBR/UCR table grouped by constraint."""
    print("\n" + "=" * 90)
    print("BASELINE RATES — grouped by constraint (a/b)")
    print("  SBR(x) = P(followed_system | Cond A, constraint x)")
    print("  UCR(x) = P(followed_user   | Cond B, constraint x)")
    print("=" * 90)
    header = f"{'conflict_id':<35} {'SBR(a)':>7} {'UCR(a)':>7} {'SBR(b)':>7} {'UCR(b)':>7} {'BA':>6} {'n':>5}"
    print(header)
    print("-" * 95)
    for row in baseline_rates:
        def _fmt(val):
            return f"{val:.3f}" if val is not None else "  N/A"

        n_total = row["n_sbr_a"] + row["n_ucr_a"] + row["n_sbr_b"] + row["n_ucr_b"]
        ba = row.get("balanced_accuracy")
        ba_str = f"{ba:.3f}" if ba is not None else " N/A"
        flags = []
        for key in ("sbr_a", "ucr_a", "sbr_b", "ucr_b"):
            if row[key] is not None and row[key] < 0.5:
                flags.append(f"*{key}")

        flag_str = f"  {' '.join(flags)}" if flags else ""
        print(
            f"{row['conflict_id']:<35} {_fmt(row['sbr_a']):>7} {_fmt(row['ucr_a']):>7} "
            f"{_fmt(row['sbr_b']):>7} {_fmt(row['ucr_b']):>7} {ba_str:>6} {n_total:>5}{flag_str}"
        )
    print("(* = rate below 0.5, flag for discard consideration)")


def _print_calibration_table(calibration: list[dict]) -> None:
    """Print float score calibration table."""
    if not calibration:
        print("\nNo float-scored conflicts with non-trivial distributions found.")
        return
    print("\n" + "=" * 140)
    print("FLOAT SCORE CALIBRATION")
    print("  constraint = which constraint (a/b) the score measures")
    print("  role = which prompt slot carried the constraint (system=Cond A, user=Cond B)")
    print("=" * 140)
    header = (
        f"{'conflict_id':<35} {'con':>3} {'role':>6} "
        f"{'try_mean':>8} {'ign_mean':>8} "
        f"{'gap':>7} {'thresh':>6} {'inv':>3} {'opt_range':>16} {'opt_mid':>7} {'bal_acc':>7}"
    )
    print(header)
    print("-" * 130)
    for row in calibration:
        try_mean = f"{row['trying_mean']:.3f}" if row["trying_mean"] is not None else "   N/A"
        ign_mean = f"{row['ignoring_mean']:.3f}" if row["ignoring_mean"] is not None else "   N/A"
        gap_str = f"{row['gap']:+.3f}" if row["gap"] is not None else "  N/A"
        inv_str = "Y" if row["is_inverted"] else "N"
        t_low = row["optimal_threshold_low"]
        t_high = row["optimal_threshold_high"]
        if abs(t_low - t_high) < 1e-9:
            opt_str = f"{t_low:.3f}"
        else:
            opt_str = f"[{t_low:.3f}, {t_high:.3f}]"
        opt_mid = (t_low + t_high) / 2
        opt_mid_str = f"{opt_mid:.3f}"
        ba_str = f"{row['balanced_accuracy']:.3f}"
        print(
            f"{row['conflict_id']:<35} {row['constraint']:>3} {row['role']:>6} "
            f"{try_mean:>8} {ign_mean:>8} "
            f"{gap_str:>7} {row['current_threshold']:>6.2f} {inv_str:>3} {opt_str:>16} {opt_mid_str:>7} {ba_str:>7}"
        )


def _print_anomaly_summary(anomalies: list[dict]) -> None:
    """Print anomaly summary."""
    if not anomalies:
        print("\nNo anomalies found.")
        return
    print("\n" + "=" * 60)
    print("ANOMALIES")
    print("=" * 60)
    by_reason: dict[str, int] = defaultdict(int)
    by_conflict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in anomalies:
        by_reason[a["anomaly_reason"]] += 1
        by_conflict[a["conflict_id"]][a["anomaly_reason"]] += 1

    print("By reason:")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason:<30} {count:>5}")

    print("\nBy conflict:")
    for cid, reasons in sorted(by_conflict.items()):
        parts = ", ".join(f"{r}={c}" for r, c in sorted(reasons.items()))
        print(f"  {cid:<40} {parts}")


def _write_combined_csv(
    baseline_rates: list[dict],
    calibration: list[dict],
    constraint_labels: dict[str, tuple[str, str]],
    output_dir: Path,
) -> None:
    """Write calibration_report.csv — 4 rows per conflict (system_a, user_a, system_b, user_b).

    Each row combines baseline rate + float calibration data for one
    (conflict, constraint, role) combination.
    """
    # Index calibration data by (conflict_id, constraint, role)
    cal_index: dict[tuple[str, str, str], dict] = {}
    for row in calibration:
        key = (row["conflict_id"], row["constraint"], row["role"])
        cal_index[key] = row

    path = output_dir / "calibration_report.csv"
    fieldnames = [
        "conflict_id",
        "constraint", "constraint_description",
        "role", "baseline_rate", "n_baseline",
        "balanced_accuracy",
        # float calibration (empty for bool-only conflicts)
        "trying_mean", "trying_min", "trying_max", "trying_std",
        "ignoring_mean", "ignoring_min", "ignoring_max", "ignoring_std",
        "gap", "threshold", "is_inverted",
        "optimal_threshold", "optimal_threshold_low", "optimal_threshold_high",
        "n_trying", "n_ignoring",
    ]

    rows = []
    for br in baseline_rates:
        cid = br["conflict_id"]
        label_a, label_b = constraint_labels.get(cid, ("?", "?"))

        # 4 rows: (constraint, role, rate_key, n_key)
        for constraint, role, rate_key, n_key, label in [
            ("a", "system", "sbr_a", "n_sbr_a", label_a),
            ("a", "user",   "ucr_a", "n_ucr_a", label_a),
            ("b", "system", "sbr_b", "n_sbr_b", label_b),
            ("b", "user",   "ucr_b", "n_ucr_b", label_b),
        ]:
            row: dict = {
                "conflict_id": cid,
                "constraint": constraint,
                "constraint_description": label,
                "role": role,
                "baseline_rate": br[rate_key],
                "n_baseline": br[n_key],
                "balanced_accuracy": br.get("balanced_accuracy", ""),
            }

            # Merge float calibration if available
            cal = cal_index.get((cid, constraint, role))
            if cal:
                for k in [
                    "trying_mean", "trying_min", "trying_max", "trying_std",
                    "ignoring_mean", "ignoring_min", "ignoring_max", "ignoring_std",
                    "gap", "is_inverted",
                    "optimal_threshold", "optimal_threshold_low", "optimal_threshold_high",
                    "n_trying", "n_ignoring",
                ]:
                    row[k] = cal[k]
                row["threshold"] = cal["current_threshold"]
            else:
                for k in [
                    "trying_mean", "trying_min", "trying_max", "trying_std",
                    "ignoring_mean", "ignoring_min", "ignoring_max", "ignoring_std",
                    "gap", "is_inverted",
                    "optimal_threshold", "optimal_threshold_low", "optimal_threshold_high",
                    "n_trying", "n_ignoring", "threshold",
                ]:
                    row[k] = ""

            rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {path} ({len(rows)} rows, {len(baseline_rates)} conflicts)")


def _write_anomalies_jsonl(anomalies: list[dict], output_dir: Path) -> None:
    """Write anomalies.jsonl."""
    if not anomalies:
        return
    path = output_dir / "anomalies.jsonl"
    with open(path, "w") as f:
        for a in anomalies:
            f.write(json.dumps(a) + "\n")
    print(f"Wrote {path} ({len(anomalies)} records)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibration analysis of experiment results."
    )
    parser.add_argument("results_file", help="Path to JSONL results file")
    parser.add_argument(
        "--output-dir", default="calibration_output", help="Output directory (default: calibration_output/)"
    )
    parser.add_argument(
        "--conflict", default=None, help="Filter to single conflict_id"
    )
    parser.add_argument(
        "--config", default="phase0_v2/config/experiment.yaml",
        help="Path to experiment config (for completeness checks)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test mode: skip completeness/consistency checks",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {args.results_file}...")
    records = load_records(args.results_file)
    print(f"Loaded {len(records)} records")

    if args.conflict:
        print(f"Filtering to conflict: {args.conflict}")

    threshold_map = build_conflict_threshold_map()
    constraint_labels = _build_constraint_labels()

    # Dataset completeness and threshold consistency checks
    if not args.smoke:
        _check_completeness(records, args.config, args.conflict)
        _check_threshold_consistency(records, threshold_map, args.conflict)

    # Constraint legend
    conflict_ids = {r["conflict_id"] for r in records if r.get("error") is None}
    if args.conflict:
        conflict_ids = {args.conflict} & conflict_ids
    _print_constraint_legend(constraint_labels, conflict_ids)

    # Phase 1: Baseline analysis
    print("\n--- Phase 1: Baseline Analysis ---")
    baseline_rates = _compute_baseline_rates(records, args.conflict)
    all_ba = _compute_all_balanced_accuracy(records, args.conflict)
    for row in baseline_rates:
        row["balanced_accuracy"] = all_ba.get(row["conflict_id"])
    _print_baseline_table(baseline_rates)

    calibration = _compute_float_calibration(records, threshold_map, args.conflict)
    _print_calibration_table(calibration)

    # Anomalies (all conditions)
    anomalies = _find_anomalies(records, args.conflict)
    _print_anomaly_summary(anomalies)

    # Write output files
    print("\n--- Writing Output Files ---")
    _write_combined_csv(baseline_rates, calibration, constraint_labels, output_dir)
    _write_anomalies_jsonl(anomalies, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
