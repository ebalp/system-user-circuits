"""Per-model threshold optimizer using Pareto frontier analysis.

Computes three metrics at each candidate threshold T (0.001 steps):
1. d_norm(T): KDE density at T, normalized so valley_min=0, lower_peak=1 (lower=better)
2. c_norm(T): Otsu within-class variance at T, normalized so best_cost=0, total_var=1 (lower=better)
3. ba(T): balanced accuracy at T from baseline conditions A/B (higher=better)

Finds the Pareto frontier (thresholds where no other T is better on all three),
then selects lexicographically: max BA first, then min(d_norm + c_norm).
"""

import argparse
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import median

import numpy as np
import yaml
from scipy.stats import gaussian_kde

from ._shared import (
    build_conflict_threshold_map,
    direction_to_verify_code,
    load_records,
)

# Default feasibility caps — filter out degenerate solutions before Pareto computation.
# Tightened from 0.05/0.05/0.90 to surface borderline picks as infeasible so the audit's
# semantic-threshold step (Phase 4.5) can interrogate them.
DEFAULT_MAX_D_NORM = 0.02
DEFAULT_MAX_C_NORM = 0.02
DEFAULT_MIN_BA = 0.95

# Ambiguity bar — stricter than feasibility caps. A pick can be feasible yet still
# ambiguous (audit-worthy). Triggers when any cost is non-trivial OR any BA is sacrificed.
AMBIGUITY_MAX_D_NORM = 0.01
AMBIGUITY_MAX_C_NORM = 0.01


def is_ambiguous(result: dict, max_ba: float | None) -> bool:
    """Return True if the per-conflict result deserves a semantic-threshold audit.

    Triggers when any of:
      - feasible is False (Pareto fell back to valley/baseline_midpoint), OR
      - d_norm > AMBIGUITY_MAX_D_NORM, OR
      - c_norm > AMBIGUITY_MAX_C_NORM, OR
      - selected BA is below the model's max achievable BA (any BA cost).

    **Lock for audit-set thresholds:** if `result["source"]` starts with
    `audit_`, the entry was derived by an LLM agent's semantic analysis and is
    treated as locked — `is_ambiguous` returns False regardless of the metrics.
    This prevents future audits from re-deriving an already-vetted T. Use
    `--force-overwrite-audit` on the optimizer (or `revert_audit_recommendation`)
    to unlock.
    """
    source = result.get("source")
    if isinstance(source, str) and source.startswith("audit_"):
        return False
    if not result.get("feasible", False):
        return True
    d = result.get("d_norm")
    if d is not None and d > AMBIGUITY_MAX_D_NORM:
        return True
    c = result.get("c_norm")
    if c is not None and c > AMBIGUITY_MAX_C_NORM:
        return True
    ba = result.get("ba")
    if ba is not None and max_ba is not None and ba < max_ba - 1e-9:
        return True
    return False


# ── Helpers ──────────────────────────────────────────────────────────────


def _normalize_score(score: float, direction: str) -> float:
    """Normalize verify_system_score to the constraint-a scale.

    For a_to_b: score already measures constraint a -> return as-is.
    For b_to_a: score measures constraint b (inverted) -> return 1 - score.
    """
    if direction == "b_to_a":
        return 1.0 - score
    return score


def intersect_ranges(r1, r2):
    """Intersect two ranges, return None if no overlap."""
    if r1 is None or r2 is None:
        return None
    low = max(r1[0], r2[0])
    high = min(r1[1], r2[1])
    if low <= high:
        return (low, high)
    return None


# ── Baseline BA computation ──────────────────────────────────────────────


def _mean_ba_at_threshold(rows: list[dict], t: float) -> float:
    """Compute mean balanced accuracy across all rows at threshold t.

    Each row must have '_trying' (list of float scores from condition A)
    and '_ignoring' (list of float scores from condition B), plus 'is_inverted'.
    """
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
            boundary = 1.0 - t
            tpr = sum(1 for s in trying if s > boundary) / n_try
            tnr = sum(1 for s in ignoring if s <= boundary) / n_ign
        else:
            tpr = sum(1 for s in trying if s >= t) / n_try
            tnr = sum(1 for s in ignoring if s < t) / n_ign

        ba_sum += (tpr + tnr) / 2
        n_rows += 1

    return ba_sum / n_rows if n_rows > 0 else 0.0


def _find_optimal_threshold_range(
    rows: list[dict],
) -> tuple[float, float, float]:
    """Find the baseline optimal threshold range via brute-force 0.001 sweep.

    Returns (opt_low, opt_high, best_ba) for the largest contiguous region.
    """
    candidates = [i / 1000 for i in range(1, 1001)]
    best_ba = 0.0
    threshold_bas: list[tuple[float, float]] = []

    for t in candidates:
        mean_ba = _mean_ba_at_threshold(rows, t)
        threshold_bas.append((t, mean_ba))
        if mean_ba > best_ba:
            best_ba = mean_ba

    # Find all contiguous regions achieving max BA
    regions: list[tuple[float, float]] = []
    region_start: float | None = None
    region_end: float = 0.0
    for t, ba in threshold_bas:
        if abs(ba - best_ba) < 1e-9:
            if region_start is None:
                region_start = t
            region_end = t
        else:
            if region_start is not None:
                regions.append((region_start, region_end))
                region_start = None
    if region_start is not None:
        regions.append((region_start, region_end))

    if not regions:
        return candidates[0], candidates[0], best_ba

    best_region = max(regions, key=lambda r: r[1] - r[0])
    return best_region[0], best_region[1], best_ba


def compute_baseline_ranges(
    records: list[dict],
) -> dict[str, dict]:
    """Compute per-conflict baseline optimal ranges directly from JSONL records.

    Returns dict mapping conflict_id -> {opt_low, opt_high, ba}.
    Only includes float-scored conflicts.
    """
    threshold_map = build_conflict_threshold_map()

    # Group baseline records by (conflict_id, verify_code, condition)
    baseline_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in records:
        if rec.get("error") is not None:
            continue
        cond = rec.get("condition")
        if cond not in ("A", "B"):
            continue
        cid = rec.get("conflict_id")
        if cid is None:
            continue
        direction = rec.get("direction", "a_to_b")
        verify_code = direction_to_verify_code(direction)
        baseline_groups[(cid, verify_code, cond)].append(rec)

    # Collect all rows per conflict across both verify_codes, then sweep once
    all_rows: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()

    for (cid, verify_code, _) in sorted(baseline_groups.keys()):
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
        if not cond_a_recs or not cond_b_recs:
            continue

        sys_trying = [r["verify_system_score"] for r in cond_a_recs if r.get("verify_system_score") is not None]
        sys_ignoring = [r["verify_system_score"] for r in cond_b_recs if r.get("verify_system_score") is not None]
        usr_trying = [r["verify_user_score"] for r in cond_b_recs if r.get("verify_user_score") is not None]
        usr_ignoring = [r["verify_user_score"] for r in cond_a_recs if r.get("verify_user_score") is not None]

        sys_info = info.sides[verify_code]["system"]
        usr_info = info.sides[verify_code]["user"]

        for trying, ignoring, side_info in [
            (sys_trying, sys_ignoring, sys_info),
            (usr_trying, usr_ignoring, usr_info),
        ]:
            all_scores = trying + ignoring
            if not all_scores:
                continue
            if set(all_scores) <= {0.0, 1.0}:
                continue
            all_rows[cid].append({
                "_trying": trying,
                "_ignoring": ignoring,
                "is_inverted": side_info.is_inverted,
            })

    # Compute optimal range per conflict across all rows
    ranges = {}
    for cid, rows in sorted(all_rows.items()):
        if not rows:
            continue
        opt_low, opt_high, ba = _find_optimal_threshold_range(rows)
        ranges[cid] = {
            "opt_low": opt_low,
            "opt_high": opt_high,
            "ba": ba,
        }

    return ranges


# ── KDE distribution analysis ───────────────────────────────────────────


def classify_distribution(scores: np.ndarray, min_sep_frac: float = 0.05):
    """Fit KDE to scores and classify as unimodal/bimodal.

    Returns (distribution_type, peaks, kde, x, density).

    Peak detection: find tallest peak, mask a window around it (±10% of score
    range), then find tallest remaining peak above 2% of max density. This
    ensures the two peaks represent distinct modes, not noise within one mode.
    """
    lo, hi = float(scores.min()), float(scores.max())
    score_range = hi - lo
    pad = max(0.02, score_range * 0.05)
    grid_lo = max(0.0, lo - pad)
    grid_hi = min(1.0, hi + pad)

    if np.std(scores) < 1e-9:
        # Zero-variance data: KDE cannot fit, return as unknown
        x = np.linspace(grid_lo, grid_hi, 1000)
        return "unknown", [], None, x, np.zeros_like(x)

    kde = gaussian_kde(scores)  # Scott's rule bandwidth
    x = np.linspace(grid_lo, grid_hi, 1000)
    density = kde(x)

    min_peak_height = 0.02 * density.max()
    min_separation = score_range * min_sep_frac

    # Find all local maxima above height threshold
    all_peaks = []

    if density[0] > density[1] and density[0] > min_peak_height:
        all_peaks.append((x[0], density[0]))

    for i in range(1, len(x) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            if density[i] > min_peak_height:
                all_peaks.append((x[i], density[i]))

    if density[-1] > density[-2] and density[-1] > min_peak_height:
        all_peaks.append((x[-1], density[-1]))

    # Find two tallest peaks that are sufficiently separated
    all_peaks.sort(key=lambda p: p[1], reverse=True)
    peaks = []
    if all_peaks:
        peaks.append(all_peaks[0])
        for p in all_peaks[1:]:
            if abs(p[0] - peaks[0][0]) >= min_separation:
                peaks.append(p)
                break
    # Sort by x position for valley computation
    peaks.sort(key=lambda p: p[0])

    if len(peaks) <= 1:
        distribution = "unimodal"
    else:
        distribution = "bimodal"

    return distribution, peaks, kde, x, density


def _find_valley_location(scores: np.ndarray) -> float | None:
    """Find KDE density minimum between the two tallest separated peaks.

    Returns the valley x-location, or None if the distribution is unimodal.
    """
    dist, peaks, kde, x, density = classify_distribution(scores)
    if len(peaks) < 2:
        return None
    x_lo, x_hi = x[0], x[-1]
    x_span = x_hi - x_lo
    pl_idx = int((peaks[0][0] - x_lo) / x_span * 999)
    pr_idx = int((peaks[1][0] - x_lo) / x_span * 999)
    seg = density[pl_idx : pr_idx + 1]
    valley_idx = pl_idx + int(np.argmin(seg))
    return round(float(x[valley_idx]), 3)


# ── Pareto frontier threshold selection ──────────────────────────────────


def compute_three_metrics(
    cond_c_scores: np.ndarray,
    baseline_rows: list[dict],
    candidates: list[float],
) -> dict | None:
    """Compute d_norm, c_norm, and BA at each candidate threshold.

    d_norm is min(valley, mass_frac) — no classification needed:
    - valley: KDE density normalized between valley floor and lower peak (low at valleys)
    - mass_frac: 2*min(P(X<T), P(X>=T)) (low when mass is on one side)
    For bimodal: valley drives the min at the valley. For decisive unimodal: mass_frac
    drives it at tails. For messy unimodal: both are high, c_norm catches the rest.

    Returns dict with arrays and metadata, or None if zero-variance data.
    """
    if np.var(cond_c_scores) < 1e-10:
        return None

    dist, peaks, kde, x, density = classify_distribution(cond_c_scores)

    # Valley density normalization (between the two tallest peaks, if found)
    has_valley = len(peaks) >= 2
    if has_valley:
        x_lo, x_hi = x[0], x[-1]
        x_span = x_hi - x_lo
        pl_idx = int((peaks[0][0] - x_lo) / x_span * 999)
        pr_idx = int((peaks[1][0] - x_lo) / x_span * 999)
        seg = density[pl_idx : pr_idx + 1]
        valley_min = float(seg.min())
        lower_peak = float(min(peaks[0][1], peaks[1][1]))
        d_range = lower_peak - valley_min
    else:
        d_range = 0.0

    # Precompute sorted scores for fast ECDF lookup
    sorted_scores = np.sort(cond_c_scores)
    n_scores = len(sorted_scores)

    # Otsu costs
    total_var = float(np.var(cond_c_scores))
    best_cost = float("inf")
    for t in candidates:
        below = cond_c_scores[cond_c_scores < t]
        above = cond_c_scores[cond_c_scores >= t]
        if len(below) == 0 or len(above) == 0:
            continue
        cost = (len(below) / len(cond_c_scores)) * float(np.var(below)) + (
            len(above) / len(cond_c_scores)
        ) * float(np.var(above))
        if cost < best_cost:
            best_cost = cost
    c_range = total_var - best_cost

    d_norms = []
    c_norms = []
    bas = []

    for t in candidates:
        # Mass-fraction: low when most mass is on one side of T
        n_below = int(np.searchsorted(sorted_scores, t, side="left"))
        frac_below = n_below / n_scores
        d_mass = 2.0 * min(frac_below, 1.0 - frac_below)

        # Valley: low when T is at a density valley between peaks
        if has_valley and d_range > 0:
            d_val = float(kde(t)[0])
            d_valley = (d_val - valley_min) / d_range
            d_valley = max(0.0, d_valley)
        else:
            d_valley = 1.0  # No valley found — valley metric non-binding

        d_norms.append(min(d_valley, d_mass))

        # Otsu cost (lower = better, 0 = best separation)
        below = cond_c_scores[cond_c_scores < t]
        above = cond_c_scores[cond_c_scores >= t]
        if len(below) == 0 or len(above) == 0:
            c_norm = 1.0
        else:
            cost = (len(below) / len(cond_c_scores)) * float(np.var(below)) + (
                len(above) / len(cond_c_scores)
            ) * float(np.var(above))
            c_norm = (cost - best_cost) / c_range if c_range > 0 else 0.0
        c_norms.append(max(0.0, c_norm))

        # BA (higher = better)
        if baseline_rows:
            ba = _mean_ba_at_threshold(baseline_rows, t)
            bas.append(ba)
        else:
            bas.append(1.0)

    return {
        "candidates": candidates,
        "d_norm": np.array(d_norms),
        "c_norm": np.array(c_norms),
        "ba": np.array(bas),
        "best_cost": best_cost,
        "total_var": total_var,
        "distribution": dist,
    }


def find_pareto_frontier(
    metrics: dict,
    max_d: float = DEFAULT_MAX_D_NORM,
    max_c: float = DEFAULT_MAX_C_NORM,
    min_ba: float = DEFAULT_MIN_BA,
) -> list[int]:
    """Find Pareto-optimal indices within feasibility caps.

    A point i dominates point j if i is at least as good on all three metrics
    and strictly better on at least one (d_norm: lower=better, c_norm: lower=better,
    ba: higher=better).
    """
    d = metrics["d_norm"]
    c = metrics["c_norm"]
    ba = metrics["ba"]

    feasible = [
        i for i in range(len(d))
        if d[i] <= max_d and c[i] <= max_c and ba[i] >= min_ba
    ]

    pareto = []
    for i in feasible:
        dominated = False
        for j in feasible:
            if i == j:
                continue
            if d[j] <= d[i] and c[j] <= c[i] and ba[j] >= ba[i]:
                if d[j] < d[i] or c[j] < c[i] or ba[j] > ba[i]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)
    return pareto


def select_threshold(
    cond_c_scores: np.ndarray,
    baseline_rows: list[dict],
    *,
    max_d: float = DEFAULT_MAX_D_NORM,
    max_c: float = DEFAULT_MAX_C_NORM,
    min_ba: float = DEFAULT_MIN_BA,
) -> dict:
    """Select optimal threshold for a single conflict via Pareto frontier.

    Returns dict with threshold, d_norm, c_norm, ba, distribution, feasible flag.
    """
    candidates = [t / 1000 for t in range(1, 1000)]

    result = {
        "threshold": None,
        "d_norm": None,
        "c_norm": None,
        "ba": None,
        "distribution": "unknown",
        "feasible": False,
        "n_pareto": 0,
    }

    if len(cond_c_scores) < 10:
        return result

    metrics = compute_three_metrics(cond_c_scores, baseline_rows, candidates)
    if metrics is None:
        return result

    result["distribution"] = metrics["distribution"]

    pareto_idx = find_pareto_frontier(metrics, max_d=max_d, max_c=max_c, min_ba=min_ba)
    result["n_pareto"] = len(pareto_idx)

    if not pareto_idx:
        return result

    # Lexicographic: max BA first, then min(d_norm + c_norm)
    pts = [(i, metrics["ba"][i], metrics["d_norm"][i], metrics["c_norm"][i]) for i in pareto_idx]
    max_ba_val = max(p[1] for p in pts)
    best_ba_pts = [p for p in pts if abs(p[1] - max_ba_val) < 1e-9]
    best = min(best_ba_pts, key=lambda p: p[2] + p[3])
    idx, ba, d_norm, c_norm = best

    result["threshold"] = candidates[idx]
    result["d_norm"] = round(float(d_norm), 6)
    result["c_norm"] = round(float(c_norm), 6)
    result["ba"] = round(float(ba), 4)
    result["feasible"] = True

    return result


# ── Data collection ──────────────────────────────────────────────────────


def _collect_float_conflicts(records: list[dict]) -> set[str]:
    """Find conflicts with non-boolean scores in any condition (A, B, or C).

    Purely boolean conflicts (only 0/1 scores everywhere) don't need
    threshold optimization.
    """
    threshold_map = build_conflict_threshold_map()
    all_unique_scores: dict[str, set[float]] = {}
    for rec in records:
        if rec.get("error"):
            continue
        cid = rec.get("conflict_id")
        if cid is None or cid not in threshold_map:
            continue
        for key in ("verify_system_score", "verify_user_score"):
            score = rec.get(key)
            if score is not None:
                all_unique_scores.setdefault(cid, set()).add(score)

    return {
        cid for cid, scores in all_unique_scores.items()
        if not scores <= {0.0, 1.0}
    }


def _collect_cond_c_scores(
    records: list[dict],
    float_conflicts: set[str],
) -> dict[str, list[float]]:
    """Collect normalized condition C scores for float conflicts."""
    cond_c_scores: dict[str, list[float]] = {}
    for rec in records:
        if rec.get("error") or rec.get("condition") != "C":
            continue
        cid = rec.get("conflict_id")
        score = rec.get("verify_system_score")
        if cid is None or score is None:
            continue
        if cid not in float_conflicts:
            continue
        direction = rec.get("direction", "a_to_b")
        cond_c_scores.setdefault(cid, []).append(_normalize_score(score, direction))
    return cond_c_scores


def _collect_baseline_rows(
    records: list[dict],
) -> dict[str, list[dict]]:
    """Build baseline rows per conflict for BA computation."""
    threshold_map = build_conflict_threshold_map()
    baseline_groups: dict[tuple, list] = defaultdict(list)
    for rec in records:
        if rec.get("error"):
            continue
        cond = rec.get("condition")
        if cond not in ("A", "B"):
            continue
        cid = rec.get("conflict_id")
        if cid is None:
            continue
        direction = rec.get("direction", "a_to_b")
        verify_code = direction_to_verify_code(direction)
        baseline_groups[(cid, verify_code, cond)].append(rec)

    all_baseline_rows: dict[str, list[dict]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()
    for cid, verify_code, _ in sorted(baseline_groups.keys()):
        if (cid, verify_code) in seen_keys:
            continue
        seen_keys.add((cid, verify_code))
        info = threshold_map.get(cid)
        if info is None or verify_code not in info.sides:
            continue
        cond_a_recs = baseline_groups.get((cid, verify_code, "A"), [])
        cond_b_recs = baseline_groups.get((cid, verify_code, "B"), [])
        if not cond_a_recs or not cond_b_recs:
            continue
        sys_trying = [r["verify_system_score"] for r in cond_a_recs if r.get("verify_system_score") is not None]
        sys_ignoring = [r["verify_system_score"] for r in cond_b_recs if r.get("verify_system_score") is not None]
        usr_trying = [r["verify_user_score"] for r in cond_b_recs if r.get("verify_user_score") is not None]
        usr_ignoring = [r["verify_user_score"] for r in cond_a_recs if r.get("verify_user_score") is not None]
        sys_info = info.sides[verify_code]["system"]
        usr_info = info.sides[verify_code]["user"]
        for trying, ignoring, side_info in [
            (sys_trying, sys_ignoring, sys_info),
            (usr_trying, usr_ignoring, usr_info),
        ]:
            all_scores = trying + ignoring
            if not all_scores or set(all_scores) <= {0.0, 1.0}:
                continue
            all_baseline_rows[cid].append({
                "_trying": trying,
                "_ignoring": ignoring,
                "is_inverted": side_info.is_inverted,
            })

    return all_baseline_rows


# ── YAML update ──────────────────────────────────────────────────────────


def update_thresholds_yaml(
    yaml_path: str | Path,
    model_id: str,
    results: list[dict],
    pareto_caps: dict,
    *,
    refresh_caps: bool = False,
    force_overwrite_audit: bool = False,
):
    """Add per-model threshold section to thresholds.yaml.

    When `refresh_caps` is True, `_meta.pareto_caps` is overwritten with the
    caps used by this run. When False (the default — used by partial
    `--conflicts` updates), existing `_meta.pareto_caps` is preserved so the
    section keeps the caps that produced the majority of its entries.

    **Audit-lock protection:** entries with `source: audit_*` are treated as
    locked — they were committed by an audit subagent's semantic analysis and
    must not be silently overwritten by Pareto re-optimization. Locked entries
    are skipped (kept as-is) unless `force_overwrite_audit=True`. The skipped
    entries are returned in the result dict's `skipped_locked` list so the
    caller can surface them to the user.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    safe_id = model_id.replace("/", "_")
    model_section = {
        "_meta": {
            "last_updated": date.today().isoformat(),
            "pareto_caps": {
                "max_d_norm": pareto_caps["max_d"],
                "max_c_norm": pareto_caps["max_c"],
                "min_ba": pareto_caps["min_ba"],
            },
        },
    }

    for r in results:
        entry = {"threshold": round(float(r["threshold"]), 3)}
        # Provenance — every entry written by the optimizer is sourced "pareto"
        # so the audit's threshold-provenance dashboard can distinguish
        # optimizer picks from audit-vetted overrides.
        entry["source"] = "pareto"
        if r.get("d_norm") is not None:
            entry["d_norm"] = round(float(r["d_norm"]), 6)
        if r.get("c_norm") is not None:
            entry["c_norm"] = round(float(r["c_norm"]), 6)
        if r.get("ba") is not None:
            entry["ba"] = round(float(r["ba"]), 4)
        if r.get("max_ba") is not None:
            entry["max_ba"] = round(float(r["max_ba"]), 4)
        entry["distribution"] = r.get("distribution", "unknown")
        entry["feasible"] = r.get("feasible", False)
        if r.get("fallback"):
            entry["fallback"] = r["fallback"]
        entry["ambiguous"] = bool(r.get("ambiguous", False))
        model_section[r["conflict_id"]] = entry

    skipped_locked: list[str] = []
    if safe_id in data and isinstance(data[safe_id], dict):
        # Merge: update only the conflicts in results, preserve the rest.
        # Partial runs (`--conflicts X,Y`) keep existing `_meta.pareto_caps`
        # so they don't get clobbered by CLI defaults that weren't used to
        # produce the rest of the section. Full-section refreshes
        # (`refresh_caps=True`) overwrite the caps to reflect this run.
        existing = data[safe_id]
        if refresh_caps or "_meta" not in existing:
            existing["_meta"] = model_section["_meta"]
        else:
            existing["_meta"]["last_updated"] = date.today().isoformat()
        for cid, entry in model_section.items():
            if cid == "_meta":
                continue
            # Audit-lock check: refuse to silently overwrite entries that an
            # audit subagent committed via apply_audit_recommendation, unless
            # force_overwrite_audit is set.
            current = existing.get(cid)
            if (
                isinstance(current, dict)
                and isinstance(current.get("source"), str)
                and current["source"].startswith("audit_")
                and not force_overwrite_audit
            ):
                skipped_locked.append(cid)
                continue
            existing[cid] = entry
    else:
        data[safe_id] = model_section
    data["_meta"] = {"last_updated": date.today().isoformat()}

    with open(yaml_path, "w") as f:
        f.write("# Float conflict thresholds — single source of truth.\n")
        f.write("# Boolean conflicts don't have thresholds and are not listed here.\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return {"skipped_locked": skipped_locked}


# ── Report generation ───────────────────────────────────────────────────


def generate_report(
    model_id: str,
    results: list[dict],
    n_cond_c: int,
    pareto_caps: dict,
) -> str:
    """Generate markdown report from Pareto optimization results."""
    today = date.today().isoformat()
    n_float = len(results)
    feasible = [r for r in results if r.get("feasible")]
    infeasible = [r for r in results if not r.get("feasible")]

    lines = [
        f"# Per-Model Threshold Optimization: {model_id}",
        "",
        f"**Date:** {today}",
        f"**Records:** {n_cond_c} condition C records analyzed",
        f"**Float conflicts:** {n_float} ({len(feasible)} feasible, {len(infeasible)} infeasible)",
        f"**Pareto caps:** d_norm ≤ {pareto_caps['max_d']}, c_norm ≤ {pareto_caps['max_c']}, BA ≥ {pareto_caps['min_ba']}",
        "",
        "## Results",
        "",
        "| Conflict | T_sel | T_curr | d_norm | c_norm | BA | Dist | Change |",
        "|----------|-------|--------|--------|--------|-----|------|--------|",
    ]

    for r in sorted(results, key=lambda x: x["conflict_id"]):
        t_cur = f"{r['current_threshold']:.3f}" if r.get("current_threshold") is not None else "—"
        if r.get("feasible"):
            t_sel = f"{r['threshold']:.3f}"
            d = f"{r['d_norm']:.4f}"
            c = f"{r['c_norm']:.4f}"
            ba = f"{r['ba']:.3f}"
            if r.get("current_threshold") is not None:
                change = r["threshold"] - r["current_threshold"]
                change_str = f"{change:+.3f}"
            else:
                change_str = "new"
        else:
            t_sel = "—"
            d = "—"
            c = "—"
            ba = "—"
            change_str = "infeasible"

        lines.append(
            f"| {r['conflict_id']} | {t_sel} | {t_cur} "
            f"| {d} | {c} | {ba} | {r.get('distribution', '?')} | {change_str} |"
        )

    if infeasible:
        lines.extend(["", "## Infeasible Conflicts", ""])
        for r in sorted(infeasible, key=lambda x: x["conflict_id"]):
            lines.append(f"- **{r['conflict_id']}**: {r.get('distribution', '?')}")

    lines.append("")
    return "\n".join(lines)


# ── Default suggestion ───────────────────────────────────────────────────


def suggest_defaults(yaml_path: str | Path) -> dict[str, float]:
    """Compute suggested default thresholds as median across per-model sections.

    Returns dict mapping conflict_id -> suggested_default_threshold.
    Only includes conflicts present in at least one per-model section.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    # Collect per-model thresholds
    per_conflict: dict[str, list[float]] = defaultdict(list)
    for key, section in data.items():
        if key in ("_meta", "default") or not isinstance(section, dict):
            continue
        for cid, entry in section.items():
            if cid == "_meta":
                continue
            if isinstance(entry, dict):
                t = entry.get("threshold")
                if t is not None and entry.get("feasible", True):
                    per_conflict[cid].append(float(t))
            elif isinstance(entry, (int, float)):
                per_conflict[cid].append(float(entry))

    return {
        cid: round(median(thresholds), 3)
        for cid, thresholds in sorted(per_conflict.items())
        if thresholds
    }


# ── Main pipeline ────────────────────────────────────────────────────────


def run_pipeline(
    results_path: str | Path,
    model_id: str | None = None,
    *,
    max_d: float = DEFAULT_MAX_D_NORM,
    max_c: float = DEFAULT_MAX_C_NORM,
    min_ba: float = DEFAULT_MIN_BA,
    conflicts: list[str] | None = None,
) -> tuple[str, list[dict], int, dict]:
    """Run the full per-model threshold optimization pipeline.

    Args:
        conflicts: If provided, only optimize these conflict IDs.
            Others are skipped entirely.

    Returns (model_id, results_list, n_cond_c_records, pareto_caps).
    """
    records = load_records(results_path)

    if model_id is None:
        for rec in records:
            if rec.get("model"):
                model_id = rec["model"]
                break
    if model_id is None:
        raise ValueError("Could not determine model ID from records")

    pareto_caps = {"max_d": max_d, "max_c": max_c, "min_ba": min_ba}

    # Find float conflicts (non-boolean scores across all conditions)
    float_conflicts = _collect_float_conflicts(records)
    if conflicts:
        requested = set(conflicts)
        float_conflicts = float_conflicts & requested

    # Collect condition C scores
    cond_c_scores = _collect_cond_c_scores(records, float_conflicts)

    # Collect baseline rows for BA computation
    baseline_rows = _collect_baseline_rows(records)

    # Compute baseline optimal ranges (for fallback when Pareto is infeasible)
    baseline_ranges = compute_baseline_ranges(records)

    # Load current thresholds for comparison
    from ..config.thresholds import get_threshold, get_threshold_info

    # Count condition C records
    n_cond_c = sum(len(scores) for scores in cond_c_scores.values())

    # Run Pareto optimization for each float conflict
    results = []
    for cid in sorted(float_conflicts):
        scores = cond_c_scores.get(cid)
        if scores is None:
            continue
        scores_arr = np.array(scores)
        bl_rows = baseline_rows.get(cid, [])
        current_t = get_threshold(cid, model_id)
        current_info = get_threshold_info(cid, model_id) or {}

        r = select_threshold(
            scores_arr,
            bl_rows,
            max_d=max_d,
            max_c=max_c,
            min_ba=min_ba,
        )
        r["conflict_id"] = cid
        r["current_threshold"] = current_t
        r["current_source"] = current_info.get("source")
        r["current_is_audit_locked"] = bool(current_info.get("is_audit_locked"))
        max_ba = baseline_ranges.get(cid, {}).get("ba")
        r["max_ba"] = round(float(max_ba), 4) if max_ba is not None else None

        # Fallback for infeasible: compute both valley and baseline midpoint,
        # pick whichever has higher BA. Ties go to valley (respects cond C shape).
        if not r["feasible"]:
            valley_t = _find_valley_location(scores_arr)
            valley_ba = None
            if valley_t is not None and bl_rows:
                valley_ba = round(_mean_ba_at_threshold(bl_rows, valley_t), 4)

            baseline_t = None
            baseline_ba = None
            if cid in baseline_ranges:
                br = baseline_ranges[cid]
                baseline_t = round((br["opt_low"] + br["opt_high"]) / 2, 3)
                baseline_ba = br["ba"]

            # Pick best: max BA, ties to valley
            if valley_t is not None and baseline_t is not None:
                if valley_ba >= baseline_ba:
                    r["threshold"] = valley_t
                    r["ba"] = valley_ba
                    r["fallback"] = "valley"
                else:
                    r["threshold"] = baseline_t
                    r["ba"] = baseline_ba
                    r["fallback"] = "baseline_midpoint"
            elif valley_t is not None:
                r["threshold"] = valley_t
                r["ba"] = valley_ba
                r["fallback"] = "valley"
            elif baseline_t is not None:
                r["threshold"] = baseline_t
                r["ba"] = baseline_ba
                r["fallback"] = "baseline_midpoint"

        r["ambiguous"] = is_ambiguous(r, r.get("max_ba"))
        results.append(r)

    return model_id, results, n_cond_c, pareto_caps


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Per-model threshold optimizer using Pareto frontier analysis."
    )
    parser.add_argument("results_jsonl", help="Path to model results JSONL file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for markdown report (default: per-model dir under calibration/output/)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write per-model thresholds to thresholds.yaml (default: dry run)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (extracted from JSONL if not provided)",
    )
    parser.add_argument(
        "--min-ba",
        type=float,
        default=DEFAULT_MIN_BA,
        help=f"Minimum BA feasibility cap (default: {DEFAULT_MIN_BA})",
    )
    parser.add_argument(
        "--max-d-norm",
        type=float,
        default=DEFAULT_MAX_D_NORM,
        help=f"Maximum d_norm feasibility cap (default: {DEFAULT_MAX_D_NORM})",
    )
    parser.add_argument(
        "--max-c-norm",
        type=float,
        default=DEFAULT_MAX_C_NORM,
        help=f"Maximum c_norm feasibility cap (default: {DEFAULT_MAX_C_NORM})",
    )
    parser.add_argument(
        "--conflicts",
        default=None,
        help="Comma-separated conflict IDs to optimize (default: all float conflicts)",
    )
    parser.add_argument(
        "--force-overwrite-audit",
        action="store_true",
        help=(
            "Overwrite per-conflict entries whose source is audit_* (set by "
            "/calibration-audit-cond-c Phase 6.5). Default: skip them and "
            "report which ones were preserved. Use only when you know the "
            "audit-derived T is wrong and you want Pareto to re-pick."
        ),
    )
    args = parser.parse_args()

    conflict_list = [c.strip() for c in args.conflicts.split(",")] if args.conflicts else None

    model_id, results, n_cond_c, pareto_caps = run_pipeline(
        args.results_jsonl,
        model_id=args.model,
        max_d=args.max_d_norm,
        max_c=args.max_c_norm,
        min_ba=args.min_ba,
        conflicts=conflict_list,
    )

    # Print summary table
    n_feas = sum(1 for r in results if r.get("feasible"))
    n_fallback = sum(1 for r in results if not r.get("feasible") and r.get("fallback"))
    n_unresolved = len(results) - n_feas - n_fallback
    n_ambig = sum(1 for r in results if r.get("ambiguous"))
    n_locked = sum(1 for r in results if r.get("current_is_audit_locked"))
    print(f"\nPer-model thresholds for: {model_id}")
    print(f"Pareto caps: d_norm ≤ {pareto_caps['max_d']}, c_norm ≤ {pareto_caps['max_c']}, BA ≥ {pareto_caps['min_ba']}")
    print(f"Condition C records: {n_cond_c}")
    print(
        f"Float conflicts: {len(results)} "
        f"({n_feas} feasible, {n_fallback} fallback, {n_unresolved} unresolved, "
        f"{n_ambig} ambiguous, {n_locked} audit-locked)"
    )
    print()
    print(
        f"{'Conflict':<35} {'T_sel':>7} {'T_curr':>7} {'Src':>5} {'d_norm':>8} {'c_norm':>8} "
        f"{'BA':>7} {'maxBA':>7} {'Dist':>10} {'Feas':>5} {'Change':>8} {'Ambig':>6}"
    )
    print("-" * 125)
    for r in sorted(results, key=lambda x: x["conflict_id"]):
        t_cur = f"{r['current_threshold']:.3f}" if r.get("current_threshold") is not None else "—"
        dist = r.get("distribution", "?")
        max_ba_str = f"{r['max_ba']:.3f}" if r.get("max_ba") is not None else "—"
        amb_str = "[AMBIG]" if r.get("ambiguous") else ""
        feas_str = "Y" if r.get("feasible") else "N"
        # Source abbreviation. Locked entries are marked 🔒 so the operator
        # immediately sees the Pareto pick won't be applied without --force.
        src = r.get("current_source")
        if r.get("current_is_audit_locked"):
            src_str = "🔒aud"
        elif src == "pareto":
            src_str = "par"
        elif src is None:
            src_str = "—"
        else:
            src_str = src[:5]
        # When an entry is audit-locked, the Pareto pick won't be applied —
        # mark the change column "(locked)" so the operator doesn't read the
        # diff as a pending change.
        is_locked = r.get("current_is_audit_locked")
        if r.get("feasible"):
            change = r["threshold"] - r["current_threshold"] if r.get("current_threshold") is not None else None
            change_str = "(locked)" if is_locked else (f"{change:+.3f}" if change is not None else "new")
            print(
                f"{r['conflict_id']:<35} {r['threshold']:>7.3f} {t_cur:>7} {src_str:>5} "
                f"{r['d_norm']:>8.4f} {r['c_norm']:>8.4f} {r['ba']:>7.3f} "
                f"{max_ba_str:>7} {dist:>10} {feas_str:>5} {change_str:>8} {amb_str:>6}"
            )
        elif r.get("fallback"):
            change = r["threshold"] - r["current_threshold"] if r.get("current_threshold") is not None else None
            change_str = "(locked)" if is_locked else (f"{change:+.3f}" if change is not None else "new")
            ba_str = f"{r['ba']:.3f}" if r.get("ba") is not None else "—"
            print(
                f"{r['conflict_id']:<35} {r['threshold']:>7.3f} {t_cur:>7} {src_str:>5} "
                f"{'—':>8} {'—':>8} {ba_str:>7} "
                f"{max_ba_str:>7} {dist:>10} {feas_str:>5} {change_str:>8} {amb_str:>6}  ← {r['fallback']}"
            )
        else:
            print(
                f"{r['conflict_id']:<35} {'—':>7} {t_cur:>7} {src_str:>5} "
                f"{'—':>8} {'—':>8} {'—':>7} "
                f"{max_ba_str:>7} {dist:>10} {feas_str:>5} {'infeas':>8} {amb_str:>6}"
            )

    print()
    print(
        "Legend: Src=🔒aud means the current YAML entry is audit-locked "
        "(Pareto pick won't be applied without --force-overwrite-audit); "
        "Src=par means optimizer-derived; Src=— means legacy entry without provenance. "
        "Feas=Y means Pareto-feasible (ships as-is); Feas=N + fallback means "
        "infeasible (used valley or baseline_midpoint); Ambig=[AMBIG] means the audit's "
        "Phase 2.5 should re-derive a threshold from semantic samples."
    )

    # Optionally update thresholds.yaml
    yaml_path = Path(__file__).parent.parent / "config" / "thresholds.yaml"
    if args.update:
        # Write feasible + fallback results (skip truly unresolved).
        # Full re-optimization (no --conflicts) refreshes _meta.pareto_caps;
        # partial updates preserve them.
        feasible_results = [r for r in results if r.get("feasible") or r.get("fallback")]
        write_result = update_thresholds_yaml(
            yaml_path, model_id, feasible_results, pareto_caps,
            refresh_caps=conflict_list is None,
            force_overwrite_audit=args.force_overwrite_audit,
        ) or {}
        skipped_locked = write_result.get("skipped_locked", [])
        if skipped_locked:
            print(
                f"\n[lock] Preserved {len(skipped_locked)} audit-set entr"
                f"{'y' if len(skipped_locked) == 1 else 'ies'} "
                f"(source: audit_*) — Pareto pick was NOT written for:"
            )
            for cid in skipped_locked:
                print(f"  - {cid}")
            print(
                "Re-run with --force-overwrite-audit to overwrite these, "
                "or use revert_audit_recommendation() to restore the prior "
                "Pareto pick first."
            )
        print(f"Updated thresholds.yaml with per-model section for {model_id}")

        # Show suggested defaults
        suggestions = suggest_defaults(yaml_path)
        current_defaults = yaml.safe_load(open(yaml_path)).get("default", {})
        changes = {
            cid: t for cid, t in suggestions.items()
            if current_defaults.get(cid) != t
        }
        if changes:
            print(f"\nSuggested default updates (median across models):")
            print(f"{'Conflict':<35} {'Current':>8} {'Suggested':>10}")
            print("-" * 55)
            for cid, t in sorted(changes.items()):
                cur = current_defaults.get(cid)
                cur_str = f"{cur:.3f}" if cur is not None else "—"
                print(f"{cid:<35} {cur_str:>8} {t:>10.3f}")
            print("\nUse --update-defaults to apply these suggestions.")
    else:
        print("Dry run — use --update to write to thresholds.yaml")


if __name__ == "__main__":
    main()
