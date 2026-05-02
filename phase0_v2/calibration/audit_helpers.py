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
from .per_model_thresholds import (
    _collect_baseline_rows,
    _collect_cond_c_scores,
    _collect_float_conflicts,
    _mean_ba_at_threshold,
    _normalize_score,
    compute_baseline_ranges,
    compute_three_metrics,
    is_ambiguous,
    select_threshold,
)

_AUDIT_BASE = Path("phase0_v2/calibration/output/condition_c_audit")
_THRESHOLDS_YAML = Path("phase0_v2/config/thresholds.yaml")


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


def sample_by_score_band(
    records: list[dict],
    conflict_id: str,
    *,
    n_bands: int = 8,
    samples_per_band: int = 4,
    bands: list[tuple[float, float]] | None = None,
    condition: str = "C",
    direction: str | None = None,
    seed: int = 0,
) -> dict[tuple[float, float], list[dict]]:
    """Stratified-sample records by verify_system_score band on the constraint-a scale.

    Used by the audit subagent's Phase 4.5 (semantic-threshold investigation):
    the agent first calls this with default `n_bands=8` to map the landscape,
    then re-calls with explicit `bands=[(lo, hi), ...]` and a larger
    `samples_per_band` to drill down on suspect ranges.

    Direction is normalized via `_normalize_score`, so the returned "score" is
    always on the constraint-a scale (high = a satisfied, low = b satisfied).

    Args:
        records: All records loaded from a results JSONL.
        conflict_id: Filter to this conflict.
        n_bands: Number of equal-width bands over [0, 1] when `bands` is None.
        samples_per_band: Cap per band; bands with more matching records are
            randomly subsampled (deterministic via `seed`).
        bands: Optional explicit list of (lo, hi) tuples. Overrides `n_bands`.
            Bands may overlap or be non-contiguous — useful for drill-down
            (e.g., `[(0.40, 0.45), (0.45, 0.50)]`). Each record is assigned to
            *every* band whose interval contains its score, so a record can
            appear in multiple buckets when bands overlap.
        condition: Filter to this condition (default "C"). Pass "A", "B", or
            "all" to override.
        direction: Optional filter to "a_to_b" or "b_to_a".
        seed: RNG seed for reproducible subsampling.

    Returns:
        dict keyed by (lo, hi) with each value a list of:
        {"record": <original record>, "score": <normalized constraint-a score>}.
        Bands with no matching records appear with an empty list so callers
        can observe sparse-band gaps explicitly.
    """
    import random

    if bands is None:
        edges = [i / n_bands for i in range(n_bands + 1)]
        bands = [
            (round(edges[i], 6), round(edges[i + 1], 6))
            for i in range(n_bands)
        ]
    else:
        bands = [(round(float(lo), 6), round(float(hi), 6)) for lo, hi in bands]

    out: dict[tuple[float, float], list[dict]] = {b: [] for b in bands}

    for r in records:
        if r.get("conflict_id") != conflict_id:
            continue
        if r.get("error") is not None:
            continue
        if condition != "all" and r.get("condition") != condition:
            continue
        if direction is not None and r.get("direction") != direction:
            continue
        raw_score = r.get("verify_system_score")
        if raw_score is None:
            continue
        score = _normalize_score(float(raw_score), r.get("direction", "a_to_b"))
        for (lo, hi) in bands:
            # [lo, hi); for the special case hi == 1.0, include 1.0 itself.
            if lo <= score < hi or (hi == 1.0 and score == 1.0):
                out[(lo, hi)].append({"record": r, "score": score})

    rng = random.Random(seed)
    for key, items in out.items():
        if len(items) > samples_per_band:
            out[key] = rng.sample(items, samples_per_band)

    return out


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
# Apply / revert audit threshold recommendations (Phase 2.5 auto-apply)
# ---------------------------------------------------------------------------


def _compute_metrics_at_threshold(
    records: list[dict],
    conflict_id: str,
    T: float,
) -> dict:
    """Recompute (ba, max_ba, d_norm, c_norm, feasible, ambiguous, distribution)
    at threshold T for one (model, conflict) pair, using the model's records.

    Returns the same fields the per-model thresholds optimizer would write to
    YAML — but evaluated at a specific T rather than the Pareto pick. Used by
    apply_audit_recommendation to populate post-change metadata.
    """
    float_conflicts = _collect_float_conflicts(records) & {conflict_id}
    if not float_conflicts:
        return {
            "ba": None, "max_ba": None,
            "d_norm": None, "c_norm": None,
            "feasible": None, "ambiguous": None,
            "distribution": "unknown",
        }

    cond_c_scores = _collect_cond_c_scores(records, float_conflicts).get(conflict_id, [])
    baseline_rows = _collect_baseline_rows(records).get(conflict_id, [])
    baseline_ranges = compute_baseline_ranges(records).get(conflict_id, {})

    if len(cond_c_scores) < 10:
        return {
            "ba": None, "max_ba": baseline_ranges.get("ba"),
            "d_norm": None, "c_norm": None,
            "feasible": None, "ambiguous": None,
            "distribution": "unknown",
        }

    candidates = [t / 1000 for t in range(1, 1000)]
    metrics = compute_three_metrics(np.array(cond_c_scores), baseline_rows, candidates)
    if metrics is None:
        return {
            "ba": None, "max_ba": baseline_ranges.get("ba"),
            "d_norm": None, "c_norm": None,
            "feasible": None, "ambiguous": None,
            "distribution": "unknown",
        }

    # Snap T to nearest candidate for index lookup
    idx = min(range(len(candidates)), key=lambda i: abs(candidates[i] - T))
    d_norm = float(metrics["d_norm"][idx])
    c_norm = float(metrics["c_norm"][idx])
    ba = float(metrics["ba"][idx]) if baseline_rows else (
        round(_mean_ba_at_threshold(baseline_rows, T), 4) if baseline_rows else 1.0
    )

    max_ba = baseline_ranges.get("ba")
    result = {
        "ba": round(ba, 4),
        "max_ba": round(float(max_ba), 4) if max_ba is not None else None,
        "d_norm": round(d_norm, 6),
        "c_norm": round(c_norm, 6),
        "distribution": metrics["distribution"],
        # `feasible` here means the new T meets the optimizer's caps. We use
        # the same DEFAULT caps the optimizer ships with so this stays in sync.
        "feasible": d_norm <= 0.02 and c_norm <= 0.02 and ba >= 0.95,
    }
    result["ambiguous"] = is_ambiguous(result, result["max_ba"])
    return result


def apply_audit_recommendation(
    audit_json_path: str | Path,
    *,
    yaml_path: str | Path | None = None,
    rescore: bool = True,
    require_high_confidence: bool = True,
) -> dict:
    """Write a Phase-2.5 threshold recommendation to thresholds.yaml + rescore.

    Reads the audit JSON, extracts `semantic_threshold.T_recommended` and
    associated metadata, writes the new threshold to thresholds.yaml under
    the model's per-model section with full provenance (`source`, `audit_run`,
    `previous` snapshot for one-command revert). Then optionally rescores the
    JSONL for that single conflict so labels reflect the new threshold.

    Skips (returns `applied: False, reason: ...`) when:
      - `semantic_threshold.ran != true` (not an ambiguous-float audit)
      - `T_recommended == T_pareto` (no change recommended)
      - `recommendation_confidence != "high"` (only when require_high_confidence)

    The YAML write is idempotent: re-applying the same audit JSON produces
    no further change. Concurrency note: the assumption is that two subagents
    rarely write to thresholds.yaml at the same instant (each owns one
    (model, conflict) pair, and audit batches are small). If contention
    becomes a concern, wrap this call in a file lock at the call site.

    Returns: dict with `applied`, `reason` (if skipped), `T_old`, `T_new`,
    `rescored_records` (count, if rescore=True), `yaml_path`.
    """
    import yaml

    audit_json_path = Path(audit_json_path)
    with open(audit_json_path) as f:
        audit = json.load(f)

    semantic = audit.get("semantic_threshold") or {}
    if not semantic.get("ran"):
        return {"applied": False, "reason": "semantic_threshold.ran is false"}

    T_pareto = semantic.get("T_pareto")
    T_new = semantic.get("T_recommended")
    if T_new is None or T_pareto is None:
        return {"applied": False, "reason": "missing T_recommended or T_pareto"}
    if abs(float(T_new) - float(T_pareto)) < 1e-9:
        return {"applied": False, "reason": "T_recommended == T_pareto", "T_old": T_pareto, "T_new": T_new}

    confidence = semantic.get("recommendation_confidence")
    if require_high_confidence and confidence != "high":
        return {
            "applied": False,
            "reason": f"recommendation_confidence={confidence!r} (require high)",
            "T_old": T_pareto, "T_new": T_new,
        }

    conflict_id = audit["conflict_id"]
    model_label = audit.get("model")  # e.g. "google_gemma-4-E2B-it" (safe id)
    if not model_label:
        return {"applied": False, "reason": "audit JSON missing 'model' field"}

    yaml_path = Path(yaml_path) if yaml_path is not None else _THRESHOLDS_YAML
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    section = data.get(model_label)
    if not isinstance(section, dict):
        return {"applied": False, "reason": f"no per-model section for {model_label} in YAML"}

    existing = section.get(conflict_id)
    # Existing entry can be a plain float (legacy) or a dict (new schema).
    if isinstance(existing, (int, float)):
        existing = {"threshold": float(existing), "source": "legacy_scalar"}
    elif existing is None:
        return {"applied": False, "reason": f"no entry for {conflict_id} in {model_label} section"}

    # Build the previous snapshot (everything currently in the entry, minus
    # nested `previous` from any earlier apply — we keep only the most recent
    # rollback target so the YAML doesn't grow unboundedly).
    previous = {k: v for k, v in existing.items() if k != "previous"}

    # Determine the audit run timestamp for provenance.
    timestamp = audit_json_path.stem.replace("audit_", "")  # e.g. "0501_1811"

    # Recompute metrics at the new T using the model's results JSONL.
    # Convention: the JSONL lives under data/results/{model_label}_results.jsonl.
    results_path = Path(f"phase0_v2/data/results/{model_label}_results.jsonl")
    new_metrics = {}
    if results_path.exists():
        records = load_records(str(results_path))
        new_metrics = _compute_metrics_at_threshold(records, conflict_id, float(T_new))

    new_entry = {
        "threshold": round(float(T_new), 3),
        "source": f"audit_{timestamp}",
        "audit_run": str(audit_json_path),
        "previous": previous,
    }
    # Merge in recomputed metrics where available.
    for k in ("ba", "max_ba", "d_norm", "c_norm", "distribution", "feasible", "ambiguous"):
        if k in new_metrics and new_metrics[k] is not None:
            new_entry[k] = new_metrics[k]
    # Drop any stale `fallback` field — the new T is no longer a Pareto fallback.
    section[conflict_id] = new_entry

    # Write YAML back with the same formatting convention as per_model_thresholds.
    with open(yaml_path, "w") as f:
        f.write("# Float conflict thresholds — single source of truth.\n")
        f.write("# Boolean conflicts don't have thresholds and are not listed here.\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Invalidate the cached YAML so subsequent get_threshold calls (notably the
    # rescore step's activate_model) read the freshly-written file.
    from ..config.conflict_config import _load_thresholds
    _load_thresholds.cache_clear()

    result = {
        "applied": True,
        "T_old": previous.get("threshold"),
        "T_new": new_entry["threshold"],
        "model": model_label,
        "conflict": conflict_id,
        "yaml_path": str(yaml_path),
    }

    if rescore and results_path.exists():
        from . import rescore as rescore_mod
        # Use the same file as input + output (in-place) and pass the model_id
        # in slash form so activate_model() can apply the just-written T.
        model_id_slash = model_label.replace("_", "/", 1)
        rescore_mod.main([
            str(results_path), str(results_path),
            "--model", model_id_slash,
        ])
        result["rescored_records"] = "all"
        result["rescored_path"] = str(results_path)

    return result


def revert_audit_recommendation(
    *,
    model_label: str,
    conflict_id: str,
    yaml_path: str | Path | None = None,
    rescore: bool = True,
) -> dict:
    """Restore the `previous` snapshot for a (model, conflict) entry.

    Copies `previous.*` back over the top-level fields and removes the
    `previous` block, then optionally rescores. Use after an auto-applied
    audit recommendation that turned out to be wrong.

    Returns dict with `reverted`, `T_old` (the audit-applied T), `T_new`
    (the restored previous T), `yaml_path`.
    """
    import yaml

    yaml_path = Path(yaml_path) if yaml_path is not None else _THRESHOLDS_YAML
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    section = data.get(model_label)
    if not isinstance(section, dict):
        return {"reverted": False, "reason": f"no per-model section for {model_label}"}

    entry = section.get(conflict_id)
    if not isinstance(entry, dict):
        return {"reverted": False, "reason": f"entry for {conflict_id} is not a dict"}

    previous = entry.get("previous")
    if not isinstance(previous, dict):
        return {"reverted": False, "reason": "no `previous` snapshot to restore"}

    T_old = entry.get("threshold")
    section[conflict_id] = dict(previous)  # new top-level = previous; no nested previous

    with open(yaml_path, "w") as f:
        f.write("# Float conflict thresholds — single source of truth.\n")
        f.write("# Boolean conflicts don't have thresholds and are not listed here.\n\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Invalidate the cached YAML so subsequent get_threshold calls (notably the
    # rescore step's activate_model) read the freshly-written file.
    from ..config.conflict_config import _load_thresholds
    _load_thresholds.cache_clear()

    result = {
        "reverted": True,
        "T_old": T_old,
        "T_new": previous.get("threshold"),
        "model": model_label,
        "conflict": conflict_id,
        "yaml_path": str(yaml_path),
    }

    if rescore:
        results_path = Path(f"phase0_v2/data/results/{model_label}_results.jsonl")
        if results_path.exists():
            from . import rescore as rescore_mod
            model_id_slash = model_label.replace("_", "/", 1)
            rescore_mod.main([
                str(results_path), str(results_path),
                "--model", model_id_slash,
            ])
            result["rescored_path"] = str(results_path)

    return result


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
    semantic = data.get("semantic_threshold") or {}

    diag_at_yaml = data.get("diagnosis_at_yaml_t") or {}

    return {
        "id": data.get("conflict_id", "???"),
        "type": verifier.get("type", "---"),
        "severity": (
            severity_obj.get("rating", "---")
            if isinstance(severity_obj, dict)
            else str(severity_obj)
        ),
        "severity_at_yaml_t": data.get("severity_at_yaml_t"),
        "error_pct": diag.get("overall_error_pct", 0),
        "error_count": diag.get("overall_error_count", 0),
        "error_n": diag.get("overall_n", 0),
        "error_pct_at_yaml_t": diag_at_yaml.get("overall_error_pct"),
        "ba": pareto.get("ba"),
        "max_ba": pareto.get("max_ba"),
        "d_norm": pareto.get("d_norm"),
        "c_norm": pareto.get("c_norm"),
        "dist": pareto.get("distribution", "---"),
        "integrity": pareto.get("baseline_integrity", "---"),
        "feasible": pareto.get("feasible", True),
        "fallback": pareto.get("fallback"),
        "ambiguous": pareto.get("ambiguous", False),
        "recommended_action": data.get("recommended_action", "None"),
        "fixes": fixes,
        "open_questions": data.get("open_questions", []),
        "semantic": {
            "ran": semantic.get("ran", False),
            "trigger_reason": semantic.get("trigger_reason"),
            "T_pareto": semantic.get("T_pareto"),
            "T_recommended": semantic.get("T_recommended"),
            "recommended_agent_ba": semantic.get("recommended_agent_ba"),
            "delta_vs_pareto": semantic.get("delta_vs_pareto"),
            "confidence": semantic.get("recommendation_confidence"),
        } if semantic else {"ran": False},
    }


def _fmt(val: float | None, width: int = 3) -> str:
    """Format a float or return '---'."""
    if val is None:
        return "---"
    if width == 3:
        return f"{val:.3f}"
    return f"{val:.6f}"


def _collect_threshold_provenance(model_label: str) -> list[dict]:
    """Read thresholds.yaml and pull per-conflict provenance for `model_label`.

    Returns one row per conflict in the model's per-model section:
      {
        "id": conflict_id,
        "threshold": float | None,
        "source": "audit_<ts>" | "pareto" | None,
        "source_kind": "audit" | "pareto" | "legacy",
        "set_on": "<ts>" or None,         # timestamp parsed from audit source
        "audit_run": str | None,          # path to the audit JSON for audit-set entries
      }

    Used by build_audit_summary's "Threshold Provenance" section so dashboards
    show at a glance which entries are agent-vetted vs. Pareto-derived.
    """
    import yaml

    if not _THRESHOLDS_YAML.exists():
        return []
    with open(_THRESHOLDS_YAML) as f:
        data = yaml.safe_load(f) or {}

    section = data.get(model_label)
    if not isinstance(section, dict):
        return []

    rows: list[dict] = []
    for cid, entry in section.items():
        if cid == "_meta":
            continue
        if isinstance(entry, (int, float)):
            rows.append({
                "id": cid,
                "threshold": float(entry),
                "source": None,
                "source_kind": "legacy",
                "set_on": None,
                "audit_run": None,
            })
            continue
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if isinstance(source, str) and source.startswith("audit_"):
            kind = "audit"
            set_on = source[len("audit_"):] or None
        elif source == "pareto":
            kind = "pareto"
            set_on = None
        else:
            kind = "legacy"
            set_on = None
        rows.append({
            "id": cid,
            "threshold": entry.get("threshold"),
            "source": source,
            "source_kind": kind,
            "set_on": set_on,
            "audit_run": entry.get("audit_run"),
        })
    return rows


def _short_action(rec: dict) -> str:
    """Return a short action label for the health table."""
    if rec["fixes"]:
        return rec["fixes"][0].get("action_type", "Adjust verifier")
    ra = rec["recommended_action"]
    if not ra:
        return "None"
    # New schema: dict with `summary` + `type`. Old schema: free-form string.
    if isinstance(ra, dict):
        action_type = ra.get("type") or ""
        if action_type and action_type != "none":
            return action_type
        summary = ra.get("summary") or ""
        return summary[:30] if summary else "None"
    if not isinstance(ra, str):
        return "None"
    if ra.lower().startswith("none"):
        return "None"
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

    # --- Threshold provenance (read from YAML, not audit JSONs, so it reflects current state) ---
    provenance_rows = _collect_threshold_provenance(model_label)
    if provenance_rows:
        n_audit = sum(1 for r in provenance_rows if r["source_kind"] == "audit")
        n_pareto = sum(1 for r in provenance_rows if r["source_kind"] == "pareto")
        n_legacy = sum(1 for r in provenance_rows if r["source_kind"] == "legacy")
        lines.append("## Threshold Provenance")
        lines.append("")
        lines.append(
            f"Per-conflict threshold sources currently in `thresholds.yaml` for this model: "
            f"**{n_audit} audit-locked**, {n_pareto} Pareto-derived, {n_legacy} legacy/unknown. "
            f"Audit-locked entries (`source: audit_*`) are protected from `--update` overwrites "
            f"and from Phase 2.5 re-derivation; use `revert_audit_recommendation()` to release them."
        )
        lines.append("")
        lines.append(
            "| Conflict | Threshold | Source | Set on | Locked? |"
        )
        lines.append(
            "|----------|-----------|--------|--------|---------|"
        )
        # Sort: audit-locked first (most interesting for human review), then by conflict id
        provenance_rows.sort(key=lambda r: (0 if r["source_kind"] == "audit" else 1, r["id"]))
        for r in provenance_rows:
            t = _fmt(r["threshold"])
            src = r["source"] or "(none)"
            set_on = r["set_on"] or "—"
            locked = "🔒 yes" if r["source_kind"] == "audit" else "no"
            lines.append(f"| {r['id']} | {t} | {src} | {set_on} | {locked} |")
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

    # --- Semantic-Threshold Recommendations ---
    semantic_rows = []
    for r in results:
        sem = r.get("semantic") or {}
        if not sem.get("ran"):
            continue
        t_par = sem.get("T_pareto")
        t_rec = sem.get("T_recommended")
        if t_par is None or t_rec is None:
            continue
        delta = sem.get("delta_vs_pareto")
        if delta is None:
            delta = abs(float(t_rec) - float(t_par))
        semantic_rows.append({
            "id": r["id"],
            "t_par": float(t_par),
            "t_rec": float(t_rec),
            "delta": float(delta),
            "agent_ba": sem.get("recommended_agent_ba"),
            "confidence": sem.get("confidence") or "---",
            "trigger": sem.get("trigger_reason") or "---",
        })

    if semantic_rows:
        semantic_rows.sort(key=lambda x: -abs(x["delta"]))
        lines.append("## Semantic-Threshold Recommendations")
        lines.append("")
        lines.append(
            "Float conflicts where the Pareto pick was flagged as ambiguous and the "
            "audit subagent re-derived a threshold from agent-graded condition-C samples. "
            "Apply via `/calibration-per-model-thresholds` (manual `--conflicts` invocation) "
            "after reviewing the rationale in each conflict's audit JSON."
        )
        lines.append("")
        lines.append(
            "| Conflict | T_pareto | T_rec | Δ | Agent BA | Confidence | Trigger |"
        )
        lines.append(
            "|----------|----------|-------|----|----------|------------|---------|"
        )
        for s in semantic_rows:
            agent_ba = _fmt(s["agent_ba"])
            lines.append(
                f"| {s['id']} | {s['t_par']:.3f} | {s['t_rec']:.3f} "
                f"| {s['delta']:+.3f} | {agent_ba} | {s['confidence']} "
                f"| {s['trigger']} |"
            )
        lines.append("")

    # --- Conflict Health ---
    # Show "Now" severity columns only when at least one conflict has
    # severity_at_yaml_t populated (i.e., its working_T differs from YAML T).
    show_now = any(r.get("severity_at_yaml_t") for r in results)
    lines.append("## Conflict Health")
    lines.append("")
    if show_now:
        lines.append(
            "| Conflict | Type | Feas | BA | d_norm | c_norm | Dist "
            "| Integrity | Now (YAML T) | Err% now | Post-fix | Err% fix | Action |"
        )
        lines.append(
            "|----------|------|------|----|--------|--------|------"
            "|-----------|--------------|----------|----------|----------|--------|"
        )
        for r in sorted(results, key=lambda x: -(x.get("error_pct_at_yaml_t") or x["error_pct"])):
            feas = "Y" if r.get("feasible", True) else "N"
            sev_now = r.get("severity_at_yaml_t") or r["severity"]
            err_now = r.get("error_pct_at_yaml_t")
            err_now_str = f"{err_now}%" if err_now is not None else f"{r['error_pct']}%"
            lines.append(
                f"| {r['id']} | {r['type']} | {feas} | {_fmt(r['ba'])} "
                f"| {_fmt(r['d_norm'], 6)} | {_fmt(r['c_norm'], 6)} "
                f"| {r['dist']} | {r['integrity']} | {sev_now} | {err_now_str} "
                f"| {r['severity']} | {r['error_pct']}% | {_short_action(r)} |"
            )
    else:
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
