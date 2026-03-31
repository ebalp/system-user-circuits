"""Audit tool for single-conflict analysis across one or more models.

Read-only CLI tool with three modes: summary, sample, query.
Provides structured analysis of JSONL results files for a single conflict.

Usage:
    # Summary (default)
    uv run python -m phase0_v2.calibration.audit_conflict \
      results_8b.jsonl results_70b.jsonl --conflict forbidden_words

    # Sample
    uv run python -m phase0_v2.calibration.audit_conflict \
      results_8b.jsonl --conflict forbidden_words \
      sample --label followed_system --direction b_to_a --n 10

    # Query
    uv run python -m phase0_v2.calibration.audit_conflict \
      results_8b.jsonl --conflict forbidden_words \
      query --condition C --response-contains "however" --count
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from ._shared import (
    compute_balanced_accuracy,
    compute_baseline_rates,
    find_anomalies,
    load_records_filtered,
)

# Meta-commentary detection — canonical location is phase0_v2.conflicts.preprocessing
from phase0_v2.conflicts.preprocessing import (
    META_COMMENTARY_RE,
    has_meta_commentary,
)
from .per_model_thresholds import compute_baseline_ranges
from ..config.thresholds import get_threshold
from ..conflicts.registry import get_conflict


# ---------------------------------------------------------------------------
# Model name helpers
# ---------------------------------------------------------------------------

_SHORT_NAME_PATTERNS = [
    (r"8B", "8B"),
    (r"70B", "70B"),
    (r"gemma.*27b", "Gemma-27B"),
    (r"gemma.*12b", "Gemma-12B"),
    (r"gemma.*9b", "Gemma-9B"),
]


def _short_model_name(model_id: str) -> str:
    """Derive a short display name from a full model ID."""
    for pattern, short in _SHORT_NAME_PATTERNS:
        if re.search(pattern, model_id, re.IGNORECASE):
            return short
    # Fallback: last component after /
    return model_id.split("/")[-1] if "/" in model_id else model_id


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------


def load_and_filter(
    file_paths: list[str], conflict_id: str
) -> list[tuple[str, str, list[dict]]]:
    """Load JSONL files, filter to conflict, return (display_name, path, records) per model.

    Uses streaming filter to avoid loading entire files into memory.
    Returns list of (display_name, file_path, filtered_records) tuples.
    """
    result = []
    for fp in file_paths:
        path = Path(fp)
        model_id, filtered = load_records_filtered(path, conflict_id)
        if model_id:
            display_name = _short_model_name(model_id)
        else:
            display_name = _short_model_name(path.stem.replace("_results", ""))
        result.append((display_name, fp, filtered))
    return result


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _condition_c_breakdown(records: list[dict]) -> dict:
    """Compute condition C label counts by direction.

    Returns dict with keys: 'a_to_b', 'b_to_a', 'total'.
    Each value is dict with keys: 'sys', 'usr', 'both', 'neither', 'n'.
    """
    def _empty():
        return {"sys": 0, "usr": 0, "both": 0, "neither": 0, "n": 0}

    by_dir = {"a_to_b": _empty(), "b_to_a": _empty(), "total": _empty()}

    for rec in records:
        if rec.get("condition") != "C":
            continue
        direction = rec.get("direction", "")
        label = rec.get("label", "")

        bucket = by_dir.get(direction)
        if bucket is None:
            continue

        bucket["n"] += 1
        by_dir["total"]["n"] += 1

        label_key = {
            "followed_system": "sys",
            "followed_user": "usr",
            "followed_both": "both",
            "followed_neither": "neither",
        }.get(label)

        if label_key:
            bucket[label_key] += 1
            by_dir["total"][label_key] += 1

    return by_dir


def _score_distribution(records: list[dict]) -> dict[str, int]:
    """Compute score distribution histogram for condition C system scores.

    Bins: [0,.1), [.1,.3), [.3,.5), [.5,.7), [.7,.9), [.9,1]
    """
    bins = {
        "[0,.1)": 0,
        "[.1,.3)": 0,
        "[.3,.5)": 0,
        "[.5,.7)": 0,
        "[.7,.9)": 0,
        "[.9,1]": 0,
    }
    for rec in records:
        if rec.get("condition") != "C":
            continue
        score = rec.get("verify_system_score")
        if score is None:
            continue
        if score < 0.1:
            bins["[0,.1)"] += 1
        elif score < 0.3:
            bins["[.1,.3)"] += 1
        elif score < 0.5:
            bins["[.3,.5)"] += 1
        elif score < 0.7:
            bins["[.5,.7)"] += 1
        elif score < 0.9:
            bins["[.7,.9)"] += 1
        else:
            bins["[.9,1]"] += 1
    return bins


def compute_summary(
    conflict_id: str,
    models_data: list[tuple[str, str, list[dict]]],
) -> dict:
    """Compute full summary data for a conflict across models.

    Returns dict with keys:
        conflict_id, is_float, threshold, constraint_a, constraint_b,
        counterbalance, models (list of per-model dicts).
    """
    threshold = get_threshold(conflict_id)
    is_float = threshold is not None

    conflict = get_conflict(conflict_id)
    if conflict:
        label_a, label_b = conflict.get_constraint_labels()
        counterbalance = "full" if conflict.supports_counterbalancing() else "none"
    else:
        label_a, label_b = "?", "?"
        counterbalance = "unknown"

    models = []
    for display_name, file_path, records in models_data:
        # Baselines
        baselines = compute_baseline_rates(records, conflict_id)
        ba_map = compute_balanced_accuracy(records, conflict_id)
        baseline = baselines[0] if baselines else {}
        ba = ba_map.get(conflict_id)

        # Condition C breakdown
        cond_c = _condition_c_breakdown(records)

        # Score distribution (float only)
        score_dist = _score_distribution(records) if is_float else None

        # Optimal threshold (float only)
        optimal = None
        if is_float:
            ranges = compute_baseline_ranges(records)
            r = ranges.get(conflict_id)
            if r:
                optimal = {"t_low": r["opt_low"], "t_high": r["opt_high"]}

        # Anomalies
        anomalies = find_anomalies(records, conflict_id)

        models.append({
            "name": display_name,
            "baseline": baseline,
            "ba": ba,
            "cond_c": cond_c,
            "score_dist": score_dist,
            "optimal": optimal,
            "n_anomalies": len(anomalies),
        })

    return {
        "conflict_id": conflict_id,
        "is_float": is_float,
        "threshold": threshold,
        "constraint_a": label_a,
        "constraint_b": label_b,
        "counterbalance": counterbalance,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(summary: dict) -> None:
    """Print formatted summary report."""
    cid = summary["conflict_id"]
    model_names = ", ".join(m["name"] for m in summary["models"])
    is_float = summary["is_float"]

    print("=" * 64)
    print(f"AUDIT: {cid}")
    print(f"Models: {model_names}")
    print("=" * 64)

    # Conflict definition
    print()
    print("CONFLICT DEFINITION")
    print(f"  Type:          {'float' if is_float else 'bool'}")
    print(f"  Constraint A:  {summary['constraint_a']}")
    print(f"  Constraint B:  {summary['constraint_b']}")
    print(f"  Counterbalance: {summary['counterbalance']}")
    if is_float:
        print(f"  Threshold:     {summary['threshold']:.3f}")
    else:
        print("  Threshold:     N/A (bool)")

    # Baselines
    print()
    print("BASELINES")
    header = f"  {'':10s} {'SBR(a)':>7} {'UCR(a)':>7} {'SBR(b)':>7} {'UCR(b)':>7} {'BA':>7}"
    print(header)
    for m in summary["models"]:
        b = m["baseline"]

        def _fmt(key):
            val = b.get(key)
            return f"{val:.3f}" if val is not None else "  N/A"

        ba_str = f"{m['ba']:.3f}" if m["ba"] is not None else "  N/A"
        print(
            f"  {m['name']:10s} {_fmt('sbr_a'):>7} {_fmt('ucr_a'):>7} "
            f"{_fmt('sbr_b'):>7} {_fmt('ucr_b'):>7} {ba_str:>7}"
        )

    # Threshold analysis (float only)
    if is_float:
        print()
        print("THRESHOLD ANALYSIS")
        print(f"  Current T:     {summary['threshold']:.3f}")
        opt_parts = []
        in_range_parts = []
        for m in summary["models"]:
            opt = m.get("optimal")
            if opt and opt["t_low"] is not None:
                t_lo = opt["t_low"]
                t_hi = opt["t_high"]
                opt_parts.append(f"{m['name']}=[{t_lo:.3f}, {t_hi:.3f}]")
                t = summary["threshold"]
                in_range = "YES" if t_lo <= t <= t_hi else "NO"
                in_range_parts.append(f"{m['name']}={in_range}")
            else:
                opt_parts.append(f"{m['name']}=N/A")
                in_range_parts.append(f"{m['name']}=N/A")
        print(f"  Optimal range: {'  '.join(opt_parts)}")
        print(f"  In range:      {'  '.join(in_range_parts)}")

    # Condition C
    print()
    print("CONDITION C (by direction)")
    for m in summary["models"]:
        cc = m["cond_c"]
        print(f"  {m['name']}:")
        for d in ("a_to_b", "b_to_a", "total"):
            b = cc[d]
            print(
                f"    {d:8s}  sys={b['sys']:<5d} usr={b['usr']:<5d} "
                f"both={b['both']:<3d} neither={b['neither']:<4d} (n={b['n']})"
            )

    # Score distribution (float only)
    if is_float:
        print()
        print("SCORE DISTRIBUTION (float, condition C, system score)")
        for m in summary["models"]:
            sd = m.get("score_dist")
            if sd:
                parts = "  ".join(f"{k}={v}" for k, v in sd.items())
                print(f"  {m['name']:10s} {parts}")

    # Anomalies
    print()
    anom_parts = "  ".join(f"{m['name']}={m['n_anomalies']}" for m in summary["models"])
    print(f"ANOMALIES: {anom_parts}")


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def sample_records(
    records: list[dict],
    conflict_id: str,
    *,
    label: str | None = None,
    direction: str | None = None,
    near_threshold: bool = False,
    anomalies: bool = False,
    n: int = 10,
) -> list[dict]:
    """Sample records matching criteria.

    Args:
        records: All records for a single model (already filtered to conflict).
        conflict_id: The conflict ID.
        label: Filter by label (followed_system, followed_user, etc.).
        direction: Filter by direction (a_to_b, b_to_a).
        near_threshold: Sort by distance to threshold (float conflicts only).
        anomalies: Return anomaly records from baselines.
        n: Max number of records to return.

    Returns list of record dicts.
    """
    if anomalies:
        result = find_anomalies(records, conflict_id)
        if len(result) > n:
            result = random.sample(result, n)
        return result

    threshold = get_threshold(conflict_id)
    is_float = threshold is not None

    if near_threshold:
        if not is_float:
            print("WARNING: --near-threshold is a no-op for bool conflicts.", file=sys.stderr)
            return []
        # Filter to condition C
        pool = [r for r in records if r.get("condition") == "C"]
        if direction:
            pool = [r for r in pool if r.get("direction") == direction]
        # Sort by distance to threshold
        pool.sort(key=lambda r: abs(r.get("verify_system_score", 0.0) - threshold))
        return pool[:n]

    # Standard filtering
    pool = [r for r in records if r.get("condition") == "C"]
    if label:
        pool = [r for r in pool if r.get("label") == label]
    if direction:
        pool = [r for r in pool if r.get("direction") == direction]

    if len(pool) > n:
        pool = random.sample(pool, n)
    return pool


def print_samples(samples: list[dict], conflict_id: str) -> None:
    """Print formatted sample output."""
    if not samples:
        print("No matching records found.")
        return

    for i, rec in enumerate(samples, 1):
        direction = rec.get("direction", "?")
        label = rec.get("label", rec.get("anomaly_reason", "?"))
        sys_score = rec.get("verify_system_score", rec.get("verify_system_result", "?"))
        usr_score = rec.get("verify_user_score", rec.get("verify_user_result", "?"))

        # Format scores
        if isinstance(sys_score, float) and sys_score not in (0.0, 1.0):
            sys_str = f"{sys_score:.3f}"
        else:
            sys_str = str(sys_score)
        if isinstance(usr_score, float) and usr_score not in (0.0, 1.0):
            usr_str = f"{usr_score:.3f}"
        else:
            usr_str = str(usr_score)

        print(f"\n[{i}/{len(samples)}] dir={direction}  label={label}")
        print(f"  sys_score={sys_str}  usr_score={usr_str}")

        task = rec.get("task_id", rec.get("task", "?"))
        sys_style = rec.get("system_style", "?")
        usr_style = rec.get("user_style", "?")
        print(f"  task: {task}")
        print(f"  system_style: {sys_style}  user_style: {usr_style}")

        # Constraints (truncated)
        sys_constraint = rec.get("system_prompt", "")
        usr_constraint = rec.get("user_prompt", "")
        if len(sys_constraint) > 80:
            sys_constraint = sys_constraint[:77] + "..."
        if len(usr_constraint) > 80:
            usr_constraint = usr_constraint[:77] + "..."
        print(f"  system_constraint: \"{sys_constraint}\"")
        print(f"  user_constraint: \"{usr_constraint}\"")

        response = rec.get("response", "")
        if len(response) > 500:
            response = response[:500] + "..."
        print(f"  response (first 500 chars):")
        print(f"    \"{response}\"")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_records(
    records: list[dict],
    conflict_id: str,
    *,
    condition: str | None = None,
    direction: str | None = None,
    response_contains: str | None = None,
    meta_commentary: bool = False,
    score_range: tuple[float, float] | None = None,
    label: str | None = None,
    count: bool = False,
    n: int = 10,
) -> list[dict] | int | dict:
    """Filter records by criteria and return matches or count.

    Args:
        records: All records for a single model (already filtered to conflict).
        conflict_id: The conflict ID.
        condition: Filter by condition (A, B, C, D).
        direction: Filter by direction (a_to_b, b_to_a).
        response_contains: Case-insensitive substring match on response.
        meta_commentary: If True, filter to responses matching META_COMMENTARY_RE.
            When combined with --count, returns per-direction breakdown.
        score_range: Filter by verify_system_score in [low, high].
        label: Filter by label.
        count: If True, return integer count instead of records.
        n: Max number of records to return (ignored if count=True).

    Returns list of record dicts, int if count=True,
    or dict with per-direction counts if meta_commentary + count.
    """
    pool = list(records)

    if condition:
        pool = [r for r in pool if r.get("condition") == condition]
    if direction:
        pool = [r for r in pool if r.get("direction") == direction]
    if label:
        pool = [r for r in pool if r.get("label") == label]
    if response_contains:
        pattern_lower = response_contains.lower()
        pool = [r for r in pool if pattern_lower in r.get("response", "").lower()]
    if meta_commentary:
        pool = [r for r in pool if has_meta_commentary(r.get("response", ""), conflict_id)]
    if score_range:
        lo, hi = score_range
        pool = [
            r for r in pool
            if r.get("verify_system_score") is not None
            and lo <= r["verify_system_score"] <= hi
        ]

    if count:
        if meta_commentary and not direction:
            # Per-direction breakdown for meta-commentary counts
            by_dir: dict[str, int] = defaultdict(int)
            for r in pool:
                d = r.get("direction", "unknown")
                by_dir[d] += 1
            return {"total": len(pool), **dict(sorted(by_dir.items()))}
        return len(pool)

    if len(pool) > n:
        pool = random.sample(pool, n)
    return pool


def print_query_results(results: list[dict] | int | dict, conflict_id: str) -> None:
    """Print query results (count, per-direction dict, or samples)."""
    if isinstance(results, dict):
        total = results.pop("total", 0)
        print(f"Meta-commentary responses: {total}")
        for direction, cnt in sorted(results.items()):
            print(f"  {direction}: {cnt}")
    elif isinstance(results, int):
        print(f"Count: {results}")
    else:
        print_samples(results, conflict_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv into (file_paths, remaining_args).

    Consumes leading positional args (not starting with '-' and not 'sample'/'query')
    as file paths, then returns the rest for argparse.
    """
    subcommands = {"sample", "query"}
    files = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            break
        if arg in subcommands:
            break
        files.append(arg)
        i += 1
    return files, argv[i:]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Pre-split: argparse can't handle nargs="+" positional before subparsers.
    results_files, rest = _split_argv(argv)

    parser = argparse.ArgumentParser(
        description="Audit tool for single-conflict analysis across one or more models.",
        prog="python -m phase0_v2.calibration.audit_conflict",
    )
    parser.add_argument(
        "--conflict",
        required=True,
        help="Conflict ID to audit",
    )

    subparsers = parser.add_subparsers(dest="mode")

    # Sample subcommand
    sample_parser = subparsers.add_parser("sample", help="Sample responses by category")
    sample_parser.add_argument("--label", help="Filter by label (followed_system, followed_user, etc.)")
    sample_parser.add_argument("--direction", help="Filter by direction (a_to_b, b_to_a)")
    sample_parser.add_argument("--near-threshold", action="store_true", help="Sort by distance to threshold (float only)")
    sample_parser.add_argument("--anomalies", action="store_true", help="Sample anomaly records from baselines")
    sample_parser.add_argument("--n", type=int, default=10, help="Number of samples (default: 10)")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Filter and count records")
    query_parser.add_argument("--condition", help="Filter by condition (A, B, C, D)")
    query_parser.add_argument("--direction", help="Filter by direction (a_to_b, b_to_a)")
    query_parser.add_argument("--label", help="Filter by label")
    query_parser.add_argument("--response-contains", help="Case-insensitive substring match on response")
    query_parser.add_argument("--meta-commentary", action="store_true",
                              help="Filter to responses containing meta-commentary patterns. "
                                   "With --count, shows per-direction breakdown.")
    query_parser.add_argument("--score-range", nargs=2, type=float, metavar=("LOW", "HIGH"),
                              help="Filter by verify_system_score in [LOW, HIGH]")
    query_parser.add_argument("--count", action="store_true", help="Return count instead of records")
    query_parser.add_argument("--n", type=int, default=10, help="Number of samples (default: 10)")

    args = parser.parse_args(rest)
    conflict_id = args.conflict
    mode = args.mode  # None means summary

    if not results_files:
        parser.error("at least one results file is required")

    # Load and filter
    models_data = load_and_filter(results_files, conflict_id)

    # Check that we have data
    total_records = sum(len(recs) for _, _, recs in models_data)
    if total_records == 0:
        print(f"No records found for conflict '{conflict_id}' in the provided files.", file=sys.stderr)
        sys.exit(1)

    if mode is None:
        # Summary mode (default)
        summary = compute_summary(conflict_id, models_data)
        print_summary(summary)

    elif mode == "sample":
        if len(models_data) > 1:
            print("WARNING: sample mode uses only the first results file.", file=sys.stderr)
        _, _, records = models_data[0]
        samples = sample_records(
            records,
            conflict_id,
            label=args.label,
            direction=args.direction,
            near_threshold=getattr(args, "near_threshold", False),
            anomalies=args.anomalies,
            n=args.n,
        )
        print_samples(samples, conflict_id)

    elif mode == "query":
        if len(models_data) > 1:
            print("WARNING: query mode uses only the first results file.", file=sys.stderr)
        _, _, records = models_data[0]
        score_range = tuple(args.score_range) if args.score_range else None
        results = query_records(
            records,
            conflict_id,
            condition=args.condition,
            direction=args.direction,
            response_contains=args.response_contains,
            meta_commentary=getattr(args, "meta_commentary", False),
            score_range=score_range,
            label=args.label,
            count=args.count,
            n=args.n,
        )
        print_query_results(results, conflict_id)


if __name__ == "__main__":
    main()
