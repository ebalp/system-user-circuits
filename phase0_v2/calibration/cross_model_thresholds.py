"""Cross-model threshold intersection analysis.

Computes optimal threshold intersections across multiple models and generates
a report with per-model ranges, intersection results, and proposed thresholds.

Usage:
    uv run python -m phase0_v2.calibration.cross_model_thresholds MODEL_CSV [MODEL_CSV ...]

    Each MODEL_CSV is a path to a calibration_report.csv from analyze.py.

Examples:
    uv run python -m phase0_v2.calibration.cross_model_thresholds \
      phase0_v2/calibration/output/meta-llama_Llama-3.1-8B-Instruct/calibration_report.csv \
      phase0_v2/calibration/output/meta-llama_Llama-3.3-70B-Instruct/calibration_report.csv

    # With current thresholds for comparison:
    uv run python -m phase0_v2.calibration.cross_model_thresholds \
      phase0_v2/calibration/output/*/calibration_report.csv \
      --thresholds phase0_v2/config/thresholds.yaml

    # Write updated thresholds (requires --confirm):
    uv run python -m phase0_v2.calibration.cross_model_thresholds \
      phase0_v2/calibration/output/*/calibration_report.csv \
      --thresholds phase0_v2/config/thresholds.yaml \
      --update --confirm
"""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..conflicts.registry import get_all_conflicts


# ---------------------------------------------------------------------------
# Model capability tiers
# ---------------------------------------------------------------------------

_TIER_KEYWORDS: dict[str, list[str]] = {
    "strong": ["70b", "27b"],
    "mid": ["8b", "7b", "20b"],
    "weak": ["1b"],
}


def _model_tier(model_id: str) -> str:
    """Classify a model into a capability tier based on its ID."""
    lower = model_id.lower()
    for tier, keywords in _TIER_KEYWORDS.items():
        for kw in keywords:
            # Match e.g. "70B" preceded by a dash or non-digit, to avoid "170B" matching "70B"
            idx = lower.find(kw)
            if idx >= 0:
                # Check the char before isn't a digit (avoid "170b" matching "70b")
                if idx == 0 or not lower[idx - 1].isdigit():
                    return tier
    return "mid"  # default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelRange:
    """Optimal threshold range(s) for one conflict from one model."""
    model_id: str
    ranges: list[tuple[float, float]]  # list of (low, high) intervals
    ba: float
    tier: str


@dataclass
class IntersectionResult:
    """Result of cross-model intersection for one conflict."""
    conflict_id: str
    model_ranges: dict[str, ModelRange]  # model_id -> ModelRange
    # Full intersection (all models)
    full_intersection: tuple[float, float] | None = None
    # Best subset intersection (if full fails)
    subset_intersection: tuple[float, float] | None = None
    subset_models: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)
    # Proposed threshold
    proposed_t: float | None = None
    current_t: float | None = None

    @property
    def intersection(self) -> tuple[float, float] | None:
        return self.full_intersection or self.subset_intersection

    @property
    def status(self) -> str:
        if self.full_intersection is not None:
            if self.current_t is not None:
                low, high = self.full_intersection
                if low <= self.current_t <= high and self.current_t == self.proposed_t:
                    return "OK"
                elif low <= self.current_t <= high:
                    return "UPDATE"
                else:
                    return "OUTSIDE"
            return "UPDATE"
        elif self.subset_intersection is not None:
            n = len(self.subset_models)
            m = n + len(self.excluded_models)
            return f"SUBSET ({n}/{m})"
        else:
            return "NO INTERSECTION"


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------

def _intersect_two(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    """Intersect two intervals. Returns None if empty."""
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    if low <= high:
        return (low, high)
    return None


def _intersect_range_lists(
    range_lists: list[list[tuple[float, float]]],
) -> tuple[float, float] | None:
    """Intersect multiple lists of intervals using Cartesian product.

    Each model may have multiple disjoint optimal regions. We try all
    combinations (one interval per model) and keep the widest valid
    intersection.
    """
    if not range_lists:
        return None

    best: tuple[float, float] | None = None
    best_width = -1.0

    for combo in itertools.product(*range_lists):
        # Intersect all intervals in this combination
        current = combo[0]
        valid = True
        for iv in combo[1:]:
            result = _intersect_two(current, iv)
            if result is None:
                valid = False
                break
            current = result
        if valid:
            width = current[1] - current[0]
            if width > best_width:
                best = current
                best_width = width

    return best


def _midpoint(interval: tuple[float, float]) -> float:
    """Compute midpoint rounded to 3 decimal places."""
    return round((interval[0] + interval[1]) / 2, 3)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_model_ranges(csv_path: Path) -> dict[str, ModelRange]:
    """Parse a calibration_report.csv and extract optimal ranges per float conflict.

    Returns {conflict_id: ModelRange}.
    """
    # Derive model_id from the parent directory name
    model_id = csv_path.parent.name

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by conflict_id, take first row with non-empty optimal_threshold_low
    seen: dict[str, ModelRange] = {}
    for row in rows:
        cid = row["conflict_id"]
        if cid in seen:
            continue
        # Skip bool conflicts (no float calibration data)
        if not row.get("trying_mean"):
            continue
        opt_low = row.get("optimal_threshold_low", "")
        opt_high = row.get("optimal_threshold_high", "")
        if not opt_low or not opt_high:
            continue

        low = float(opt_low)
        high = float(opt_high)

        # Check for disjoint regions
        regions_str = row.get("optimal_regions", "")
        if regions_str:
            regions = [(r[0], r[1]) for r in json.loads(regions_str)]
        else:
            regions = [(low, high)]

        ba = float(row["balanced_accuracy"]) if row.get("balanced_accuracy") else 0.0

        seen[cid] = ModelRange(
            model_id=model_id,
            ranges=regions,
            ba=ba,
            tier=_model_tier(model_id),
        )

    return seen


# ---------------------------------------------------------------------------
# Cross-model intersection
# ---------------------------------------------------------------------------

def compute_intersections(
    all_model_ranges: dict[str, dict[str, ModelRange]],
    current_thresholds: dict[str, float],
) -> list[IntersectionResult]:
    """Compute cross-model intersections for all float conflicts.

    Args:
        all_model_ranges: {model_id: {conflict_id: ModelRange}}
        current_thresholds: {conflict_id: current_threshold}

    Returns:
        List of IntersectionResult, sorted by conflict_id.
    """
    # Collect all float conflict IDs across models
    all_conflicts: set[str] = set()
    for model_ranges in all_model_ranges.values():
        all_conflicts.update(model_ranges.keys())

    results = []
    model_ids = sorted(all_model_ranges.keys())

    for cid in sorted(all_conflicts):
        # Gather ranges for models that have this conflict
        present_models = []
        for mid in model_ids:
            if cid in all_model_ranges[mid]:
                present_models.append(mid)

        if len(present_models) < 2:
            # Only one model has this conflict — use its range directly
            if present_models:
                mid = present_models[0]
                mr = all_model_ranges[mid][cid]
                best = _intersect_range_lists([mr.ranges])
                result = IntersectionResult(
                    conflict_id=cid,
                    model_ranges={mid: mr},
                    full_intersection=best,
                    proposed_t=_midpoint(best) if best else None,
                    current_t=current_thresholds.get(cid),
                )
            else:
                result = IntersectionResult(
                    conflict_id=cid,
                    model_ranges={},
                    current_t=current_thresholds.get(cid),
                )
            results.append(result)
            continue

        model_ranges_map = {mid: all_model_ranges[mid][cid] for mid in present_models}

        # Try full intersection
        range_lists = [model_ranges_map[mid].ranges for mid in present_models]
        full = _intersect_range_lists(range_lists)

        result = IntersectionResult(
            conflict_id=cid,
            model_ranges=model_ranges_map,
            current_t=current_thresholds.get(cid),
        )

        if full is not None:
            result.full_intersection = full
            result.proposed_t = _midpoint(full)
        else:
            # Try subsets: from (N-1) down to 2
            best_subset: tuple[float, float] | None = None
            best_subset_models: list[str] = []
            best_subset_size = 0
            best_subset_width = -1.0

            for size in range(len(present_models) - 1, 1, -1):
                if best_subset_size > size:
                    break  # already found a larger agreeing subset
                for subset in itertools.combinations(present_models, size):
                    sub_ranges = [model_ranges_map[mid].ranges for mid in subset]
                    inter = _intersect_range_lists(sub_ranges)
                    if inter is not None:
                        width = inter[1] - inter[0]
                        if (size > best_subset_size) or (
                            size == best_subset_size and width > best_subset_width
                        ):
                            best_subset = inter
                            best_subset_models = list(subset)
                            best_subset_size = size
                            best_subset_width = width
                # Found at least one subset of this size? Stop going smaller
                if best_subset_size == size:
                    break

            if best_subset is not None:
                result.subset_intersection = best_subset
                result.subset_models = best_subset_models
                result.excluded_models = [
                    mid for mid in present_models if mid not in best_subset_models
                ]
                result.proposed_t = _midpoint(best_subset)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Stale entry detection
# ---------------------------------------------------------------------------

def find_stale_entries(current_thresholds: dict[str, float]) -> list[str]:
    """Find threshold entries for conflicts not in the registry."""
    registered = {c.conflict_id for c in get_all_conflicts()}
    return sorted(cid for cid in current_thresholds if cid not in registered)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_range(r: tuple[float, float]) -> str:
    if abs(r[1] - r[0]) < 0.0005:
        return f"{r[0]:.3f}"
    return f"[{r[0]:.3f}, {r[1]:.3f}]"


def _fmt_ranges(ranges: list[tuple[float, float]]) -> str:
    return ", ".join(_fmt_range(r) for r in ranges)


def _distance_description(t: float, ranges: list[tuple[float, float]]) -> str:
    """Describe how far t is from the nearest range edge."""
    for low, high in ranges:
        if low <= t <= high:
            return "in range"
    # Find closest edge
    min_dist = float("inf")
    direction = ""
    for low, high in ranges:
        if t < low and (low - t) < min_dist:
            min_dist = low - t
            direction = "below"
        if t > high and (t - high) < min_dist:
            min_dist = t - high
            direction = "above"
    return f"{min_dist:.3f} {direction}"


def generate_report(
    results: list[IntersectionResult],
    stale_entries: list[str],
    model_ids: list[str],
) -> str:
    """Generate a markdown report."""
    lines: list[str] = []
    lines.append("# Cross-Model Threshold Report\n")
    lines.append(f"**Models**: {', '.join(model_ids)}")
    lines.append(f"**Model tiers**: {', '.join(f'{m} ({_model_tier(m)})' for m in model_ids)}\n")

    # --- Main table ---
    lines.append("## Threshold intersection results\n")
    header = "| Conflict |"
    sep = "|----------|"
    for mid in model_ids:
        short = mid.split("_")[-1] if "_" in mid else mid
        header += f" {short} range |"
        sep += "--------------|"
    header += " Intersection | New T | Current T | Status |"
    sep += "-------------|-------|-----------|--------|"
    lines.append(header)
    lines.append(sep)

    counts = {"OK": 0, "UPDATE": 0, "OUTSIDE": 0, "SUBSET": 0, "NO INTERSECTION": 0}

    for r in results:
        row = f"| {r.conflict_id} |"
        for mid in model_ids:
            if mid in r.model_ranges:
                row += f" {_fmt_ranges(r.model_ranges[mid].ranges)} |"
            else:
                row += " -- |"
        inter = r.intersection
        row += f" {_fmt_range(inter) if inter else '--'} |"
        row += f" {r.proposed_t:.3f} |" if r.proposed_t is not None else " -- |"
        row += f" {r.current_t:.3f} |" if r.current_t is not None else " -- |"
        status = r.status
        row += f" {status} |"
        lines.append(row)

        # Count statuses
        if status.startswith("SUBSET"):
            counts["SUBSET"] += 1
        elif status in counts:
            counts[status] += 1

    # --- Inconsistency analysis ---
    subset_or_none = [r for r in results if r.status.startswith("SUBSET") or r.status == "NO INTERSECTION"]
    if subset_or_none:
        lines.append("\n## Inconsistency analysis\n")
        for r in subset_or_none:
            lines.append(f"### {r.conflict_id}\n")
            if r.subset_models:
                lines.append(f"**Models agreeing:** {', '.join(r.subset_models)}")
                for mid in r.excluded_models:
                    mr = r.model_ranges[mid]
                    tier = mr.tier
                    lines.append(f"**Excluded:** {mid} ({tier} tier)")
                    lines.append(f"  - Optimal range: {_fmt_ranges(mr.ranges)}, BA: {mr.ba:.3f}")
                    if r.proposed_t is not None:
                        desc = _distance_description(r.proposed_t, mr.ranges)
                        if desc == "in range":
                            lines.append(f"  - Proposed T={r.proposed_t:.3f} is **in range** — BA unaffected")
                        else:
                            lines.append(f"  - Proposed T={r.proposed_t:.3f} is **outside** ({desc}) — BA may degrade")

                # Assessment
                excluded_tiers = [r.model_ranges[mid].tier for mid in r.excluded_models]
                if all(t == "weak" for t in excluded_tiers):
                    lines.append(f"\n**Assessment:** Expected — weak-tier model(s) are sole outlier(s). "
                                 f"Recommend using {len(r.subset_models)}-model intersection.")
                elif any(t == "strong" for t in excluded_tiers):
                    lines.append(f"\n**Assessment:** Investigate — strong-tier model excluded. "
                                 f"May indicate verifier issue or genuinely different behavior.")
                else:
                    lines.append(f"\n**Assessment:** Trade-off — mid-tier model excluded. "
                                 f"Check BA impact before deciding.")
            else:
                lines.append("No valid intersection found even pairwise. Investigate verifier behavior across models.")
                for mid, mr in r.model_ranges.items():
                    lines.append(f"  - {mid} ({mr.tier}): {_fmt_ranges(mr.ranges)}, BA: {mr.ba:.3f}")
            lines.append("")

    # --- Stale entries ---
    lines.append("## Stale entries\n")
    if stale_entries:
        for entry in stale_entries:
            lines.append(f"- `{entry}`")
    else:
        lines.append("None.")

    # --- Summary ---
    lines.append("\n## Summary\n")
    lines.append(f"- **{counts['OK']}** thresholds OK (no change needed)")
    lines.append(f"- **{counts['UPDATE']}** thresholds to UPDATE (in range but not at midpoint)")
    lines.append(f"- **{counts['OUTSIDE']}** thresholds OUTSIDE range")
    if counts["SUBSET"] > 0:
        lines.append(f"- **{counts['SUBSET']}** thresholds using SUBSET intersection")
    if counts["NO INTERSECTION"] > 0:
        lines.append(f"- **{counts['NO INTERSECTION']}** thresholds with NO INTERSECTION")
    lines.append(f"- **{len(stale_entries)}** stale entries to remove")

    # --- Proposed changes ---
    changes = [r for r in results if r.proposed_t is not None and r.current_t != r.proposed_t]
    if changes:
        lines.append("\n## Proposed changes\n")
        lines.append("| Conflict | Current T | New T | Delta |")
        lines.append("|----------|-----------|-------|-------|")
        for r in changes:
            cur = r.current_t if r.current_t is not None else 0.0
            delta = r.proposed_t - cur
            lines.append(f"| {r.conflict_id} | {cur:.3f} | {r.proposed_t:.3f} | {delta:+.3f} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _short_model(model_id: str) -> str:
    """Extract a short label like '8B', '70B', 'gemma-27b'."""
    lower = model_id.lower()
    for label in ["8b", "70b", "27b", "7b", "20b", "1b"]:
        idx = lower.find(label)
        if idx >= 0 and (idx == 0 or not lower[idx - 1].isdigit()):
            # Include prefix hint for non-llama models
            if "gemma" in lower:
                return f"gemma-{label.upper()}"
            if "qwen" in lower:
                return f"qwen-{label.upper()}"
            if "gpt" in lower or "oss" in lower:
                return f"gpt-{label.upper()}"
            return label.upper()
    return model_id.split("_")[-1][:12]


def _fmt_range_compact(r: tuple[float, float]) -> str:
    if abs(r[1] - r[0]) < 0.0005:
        return f"{r[0]:.3f}"
    return f"[{r[0]:.3f},{r[1]:.3f}]"


def _fmt_ranges_compact(ranges: list[tuple[float, float]]) -> str:
    return " + ".join(_fmt_range_compact(r) for r in ranges)


def print_console_table(
    results: list[IntersectionResult],
    model_ids: list[str],
    stale_entries: list[str],
) -> None:
    """Print a formatted console table of all float threshold intersections."""
    # Build short model labels
    short_labels = [_short_model(mid) for mid in model_ids]

    # Compute column widths
    cid_w = max(len("conflict_id"), max(len(r.conflict_id) for r in results))
    range_ws = []
    for i, mid in enumerate(model_ids):
        col_w = len(short_labels[i])
        for r in results:
            if mid in r.model_ranges:
                val = _fmt_ranges_compact(r.model_ranges[mid].ranges)
            else:
                val = "--"
            col_w = max(col_w, len(val))
        range_ws.append(col_w)
    inter_w = max(len("intersection"), 15)
    for r in results:
        inter = r.full_intersection
        val = _fmt_range_compact(inter) if inter else "None"
        inter_w = max(inter_w, len(val))

    # Print header
    header = f"{'conflict_id':<{cid_w}}"
    for i, label in enumerate(short_labels):
        header += f"  {label:>{range_ws[i]}}"
    header += f"  {'intersection':>{inter_w}}  {'cur_T':>6}  {'new_T':>6}  status"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("CROSS-MODEL THRESHOLD INTERSECTION")
    print("=" * len(header))
    print(header)
    print(sep)

    counts = {"OK": 0, "UPDATE": 0, "OUTSIDE": 0, "SUBSET": 0, "NO INTERSECTION": 0}

    for r in results:
        status = r.status
        line = f"{r.conflict_id:<{cid_w}}"
        for i, mid in enumerate(model_ids):
            if mid in r.model_ranges:
                val = _fmt_ranges_compact(r.model_ranges[mid].ranges)
            else:
                val = "--"
            line += f"  {val:>{range_ws[i]}}"

        inter = r.full_intersection
        inter_str = _fmt_range_compact(inter) if inter else "None"
        line += f"  {inter_str:>{inter_w}}"

        cur_str = f"{r.current_t:.3f}" if r.current_t is not None else "--"
        line += f"  {cur_str:>6}"

        if r.proposed_t is not None and r.current_t != r.proposed_t:
            line += f"  {r.proposed_t:.3f}"
        else:
            line += f"  {'':>6}"

        if status != "OK":
            line += f"  {status}"

        print(line)

        if status.startswith("SUBSET"):
            counts["SUBSET"] += 1
        elif status in counts:
            counts[status] += 1

    print(sep)
    parts = []
    for key in ["OK", "UPDATE", "OUTSIDE", "SUBSET", "NO INTERSECTION"]:
        if counts[key] > 0:
            parts.append(f"{counts[key]} {key}")
    print(f"  {', '.join(parts)}")
    if stale_entries:
        print(f"  Stale entries to remove: {', '.join(stale_entries)}")
    print()


# ---------------------------------------------------------------------------
# YAML update
# ---------------------------------------------------------------------------

def update_thresholds_yaml(
    yaml_path: Path,
    results: list[IntersectionResult],
    stale_entries: list[str],
    model_ids: list[str],
    *,
    skip_subset: bool = False,
) -> dict[str, float]:
    """Write updated thresholds.yaml. Returns the new threshold dict.

    Args:
        skip_subset: If True, don't update thresholds that only have subset intersections.
    """
    new_thresholds: dict[str, float] = {}

    for r in results:
        if r.proposed_t is None:
            # NO INTERSECTION — keep current if exists
            if r.current_t is not None and r.conflict_id not in stale_entries:
                new_thresholds[r.conflict_id] = r.current_t
            continue
        if skip_subset and r.full_intersection is None:
            # Subset only — keep current
            if r.current_t is not None and r.conflict_id not in stale_entries:
                new_thresholds[r.conflict_id] = r.current_t
            continue
        if r.conflict_id in stale_entries:
            continue
        new_thresholds[r.conflict_id] = r.proposed_t

    # Sort alphabetically
    new_thresholds = dict(sorted(new_thresholds.items()))

    # Write YAML with header
    from datetime import date
    model_list = ", ".join(model_ids)
    header = (
        f"# Float conflict thresholds — single source of truth.\n"
        f"# Cross-model intersection midpoints for: {model_list}\n"
        f"# Last updated: {date.today().isoformat()}\n"
        f"# Boolean conflicts don't have thresholds and are not listed here.\n\n"
    )

    with open(yaml_path, "w") as f:
        f.write(header)
        yaml.dump(new_thresholds, f, default_flow_style=False, sort_keys=True)

    return new_thresholds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model threshold intersection analysis."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Paths to calibration_report.csv files (one per model).",
    )
    parser.add_argument(
        "--thresholds",
        default="phase0_v2/config/thresholds.yaml",
        help="Path to current thresholds.yaml (default: phase0_v2/config/thresholds.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Path for the markdown report (default: stdout).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update thresholds.yaml with intersection midpoints.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --update to actually write changes.",
    )
    parser.add_argument(
        "--include-subset",
        action="store_true",
        help="Also apply subset intersection thresholds (default: only full intersections).",
    )
    args = parser.parse_args()

    # Load current thresholds
    thresholds_path = Path(args.thresholds)
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            current_thresholds = yaml.safe_load(f) or {}
    else:
        current_thresholds = {}

    # Parse CSVs
    all_model_ranges: dict[str, dict[str, ModelRange]] = {}
    model_ids: list[str] = []
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"ERROR: {csv_path} does not exist", file=sys.stderr)
            sys.exit(1)
        model_id = csv_path.parent.name
        model_ids.append(model_id)
        all_model_ranges[model_id] = parse_model_ranges(csv_path)

    # Compute intersections
    results = compute_intersections(all_model_ranges, current_thresholds)
    stale = find_stale_entries(current_thresholds)

    # Print console table
    print_console_table(results, model_ids, stale)

    # Generate report
    report = generate_report(results, stale, model_ids)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Report written to {out_path}")
    else:
        print(report)

    # Update thresholds
    if args.update:
        if not args.confirm:
            print("\n⚠ --update requires --confirm to write changes.", file=sys.stderr)
            print("Review the report above, then re-run with --update --confirm.", file=sys.stderr)
            sys.exit(1)
        new_t = update_thresholds_yaml(
            thresholds_path,
            results,
            stale,
            model_ids,
            skip_subset=not args.include_subset,
        )
        print(f"\nUpdated {thresholds_path} with {len(new_t)} thresholds.")
        if stale:
            print(f"Removed {len(stale)} stale entries: {', '.join(stale)}")


if __name__ == "__main__":
    main()
