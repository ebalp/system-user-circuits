"""Cross-tabulate response structure × constraint type × verifier label.

Two usage modes:

1. CLI — full model analysis or per-conflict:
    uv run python -m phase0_v2.calibration.response_type_analysis \
      phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl

    uv run python -m phase0_v2.calibration.response_type_analysis \
      phase0_v2/data/results/*_results.jsonl --conflict keyword_avoidance

2. Importable by audit subagents:
    from phase0_v2.calibration._shared import load_records_filtered
    from phase0_v2.calibration.response_type_analysis import (
        compute_structure_stats,
        compute_conflict_structure,
        print_structure_stats,
        print_conflict_structure,
        load_constraint_type_map,
    )

    type_map = load_constraint_type_map()
    model_id, records = load_records_filtered(path, "keyword_avoidance")
    stats = compute_conflict_structure(records, type_map)
    print_conflict_structure(stats, model=model_id, conflict="keyword_avoidance")
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ._shared import load_records, load_records_filtered
from phase0_v2.conflicts.preprocessing import classify_response as _classify_tags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LABELS = ("followed_system", "followed_user", "followed_both", "followed_neither")
_CONSTRAINT_TYPES = ("presence", "avoidance", "ambiguous")

_SHORT_NAME_PATTERNS = [
    (r"8B", "8B"),
    (r"1B", "1B"),
    (r"70B", "70B"),
    (r"gemma.*27b", "Gemma-27B"),
    (r"gpt-oss-20b", "gpt-oss-20B"),
    (r"Qwen2\.5-7B", "Qwen-7B"),
]

_CONFLICTS_YAML = Path(__file__).resolve().parent.parent / "config" / "conflicts.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_model_name(model_id: str) -> str:
    for pattern, short in _SHORT_NAME_PATTERNS:
        if re.search(pattern, model_id, re.IGNORECASE):
            return short
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.1f}%" if total else "0.0%"


def _pattern_str(pattern: tuple | list) -> str:
    return " -> ".join(pattern) if pattern else "(clean)"


def _get_structure(rec: dict) -> tuple:
    """Get structure pattern as a tuple (hashable key)."""
    tags = rec.get("refusal_tags")
    if tags is None:
        return ("content",)
    return tuple(tags["structure"])


def classify_response(rec: dict) -> str:
    """Classify a record into one of four response types from precomputed tags."""
    return _classify_tags(rec.get("refusal_tags"))


def get_system_constraint_type(rec: dict, type_map: dict) -> str:
    """Return the system-side constraint type (presence/avoidance/ambiguous).

    In a_to_b: system = constraint_a.
    In b_to_a: system = constraint_b.
    """
    cid = rec.get("conflict_id")
    direction = rec.get("direction")
    types = type_map.get(cid, {})
    if direction == "a_to_b":
        return types.get("a", "unknown")
    elif direction == "b_to_a":
        return types.get("b", "unknown")
    return "unknown"


def load_constraint_type_map(yaml_path: Path | None = None) -> dict[str, dict[str, str]]:
    """Build {conflict_id: {"a": type, "b": type}} from conflicts.yaml."""
    path = yaml_path or _CONFLICTS_YAML
    with open(path) as f:
        data = yaml.safe_load(f)

    result = {}
    for cid, info in data.items():
        if cid.startswith("_"):
            continue
        result[cid] = {
            "a": info.get("constraint_a_type", "unknown"),
            "b": info.get("constraint_b_type", "unknown"),
        }
    return result


# ---------------------------------------------------------------------------
# Compute functions — return data, importable by subagents
# ---------------------------------------------------------------------------


def compute_structure_stats(
    records: list[dict],
    type_map: dict | None = None,
    *,
    condition: str = "C",
    top_n: int = 20,
) -> dict:
    """Compute structure pattern statistics.

    Args:
        records: JSONL records (any condition).
        type_map: constraint type map (optional, adds constraint breakdown).
        condition: filter to this condition.
        top_n: return top N patterns.

    Returns:
        {
            "total": int,
            "patterns": [
                {
                    "structure": tuple,
                    "count": int,
                    "labels": Counter,
                    "constraint_types": Counter,  # if type_map provided
                },
                ...
            ]
        }
    """
    filtered = [r for r in records if r.get("condition") == condition]
    pattern_counts: Counter = Counter()
    pattern_labels: dict[tuple, Counter] = defaultdict(Counter)
    pattern_constraints: dict[tuple, Counter] = defaultdict(Counter)

    for rec in filtered:
        pat = _get_structure(rec)
        pattern_counts[pat] += 1
        pattern_labels[pat][rec.get("label", "?")] += 1
        if type_map:
            ct = get_system_constraint_type(rec, type_map)
            pattern_constraints[pat][ct] += 1

    patterns = []
    for pat, count in pattern_counts.most_common(top_n):
        entry = {
            "structure": pat,
            "count": count,
            "labels": dict(pattern_labels[pat]),
        }
        if type_map:
            entry["constraint_types"] = dict(pattern_constraints[pat])
        patterns.append(entry)

    return {"total": len(filtered), "patterns": patterns}


def compute_structure_by_constraint(
    records: list[dict],
    type_map: dict,
    *,
    condition: str = "C",
    top_n: int = 10,
) -> dict:
    """Structure patterns grouped by system constraint type.

    Returns:
        {
            "presence":  {"total": N, "patterns": [...]},
            "avoidance": {"total": N, "patterns": [...]},
            "ambiguous": {"total": N, "patterns": [...]},
        }
    """
    filtered = [r for r in records if r.get("condition") == condition]
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in filtered:
        ct = get_system_constraint_type(rec, type_map)
        groups[ct].append(rec)

    result = {}
    for ct in _CONSTRAINT_TYPES:
        recs = groups.get(ct, [])
        if not recs:
            result[ct] = {"total": 0, "patterns": []}
            continue
        pattern_counts: Counter = Counter()
        pattern_labels: dict[tuple, Counter] = defaultdict(Counter)
        for rec in recs:
            pat = _get_structure(rec)
            pattern_counts[pat] += 1
            pattern_labels[pat][rec.get("label", "?")] += 1

        patterns = []
        for pat, count in pattern_counts.most_common(top_n):
            patterns.append({
                "structure": pat,
                "count": count,
                "labels": dict(pattern_labels[pat]),
            })
        result[ct] = {"total": len(recs), "patterns": patterns}

    return result


def compute_conflict_structure(
    records: list[dict],
    type_map: dict | None = None,
    *,
    condition: str = "C",
    top_n: int = 10,
) -> dict:
    """Per-direction structure analysis for a single conflict's records.

    Returns:
        {
            "a_to_b": {"total": N, "sys_constraint": str, "patterns": [...]},
            "b_to_a": {"total": N, "sys_constraint": str, "patterns": [...]},
        }
    """
    filtered = [r for r in records if r.get("condition") == condition]
    result = {}
    for direction in ("a_to_b", "b_to_a"):
        dir_recs = [r for r in filtered if r.get("direction") == direction]
        if not dir_recs:
            result[direction] = {"total": 0, "sys_constraint": "?", "patterns": []}
            continue

        # Determine system constraint type for this direction
        sys_ct = "?"
        if type_map:
            sys_ct = get_system_constraint_type(dir_recs[0], type_map)

        pattern_counts: Counter = Counter()
        pattern_labels: dict[tuple, Counter] = defaultdict(Counter)
        for rec in dir_recs:
            pat = _get_structure(rec)
            pattern_counts[pat] += 1
            pattern_labels[pat][rec.get("label", "?")] += 1

        patterns = []
        for pat, count in pattern_counts.most_common(top_n):
            patterns.append({
                "structure": pat,
                "count": count,
                "labels": dict(pattern_labels[pat]),
            })
        result[direction] = {
            "total": len(dir_recs),
            "sys_constraint": sys_ct,
            "patterns": patterns,
        }

    return result


def compute_refusal_overview(
    records: list[dict],
    type_map: dict | None = None,
    *,
    condition: str = "C",
) -> list[dict]:
    """Per-conflict, per-direction refusal/meta/clean rates.

    Returns a list of row dicts sorted by refusal rate descending:
        [
            {
                "conflict": str,
                "direction": str,
                "sys_constraint": str,
                "total": int,
                "refusal": int,      # structure starts with 'refusal'
                "bare_refusal": int,  # structure is ['refusal'] or ['refusal','helpfulness_followup']
                "meta_only": int,     # has metacommentary, no refusal
                "clean": int,         # pure content
                "labels": Counter,    # verifier label distribution
            },
            ...
        ]
    """
    filtered = [r for r in records if r.get("condition") == condition]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in filtered:
        key = (rec.get("conflict_id", "?"), rec.get("direction", "?"))
        groups[key].append(rec)

    _BARE = (("refusal",), ("refusal", "helpfulness_followup"))

    rows = []
    for (cid, direction), recs in groups.items():
        n = len(recs)
        refusal = 0
        bare = 0
        meta_only = 0
        clean = 0
        labels: Counter = Counter()

        sys_ct = "?"
        if type_map and recs:
            sys_ct = get_system_constraint_type(recs[0], type_map)

        for rec in recs:
            pat = _get_structure(rec)
            labels[rec.get("label", "?")] += 1
            if pat[0] == "refusal":
                refusal += 1
                if pat in _BARE:
                    bare += 1
            elif "metacommentary" in pat:
                meta_only += 1
            elif pat == ("content",):
                clean += 1

        def _p(count: int) -> float:
            return round(count / n * 100, 1) if n else 0.0

        rows.append({
            "conflict": cid,
            "direction": direction,
            "sys_constraint": sys_ct,
            "total": n,
            "refusal": refusal,
            "bare_refusal": bare,
            "refusal_content": refusal - bare,
            "meta_only": meta_only,
            "clean": clean,
            "refusal_pct": _p(refusal),
            "bare_refusal_pct": _p(bare),
            "refusal_content_pct": _p(refusal - bare),
            "meta_only_pct": _p(meta_only),
            "clean_pct": _p(clean),
            "labels": labels,
        })

    # Group by conflict, sort by max refusal rate across directions
    by_conflict: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_conflict[row["conflict"]].append(row)

    def _max_refusal_rate(conflict_rows: list[dict]) -> float:
        return max(
            (r["refusal"] / r["total"] if r["total"] else 0) for r in conflict_rows
        )

    sorted_rows = []
    for cid in sorted(by_conflict, key=lambda c: _max_refusal_rate(by_conflict[c]), reverse=True):
        # Within a conflict, a_to_b first
        conflict_rows = sorted(by_conflict[cid], key=lambda r: r["direction"])
        sorted_rows.extend(conflict_rows)

    return sorted_rows


def sample_by_structure(
    records: list[dict],
    structure: list[str] | None = None,
    *,
    condition: str = "C",
    direction: str | None = None,
    label: str | None = None,
    segment_wc: dict[str, tuple[int | None, int | None]] | None = None,
    n: int = 10,
) -> list[dict]:
    """Return up to n records matching filters.

    Args:
        records: JSONL records.
        structure: e.g. ["refusal", "metacommentary", "content"]. None = any.
        condition: filter to this condition.
        direction: optional direction filter.
        label: optional label filter.
        segment_wc: per-segment word count filter. Maps segment type to
            (min, max) bounds (None = unbounded). Applies to the TOTAL
            word count across all segments of that type.
            E.g. {"content": (None, 20)} = content segments sum to ≤20 words.
        n: max records to return.
    """
    target = tuple(structure) if structure else None
    result = []
    for rec in records:
        if rec.get("condition") != condition:
            continue
        if direction and rec.get("direction") != direction:
            continue
        if label and rec.get("label") != label:
            continue
        pat = _get_structure(rec)
        if target and pat != target:
            continue
        if segment_wc:
            tags = rec.get("refusal_tags")
            if not tags:
                continue
            ok = True
            for seg_type, (lo, hi) in segment_wc.items():
                wc = sum(
                    w for s, w in zip(tags["structure"], tags["word_counts"])
                    if s == seg_type
                )
                if lo is not None and wc < lo:
                    ok = False
                    break
                if hi is not None and wc > hi:
                    ok = False
                    break
            if not ok:
                continue
        result.append(rec)
        if len(result) >= n:
            break
    return result


def print_samples(records: list[dict], *, max_resp_len: int = 300) -> None:
    """Print sampled records with structure and response text."""
    for i, rec in enumerate(records, 1):
        cid = rec.get("conflict_id", "?")
        direction = rec.get("direction", "?")
        label = rec.get("label", "?")
        tags = rec.get("refusal_tags")
        struct = tags["structure"] if tags else ["content"]
        wc = tags["word_counts"] if tags else []

        resp = rec.get("response", "").strip()
        if len(resp) > max_resp_len:
            resp = resp[:max_resp_len] + "..."

        wc_str = ", ".join(f"{s}:{w}" for s, w in zip(struct, wc)) if wc else ""
        print(f"\n  [{i}] {cid} {direction} label={label}")
        print(f"      [{' -> '.join(struct)}]  ({wc_str})")
        print(f"      \"{resp}\"")


# ---------------------------------------------------------------------------
# Print functions — format computed data for console
# ---------------------------------------------------------------------------


def print_structure_stats(stats: dict, *, model: str = "", title: str = "") -> None:
    """Print structure pattern distribution with label breakdown."""
    if title:
        print(f"\n  {title}")
    if model:
        print(f"  Model: {_short_model_name(model)} (N={stats['total']:,})")
    print()
    print(
        f"  {'Pattern':<50} {'N':>6} {'%':>6}  "
        f"{chr(0x2192)+'sys':>6} {chr(0x2192)+'usr':>6} {chr(0x2192)+'nei':>6}"
    )
    print(f"  {'-' * 50} {'-' * 6} {'-' * 6}  {'-' * 6} {'-' * 6} {'-' * 6}")

    for p in stats["patterns"]:
        n = p["count"]
        labels = p["labels"]
        print(
            f"  {_pattern_str(p['structure']):<50} {n:>6} "
            f"{_pct(n, stats['total']):>6}  "
            f"{_pct(labels.get('followed_system', 0), n):>6} "
            f"{_pct(labels.get('followed_user', 0), n):>6} "
            f"{_pct(labels.get('followed_neither', 0), n):>6}"
        )


def print_structure_by_constraint(data: dict, *, model: str = "") -> None:
    """Print structure patterns grouped by system constraint type."""
    if model:
        print(f"\n  Model: {_short_model_name(model)}")

    for ct in _CONSTRAINT_TYPES:
        info = data.get(ct, {"total": 0, "patterns": []})
        if info["total"] == 0:
            continue
        print(f"\n  System constraint: {ct} (N={info['total']:,})")
        print(
            f"    {'Pattern':<48} {'N':>6} {'%':>6}  "
            f"{chr(0x2192)+'sys':>6} {chr(0x2192)+'usr':>6}"
        )
        print(f"    {'-' * 48} {'-' * 6} {'-' * 6}  {'-' * 6} {'-' * 6}")
        for p in info["patterns"]:
            n = p["count"]
            labels = p["labels"]
            print(
                f"    {_pattern_str(p['structure']):<48} {n:>6} "
                f"{_pct(n, info['total']):>6}  "
                f"{_pct(labels.get('followed_system', 0), n):>6} "
                f"{_pct(labels.get('followed_user', 0), n):>6}"
            )


def print_conflict_structure(
    stats: dict, *, model: str = "", conflict: str = "",
) -> None:
    """Print per-direction structure for a single conflict."""
    header = conflict or "conflict"
    if model:
        header = f"{_short_model_name(model)} / {header}"
    print(f"\n  {header}")

    for direction in ("a_to_b", "b_to_a"):
        info = stats.get(direction, {"total": 0, "patterns": []})
        if info["total"] == 0:
            continue
        sys_ct = info.get("sys_constraint", "?")
        print(f"\n    {direction} (system={sys_ct}, N={info['total']:,})")
        print(
            f"      {'Pattern':<46} {'N':>5} {'%':>6}  "
            f"{chr(0x2192)+'sys':>6} {chr(0x2192)+'usr':>6}"
        )
        print(f"      {'-' * 46} {'-' * 5} {'-' * 6}  {'-' * 6} {'-' * 6}")
        for p in info["patterns"]:
            n = p["count"]
            labels = p["labels"]
            print(
                f"      {_pattern_str(p['structure']):<46} {n:>5} "
                f"{_pct(n, info['total']):>6}  "
                f"{_pct(labels.get('followed_system', 0), n):>6} "
                f"{_pct(labels.get('followed_user', 0), n):>6}"
            )


def print_refusal_overview(rows: list[dict], *, model: str = "") -> None:
    """Print per-conflict, per-direction refusal rate table."""
    if model:
        print(f"\n  Model: {_short_model_name(model)} — condition C refusal overview")
    print()
    print(
        f"  {'Conflict':<35} {'Dir':<7} {'Sys':<10} "
        f"{'N':>5} {'Bare':>6} {'R+C':>6} {'M+C':>6} {'Clean':>6} "
        f"{chr(0x2192)+'sys':>6} {chr(0x2192)+'usr':>6}"
    )
    print(
        f"  {'-' * 35} {'-' * 7} {'-' * 10} "
        f"{'-' * 5} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} "
        f"{'-' * 6} {'-' * 6}"
    )
    for row in rows:
        n = row["total"]
        if n == 0:
            continue
        labels = row["labels"]
        ref_cont = row["refusal"] - row["bare_refusal"]
        print(
            f"  {row['conflict']:<35} {row['direction']:<7} {row['sys_constraint']:<10} "
            f"{n:>5} {_pct(row['bare_refusal'], n):>6} {_pct(ref_cont, n):>6} "
            f"{_pct(row['meta_only'], n):>6} {_pct(row['clean'], n):>6} "
            f"{_pct(labels.get('followed_system', 0), n):>6} "
            f"{_pct(labels.get('followed_user', 0), n):>6}"
        )


def print_refusal_summary(
    rows: list[dict], *, model: str = "", conflict: str = "",
) -> None:
    """Print compact per-direction aggregate summary for a single conflict."""
    header = conflict or "conflict"
    if model:
        header = f"{_short_model_name(model)} / {header}"
    print(f"\n  {header} — structure summary")
    for row in rows:
        n = row["total"]
        if n == 0:
            continue
        d = row["direction"]
        sc = row["sys_constraint"]
        ref_cont = row["refusal"] - row["bare_refusal"]
        print(
            f"    {d} (sys={sc}, N={n:,}): "
            f"bare_ref={_pct(row['bare_refusal'], n)}  "
            f"ref+cont={_pct(ref_cont, n)}  "
            f"meta+cont={_pct(row['meta_only'], n)}  "
            f"clean={_pct(row['clean'], n)}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-tabulate response structure x constraint type x verifier label.",
        prog="python -m phase0_v2.calibration.response_type_analysis",
    )
    parser.add_argument("files", nargs="+", help="JSONL results files")
    parser.add_argument(
        "--conflict",
        help="Analyze a single conflict (uses fast filtered loading)",
    )
    parser.add_argument(
        "--focus",
        choices=["ambiguous", "by-constraint", "overview"],
        help="Focus mode: 'overview' (per-conflict refusal rates), 'by-constraint', or 'ambiguous'",
    )
    parser.add_argument(
        "--structure",
        help="Sample responses matching this structure pattern (comma-separated, e.g. 'refusal,metacommentary,content')",
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="Number of samples to show with --structure (default: 10)",
    )
    parser.add_argument(
        "--direction",
        help="Filter by direction (a_to_b, b_to_a) — used with --structure",
    )
    parser.add_argument(
        "--label",
        help="Filter by label (followed_system, followed_user, etc.) — used with --structure",
    )
    parser.add_argument(
        "--max-wc",
        help="Max word count for a segment type, e.g. 'content:20' or 'refusal:5,content:20'",
    )
    parser.add_argument(
        "--min-wc",
        help="Min word count for a segment type, e.g. 'content:50'",
    )
    parser.add_argument(
        "--conflicts-yaml",
        help="Path to conflicts.yaml (default: auto-detected)",
    )

    args = parser.parse_args(argv)

    yaml_path = Path(args.conflicts_yaml) if args.conflicts_yaml else None
    type_map = load_constraint_type_map(yaml_path)

    for fp in args.files:
        if args.structure or args.max_wc or args.min_wc:
            # Sample mode: show responses matching filters
            struct = [s.strip() for s in args.structure.split(",")] if args.structure else None
            if args.conflict:
                _, records = load_records_filtered(fp, args.conflict)
            else:
                records = load_records(fp)

            # Parse word count bounds
            segment_wc: dict[str, tuple[int | None, int | None]] | None = None
            if args.max_wc or args.min_wc:
                segment_wc = {}
                if args.max_wc:
                    for part in args.max_wc.split(","):
                        seg, val = part.strip().split(":")
                        segment_wc.setdefault(seg, (None, None))
                        segment_wc[seg] = (segment_wc[seg][0], int(val))
                if args.min_wc:
                    for part in args.min_wc.split(","):
                        seg, val = part.strip().split(":")
                        segment_wc.setdefault(seg, (None, None))
                        segment_wc[seg] = (int(val), segment_wc[seg][1])

            samples = sample_by_structure(
                records, struct,
                direction=args.direction,
                label=args.label,
                segment_wc=segment_wc,
                n=args.sample,
            )
            filter_desc = f"[{' -> '.join(struct)}]" if struct else "(any structure)"
            if args.max_wc:
                filter_desc += f" max-wc={args.max_wc}"
            if args.min_wc:
                filter_desc += f" min-wc={args.min_wc}"
            if not samples:
                print(f"  No matches for {filter_desc}")
            else:
                print(f"  {len(samples)} samples matching {filter_desc}")
                print_samples(samples)
        elif args.conflict:
            # Fast path: load only this conflict's records
            model_id, records = load_records_filtered(fp, args.conflict)
            model = model_id or "unknown"
            # Aggregate summary first
            overview = compute_refusal_overview(records, type_map)
            print_refusal_summary(overview, model=model, conflict=args.conflict)
            # Then detailed pattern table
            stats = compute_conflict_structure(records, type_map)
            print_conflict_structure(stats, model=model, conflict=args.conflict)
        elif args.focus == "by-constraint":
            records = load_records(fp)
            model = next((r.get("model", "unknown") for r in records if r.get("model")), "unknown")
            data = compute_structure_by_constraint(records, type_map)
            print()
            print("=" * 80)
            print(f"  STRUCTURE BY CONSTRAINT TYPE: {_short_model_name(model)}")
            print("=" * 80)
            print_structure_by_constraint(data, model=model)
        elif args.focus == "overview":
            records = load_records(fp)
            model = next((r.get("model", "unknown") for r in records if r.get("model")), "unknown")
            rows = compute_refusal_overview(records, type_map)
            print()
            print("=" * 80)
            print(f"  REFUSAL OVERVIEW: {_short_model_name(model)}")
            print("=" * 80)
            print_refusal_overview(rows, model=model)
        elif args.focus == "ambiguous":
            records = load_records(fp)
            model = next((r.get("model", "unknown") for r in records if r.get("model")), "unknown")
            # Filter to ambiguous conflicts only
            ambig_cids = {
                cid for cid, types in type_map.items()
                if types.get("a") == "ambiguous" or types.get("b") == "ambiguous"
            }
            ambig_recs = [r for r in records if r.get("conflict_id") in ambig_cids]
            stats = compute_structure_stats(ambig_recs, type_map)
            print()
            print("=" * 80)
            print(f"  AMBIGUOUS CONSTRAINTS: {_short_model_name(model)}")
            print("=" * 80)
            print_structure_stats(stats, model=model)
        else:
            # Full analysis
            records = load_records(fp)
            model = next((r.get("model", "unknown") for r in records if r.get("model")), "unknown")
            short = _short_model_name(model)

            c_recs = [r for r in records if r.get("condition") == "C"]
            if not c_recs:
                print(f"\n  {short}: no condition C records, skipping.")
                continue

            print()
            print("=" * 80)
            print(f"  RESPONSE STRUCTURE ANALYSIS: {short}")
            print("=" * 80)

            stats = compute_structure_stats(c_recs, type_map)
            print_structure_stats(stats, title="All condition C patterns")

            data = compute_structure_by_constraint(c_recs, type_map)
            print()
            print("-" * 80)
            print_structure_by_constraint(data, model=model)

    print()


if __name__ == "__main__":
    main()
