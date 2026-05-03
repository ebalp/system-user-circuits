"""Cross-cutting response structure tagger — CLI and analysis tools.

Core segmentation logic lives in phase0_v2.conflicts.preprocessing.
This module provides:
  - CLI subcommands: explore, tag, report
  - Record-level helpers for dataset analysis
  - Re-exports of preprocessing functions for backward compatibility

Usage:
    uv run python -m phase0_v2.calibration.refusal_tagger explore \
      results.jsonl --pattern "^I (?:cannot|can't)" --sample 10 --condition C

    uv run python -m phase0_v2.calibration.refusal_tagger tag \
      results.jsonl  # writes refusal_tags in place

    uv run python -m phase0_v2.calibration.refusal_tagger report \
      results_8b.jsonl results_70b.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from ._shared import load_records

# Core segmentation — canonical location is phase0_v2.conflicts.preprocessing
from phase0_v2.conflicts.preprocessing import (
    REFUSAL_PREFIX_RE as _REFUSAL_PREFIX_RE,
    classify_response,
    content_word_count,
    extract_content,
    has_meta,
    has_meta_commentary,
    has_refusal_prefix,
    is_bare_refusal,
    refusal_word_count,
    tag_response,
)

# Re-export for backward compatibility
__all__ = [
    "_REFUSAL_PREFIX_RE",
    "classify_response",
    "content_word_count",
    "extract_content",
    "has_meta",
    "has_meta_commentary",
    "has_refusal_prefix",
    "is_bare_refusal",
    "refusal_word_count",
    "tag_response",
]

DEFAULT_BARE_THRESHOLD = 10
_MAX_BARE_REFUSAL_WORDS = 25


# ---------------------------------------------------------------------------
# Record-level helpers
# ---------------------------------------------------------------------------


def _get_tags(rec: dict, *, recompute: bool = False) -> dict | None:
    """Get refusal tags for a record, using precomputed tags when available."""
    if not recompute:
        tags = rec.get("refusal_tags")
        if tags is not None:
            return tags

    return tag_response(
        rec.get("response", ""),
        rec.get("conflict_id"),
    )


def _model_from_records(records: list[dict]) -> str:
    """Extract model name from first record."""
    for rec in records:
        model = rec.get("model")
        if model:
            return model
    return "unknown"


def _filter_records(
    records: list[dict],
    *,
    condition: str | None = None,
    conflict: str | None = None,
    direction: str | None = None,
    label: str | None = None,
) -> list[dict]:
    """Filter records by condition/conflict/direction/label."""
    pool = records
    if condition:
        pool = [r for r in pool if r.get("condition") == condition]
    if conflict:
        pool = [r for r in pool if r.get("conflict_id") == conflict]
    if direction:
        pool = [r for r in pool if r.get("direction") == direction]
    if label:
        pool = [r for r in pool if r.get("label") == label]
    return pool


# ---------------------------------------------------------------------------
# Explore subcommand
# ---------------------------------------------------------------------------


def _run_explore(records: list[dict], args) -> None:
    """Test a regex pattern (or built-in tagger) against records."""
    model = _model_from_records(records)
    recompute = getattr(args, "recompute", False)

    filtered = _filter_records(
        records,
        condition=args.condition,
        conflict=args.conflict,
        direction=args.direction,
        label=args.label,
    )

    # Custom --pattern always uses live regex; otherwise use precomputed tags
    if args.pattern:
        try:
            pat = re.compile(args.pattern, re.IGNORECASE)
        except re.error as e:
            print(f"Invalid regex: {e}", file=sys.stderr)
            sys.exit(1)

        def match_fn(rec):
            resp = rec.get("response", "").replace("\u2019", "'").strip()
            return bool(pat.search(resp))
    else:

        def match_fn(rec):
            tags = _get_tags(rec, recompute=recompute)
            return tags is not None

    matches = [r for r in filtered if match_fn(r)]

    if args.sample:
        # Sample mode
        sample = matches
        if len(sample) > args.sample:
            sample = random.sample(sample, args.sample)

        for i, rec in enumerate(sample, 1):
            cond = rec.get("condition", "?")
            direction = rec.get("direction", "?")
            cid = rec.get("conflict_id", "?")
            lbl = rec.get("label", "?")
            resp = rec.get("response", "").replace("\u2019", "'")

            tags = _get_tags(rec, recompute=recompute)

            display_resp = resp.strip()
            if len(display_resp) > 200:
                display_resp = display_resp[:200] + "..."

            print(
                f"\n[{i}] condition={cond} direction={direction} "
                f"conflict={cid} label={lbl}"
            )
            print(f'    "{display_resp}"')
            if tags:
                struct_str = " -> ".join(tags["structure"])
                wc_str = ", ".join(
                    f"{s}:{w}" for s, w in zip(tags["structure"], tags["word_counts"])
                )
                print(f"    [{struct_str}]  ({wc_str})")
        return

    # Stats mode
    pattern_desc = args.pattern if args.pattern else "(built-in structure tagger)"
    total = len(filtered)
    n_matches = len(matches)
    pct = (n_matches / total * 100) if total else 0

    print(f"Pattern: {pattern_desc}")
    print(f"Model: {model}")
    print(f"Total records: {total:,} | Matches: {n_matches:,} ({pct:.2f}%)")

    # By condition
    by_cond = defaultdict(int)
    cond_totals = defaultdict(int)
    for r in filtered:
        cond_totals[r.get("condition", "?")] += 1
    for r in matches:
        by_cond[r.get("condition", "?")] += 1

    cond_parts = []
    for c in ("A", "B", "C", "D"):
        cnt = by_cond.get(c, 0)
        tot = cond_totals.get(c, 0)
        pct_c = (cnt / tot * 100) if tot else 0
        cond_parts.append(f"{c}: {cnt} ({pct_c:.2f}%)")
    print(f"\nBy condition:    {'  '.join(cond_parts)}")

    # By label (condition C only)
    c_matches = [r for r in matches if r.get("condition") == "C"]
    if c_matches:
        by_label = defaultdict(int)
        for r in c_matches:
            by_label[r.get("label", "?")] += 1
        label_parts = []
        for lbl in (
            "followed_system",
            "followed_user",
            "followed_neither",
            "followed_both",
        ):
            label_parts.append(
                f"{lbl.replace('followed_', '')}: {by_label.get(lbl, 0)}"
            )
        print(f"By label (C):    {'  '.join(label_parts)}")

        by_dir = defaultdict(int)
        for r in c_matches:
            by_dir[r.get("direction", "?")] += 1
        dir_parts = [f"{d}: {by_dir.get(d, 0)}" for d in ("a_to_b", "b_to_a")]
        print(f"By direction (C): {'  '.join(dir_parts)}")

    # Bare refusal count
    bare_count = 0
    prefix_count = 0
    for r in matches:
        tags = _get_tags(r, recompute=recompute)
        if has_refusal_prefix(tags):
            prefix_count += 1
            if is_bare_refusal(tags):
                bare_count += 1

    if prefix_count > 0:
        bare_pct = bare_count / prefix_count * 100
        print(
            f"Bare refusals: {bare_count} / {prefix_count} ({bare_pct:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Tag subcommand
# ---------------------------------------------------------------------------


def _run_tag(args) -> None:
    """Persist refusal tags into JSONL records."""
    print(f"Loading records from {args.input_file}...")
    records = load_records(args.input_file)
    print(f"Loaded {len(records)} records")

    tagged = 0
    for i, rec in enumerate(records):
        if i % 10000 == 0 and i > 0:
            print(f"  processing {i}/{len(records)}...", file=sys.stderr)
        tags = tag_response(
            rec.get("response", ""),
            rec.get("conflict_id"),
        )
        if tags:
            rec["refusal_tags"] = tags
            tagged += 1
        elif "refusal_tags" in rec:
            del rec["refusal_tags"]

    output = Path(args.output) if args.output else Path(args.input_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Tagged {tagged}/{len(records)} records with structure")
    print(f"Wrote {output}{' (in place)' if not args.output else ''}")


# ---------------------------------------------------------------------------
# Report subcommand
# ---------------------------------------------------------------------------


def _run_report(args) -> None:
    """Cross-model, cross-conflict refusal prevalence report."""
    recompute = getattr(args, "recompute", False)

    # Load all models, resolve tags (precomputed or live)
    model_data: list[tuple[str, list[dict]]] = []
    for fp in args.files:
        records = load_records(fp)
        model = _model_from_records(records)
        for rec in records:
            rec["_tags"] = _get_tags(rec, recompute=recompute)
        model_data.append((model, records))

    # Table 1: Per-model prevalence by condition
    print("=" * 70)
    print("TABLE 1: PER-MODEL REFUSAL PREVALENCE BY CONDITION")
    print("=" * 70)
    print(
        f"{'Model':<30} {'Cond':>4} {'Total':>6} "
        f"{'Prefix':>8} {'Bare':>8} {'Meta':>8}"
    )
    print("-" * 70)

    for model, records in model_data:
        short = model.split("/")[-1] if "/" in model else model
        if len(short) > 28:
            short = short[:25] + "..."

        for cond in ("A", "B", "C", "D"):
            cond_recs = [r for r in records if r.get("condition") == cond]
            n = len(cond_recs)
            if n == 0:
                continue

            n_prefix = sum(1 for r in cond_recs if has_refusal_prefix(r["_tags"]))
            n_bare = sum(1 for r in cond_recs if is_bare_refusal(r["_tags"]))
            n_meta = sum(1 for r in cond_recs if has_meta(r["_tags"]))

            def _pct(count, total=n):
                return f"{count / total * 100:.1f}%" if total else "0%"

            print(
                f"{short:<30} {cond:>4} {n:>6} "
                f"{_pct(n_prefix):>8} {_pct(n_bare):>8} {_pct(n_meta):>8}"
            )
        print()

    # Table 2: Top conflicts by bare refusal (condition C, per model)
    print("=" * 70)
    print("TABLE 2: TOP CONFLICTS BY BARE REFUSAL (CONDITION C)")
    print("=" * 70)

    for model, records in model_data:
        short = model.split("/")[-1] if "/" in model else model
        c_recs = [r for r in records if r.get("condition") == "C"]

        by_conflict: dict[str, dict] = defaultdict(
            lambda: {"bare": 0, "prefix": 0, "total": 0, "labels": defaultdict(int)}
        )
        for r in c_recs:
            cid = r.get("conflict_id", "?")
            by_conflict[cid]["total"] += 1
            tags = r["_tags"]
            if has_refusal_prefix(tags):
                by_conflict[cid]["prefix"] += 1
            if is_bare_refusal(tags):
                by_conflict[cid]["bare"] += 1
                by_conflict[cid]["labels"][r.get("label", "?")] += 1

        sorted_conflicts = sorted(
            by_conflict.items(), key=lambda x: x[1]["bare"], reverse=True
        )
        top = [item for item in sorted_conflicts if item[1]["bare"] > 0][:15]

        if top:
            print(f"\n{short}:")
            print(
                f"  {'Conflict':<40} {'Bare':>5} {'Prefix':>7} "
                f"{'Labels'}"
            )
            print(f"  {'-' * 40} {'-' * 5} {'-' * 7} {'-' * 30}")
            for cid, info in top:
                lbl_str = ", ".join(
                    f"{k.replace('followed_', '')}={v}"
                    for k, v in sorted(info["labels"].items())
                )
                print(
                    f"  {cid:<40} {info['bare']:>5} {info['prefix']:>7} "
                    f"{lbl_str}"
                )

    # Table 3: Label concerns — bare refusals labeled followed_system or followed_user
    print()
    print("=" * 70)
    print("TABLE 3: LABEL CONCERNS (BARE REFUSALS WITH followed_system/followed_user)")
    print("=" * 70)

    for model, records in model_data:
        short = model.split("/")[-1] if "/" in model else model
        c_recs = [r for r in records if r.get("condition") == "C"]

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in c_recs:
            key = (r.get("conflict_id", "?"), r.get("direction", "?"))
            groups[key].append(r)

        concerns = []
        for (cid, direction), recs_group in sorted(groups.items()):
            bare_sys = 0
            bare_usr = 0
            for r in recs_group:
                if is_bare_refusal(r["_tags"]):
                    lbl = r.get("label", "")
                    if lbl == "followed_system":
                        bare_sys += 1
                    elif lbl == "followed_user":
                        bare_usr += 1

            if bare_sys > 0 or bare_usr > 0:
                concerns.append((cid, direction, bare_sys, bare_usr))

        if concerns:
            print(f"\n{short}:")
            print(
                f"  {'Conflict':<40} {'Dir':>7} "
                f"{chr(0x2192)+'sys':>5} {chr(0x2192)+'usr':>5}"
            )
            print(f"  {'-' * 40} {'-' * 7} {'-' * 5} {'-' * 5}")
            for cid, direction, bs, bu in concerns:
                print(f"  {cid:<40} {direction:>7} {bs:>5} {bu:>5}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-cutting response structure tagger for dataset-wide analysis.",
        prog="python -m phase0_v2.calibration.refusal_tagger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- explore ---
    explore_p = subparsers.add_parser(
        "explore", help="Test regex patterns against real data"
    )
    explore_p.add_argument("input_file", help="JSONL results file")
    explore_p.add_argument(
        "--pattern",
        help="Custom regex to test (default: built-in structure tagger)",
    )
    explore_p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Show N matching responses (0 = stats mode)",
    )
    explore_p.add_argument("--condition", help="Filter by condition (A, B, C, D)")
    explore_p.add_argument("--conflict", help="Filter by conflict ID")
    explore_p.add_argument(
        "--direction", help="Filter by direction (a_to_b, b_to_a)"
    )
    explore_p.add_argument("--label", help="Filter by current label")
    explore_p.add_argument(
        "--recompute",
        action="store_true",
        help="Force live recomputation instead of using precomputed refusal_tags",
    )

    # --- tag ---
    tag_p = subparsers.add_parser(
        "tag", help="Persist refusal_tags into JSONL records"
    )
    tag_p.add_argument("input_file", help="Input JSONL results file")
    tag_p.add_argument(
        "--output",
        default=None,
        help="Output JSONL file (default: overwrite input file in place)",
    )

    # --- report ---
    report_p = subparsers.add_parser(
        "report", help="Cross-model refusal prevalence report"
    )
    report_p.add_argument("files", nargs="+", help="JSONL results files")
    report_p.add_argument(
        "--recompute",
        action="store_true",
        help="Force live recomputation instead of using precomputed refusal_tags",
    )

    args = parser.parse_args(argv)

    if args.command == "explore":
        print(f"Loading records from {args.input_file}...")
        records = load_records(args.input_file)
        print(f"Loaded {len(records)} records")
        _run_explore(records, args)
    elif args.command == "tag":
        _run_tag(args)
    elif args.command == "report":
        _run_report(args)


if __name__ == "__main__":
    main()
