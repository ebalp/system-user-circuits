"""Threshold rescoring and re-verification of existing results without re-querying.

Two modes:
  1. Rescore (default): re-apply thresholds to stored float scores.
  2. Reverify (--reverify): re-run verify functions on stored response text,
     producing fresh scores and results. Use after changing verifier logic.

Usage:
    uv run python -m phase0_v2.calibration.rescore <input_file> <output_file> [options]

Examples:
    # No-op verification run (uses current registry thresholds):
    uv run python -m phase0_v2.calibration.rescore results.jsonl rescored.jsonl

    # Per-conflict threshold overrides:
    uv run python -m phase0_v2.calibration.rescore results.jsonl rescored.jsonl --thresholds '{"sentence_chaining": 0.3}'

    # Global threshold for all conflicts:
    uv run python -m phase0_v2.calibration.rescore results.jsonl rescored.jsonl --global-threshold 0.5

    # Re-run verify functions after changing verifier code:
    uv run python -m phase0_v2.calibration.rescore results.jsonl reverified.jsonl --reverify

    # Reverify only specific conflicts:
    uv run python -m phase0_v2.calibration.rescore results.jsonl reverified.jsonl --reverify --conflicts first_vs_third_person,bullets_and_sub_bullets
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ._shared import (
    load_records,
    build_conflict_threshold_map,
    direction_to_verify_code,
    apply_threshold,
    compute_label,
    load_model_thresholds,
)


def _parse_thresholds(thresholds_arg: str | None) -> dict[str, float]:
    """Parse --thresholds argument (JSON string or path to JSON file)."""
    if thresholds_arg is None:
        return {}
    # Try as file path first
    path = Path(thresholds_arg)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        # Try as inline JSON
        try:
            data = json.loads(thresholds_arg)
        except json.JSONDecodeError:
            print(f"Error: --thresholds must be a JSON string or path to a JSON file", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, dict):
        print("Error: --thresholds must be a JSON object mapping conflict_id to threshold", file=sys.stderr)
        sys.exit(1)

    return {k: float(v) for k, v in data.items()}


def _reverify_record(rec: dict, conflict, direction_code: str) -> dict:
    """Re-run verify functions on stored response text, return updated record."""
    response = rec.get("response", "")
    instruction_args = rec.get("instruction_args", {})

    # Store args so verify functions can access them
    if instruction_args:
        conflict.build_system_prompt(direction=direction_code, **instruction_args)

    sys_ok = conflict.verify_followed_system(response, direction=direction_code)
    usr_ok = conflict.verify_followed_user(response, direction=direction_code)
    sys_score = conflict.score_system(response, direction=direction_code)
    usr_score = conflict.score_user(response, direction=direction_code)

    label = compute_label(sys_ok, usr_ok)

    updated = dict(rec)
    updated["verify_system_result"] = sys_ok
    updated["verify_user_result"] = usr_ok
    updated["verify_system_score"] = sys_score
    updated["verify_user_score"] = usr_score
    updated["label"] = label
    return updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rescore or reverify results."
    )
    parser.add_argument("input_file", help="Input JSONL results file")
    parser.add_argument("output_file", help="Output JSONL file with recomputed labels")
    parser.add_argument(
        "--thresholds",
        default=None,
        help='JSON string or path to JSON file: {"conflict_id": threshold}',
    )
    parser.add_argument(
        "--global-threshold",
        type=float,
        default=None,
        help="Single threshold for all conflicts (overrides per-conflict)",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="Re-run verify functions on response text (use after changing verifier code)",
    )
    parser.add_argument(
        "--conflicts",
        default=None,
        help="Comma-separated conflict IDs to reverify (default: all). Only used with --reverify.",
    )
    parser.add_argument(
        "--model-config", default=None,
        help="Model ID to load per-model thresholds from experiment config",
    )
    parser.add_argument(
        "--config", default="phase0_v2/config/experiment.yaml",
        help="Path to experiment config (used with --model-config)",
    )
    args = parser.parse_args(argv)

    if args.reverify and (args.thresholds or args.global_threshold):
        print("Error: --reverify cannot be combined with --thresholds or --global-threshold", file=sys.stderr)
        sys.exit(1)

    print(f"Loading records from {args.input_file}...")
    records = load_records(args.input_file)
    print(f"Loaded {len(records)} records")

    if args.reverify:
        _run_reverify(records, args)
    else:
        _run_rescore(records, args)


def _run_reverify(records: list[dict], args) -> None:
    """Re-run verify functions on stored response text."""
    from ..conflicts.registry import get_conflict

    conflict_filter = None
    if args.conflicts:
        conflict_filter = set(args.conflicts.split(","))
        print(f"Reverifying conflicts: {conflict_filter}")
    else:
        print("Reverifying all conflicts")

    total = 0
    skipped_error = 0
    skipped_unknown = 0
    skipped_filter = 0
    changes: dict[str, list[dict]] = defaultdict(list)
    output_records = []

    for rec in records:
        if rec.get("error") is not None:
            output_records.append(rec)
            skipped_error += 1
            continue

        cid = rec["conflict_id"]

        # If filtering, pass through non-target conflicts unchanged
        if conflict_filter and cid not in conflict_filter:
            output_records.append(rec)
            skipped_filter += 1
            continue

        conflict = get_conflict(cid)
        if conflict is None:
            print(f"  Warning: unknown conflict_id '{cid}', passing through", file=sys.stderr)
            output_records.append(rec)
            skipped_unknown += 1
            continue

        direction_code = direction_to_verify_code(rec["direction"])
        total += 1

        updated = _reverify_record(rec, conflict, direction_code)

        old_label = rec["label"]
        new_label = updated["label"]
        if old_label != new_label:
            changes[cid].append({"old_label": old_label, "new_label": new_label})

        output_records.append(updated)

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    print(f"\n{'=' * 70}")
    print("REVERIFY SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total records:      {len(records)}")
    print(f"Reverified:         {total}")
    print(f"Skipped (errors):   {skipped_error}")
    print(f"Skipped (unknown):  {skipped_unknown}")
    if skipped_filter:
        print(f"Skipped (filtered): {skipped_filter}")

    total_changes = sum(len(v) for v in changes.values())
    print(f"Labels changed:     {total_changes}")

    if changes:
        print(f"\n{'conflict_id':<40} {'changed':>8}")
        print("-" * 50)
        for cid in sorted(changes.keys()):
            print(f"{cid:<40} {len(changes[cid]):>8}")

        print(f"\nLabel transitions:")
        transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        for cid, change_list in changes.items():
            for ch in change_list:
                transition_counts[(ch["old_label"], ch["new_label"])] += 1
        for (old, new), count in sorted(transition_counts.items()):
            print(f"  {old:<20} -> {new:<20} {count:>5}")
    else:
        print("\nNo label changes — verifiers produced identical results.")

    print(f"\nWrote {output_path}")
    print("Done.")


def _run_rescore(records: list[dict], args) -> None:
    """Re-apply thresholds to stored scores."""
    threshold_overrides = _parse_thresholds(args.thresholds)
    global_threshold = args.global_threshold

    # Load per-model thresholds if --model-config is specified
    model_thresholds: dict[str, float] = {}
    if args.model_config:
        model_thresholds = load_model_thresholds(args.config, args.model_config)
        if model_thresholds:
            print(f"Using per-model thresholds for '{args.model_config}': {len(model_thresholds)} overrides")
        else:
            print(f"No thresholds found for model '{args.model_config}' in {args.config}")

    # Merge: CLI --thresholds take precedence over model config thresholds
    merged_overrides = dict(model_thresholds)
    merged_overrides.update(threshold_overrides)

    threshold_map = build_conflict_threshold_map(threshold_overrides=merged_overrides)

    if global_threshold is not None:
        print(f"Global threshold: {global_threshold}")
    if merged_overrides:
        print(f"Per-conflict overrides: {merged_overrides}")
    if global_threshold is None and not merged_overrides:
        print("No threshold overrides — using current registry thresholds (verification run)")

    total = 0
    skipped_error = 0
    skipped_unknown = 0
    changes: dict[str, list[dict]] = defaultdict(list)
    threshold_used: dict[str, tuple[float, float]] = {}

    output_records = []

    for rec in records:
        if rec.get("error") is not None:
            output_records.append(rec)
            skipped_error += 1
            continue

        total += 1
        cid = rec["conflict_id"]
        info = threshold_map.get(cid)

        if info is None:
            print(f"  Warning: unknown conflict_id '{cid}', passing through unchanged", file=sys.stderr)
            output_records.append(rec)
            skipped_unknown += 1
            continue

        direction_code = direction_to_verify_code(rec["direction"])

        if direction_code not in info.sides:
            output_records.append(rec)
            skipped_unknown += 1
            continue

        old_threshold = info.threshold
        if global_threshold is not None:
            effective_threshold = global_threshold
        elif cid in merged_overrides:
            effective_threshold = merged_overrides[cid]
        else:
            effective_threshold = old_threshold

        if cid not in threshold_used:
            threshold_used[cid] = (old_threshold, effective_threshold)

        sys_score = rec.get("verify_system_score")
        usr_score = rec.get("verify_user_score")

        if sys_score is None or usr_score is None:
            output_records.append(rec)
            continue

        sys_inverted = info.sides[direction_code]["system"].is_inverted
        usr_inverted = info.sides[direction_code]["user"].is_inverted

        new_sys_result = apply_threshold(sys_score, effective_threshold, sys_inverted)
        new_usr_result = apply_threshold(usr_score, effective_threshold, usr_inverted)
        new_label = compute_label(new_sys_result, new_usr_result)

        old_label = rec["label"]
        if old_label != new_label:
            changes[cid].append({"old_label": old_label, "new_label": new_label})

        updated = dict(rec)
        updated["verify_system_result"] = new_sys_result
        updated["verify_user_result"] = new_usr_result
        updated["label"] = new_label
        output_records.append(updated)

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    # Summary
    print(f"\n{'=' * 70}")
    print("RESCORE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total records:      {len(records)}")
    print(f"Scored records:     {total}")
    print(f"Skipped (errors):   {skipped_error}")
    print(f"Skipped (unknown):  {skipped_unknown}")

    total_changes = sum(len(v) for v in changes.values())
    print(f"Labels changed:     {total_changes}")

    if changes:
        print(f"\n{'conflict_id':<40} {'old_thresh':>10} {'new_thresh':>10} {'changed':>8}")
        print("-" * 70)
        for cid in sorted(changes.keys()):
            old_t, new_t = threshold_used[cid]
            n_changed = len(changes[cid])
            print(f"{cid:<40} {old_t:>10.3f} {new_t:>10.3f} {n_changed:>8}")

        print(f"\nLabel transitions:")
        transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        for cid, change_list in changes.items():
            for ch in change_list:
                transition_counts[(ch["old_label"], ch["new_label"])] += 1
        for (old, new), count in sorted(transition_counts.items()):
            print(f"  {old:<20} -> {new:<20} {count:>5}")
    else:
        print("\nNo label changes — output matches input.")

    print(f"\nWrote {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
