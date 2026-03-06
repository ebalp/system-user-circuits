"""Quick in-memory reverify + metrics for a single conflict.

Used to test verifier changes without modifying the results file.

Usage:
    uv run python -m phase0_v2.calibration.test_verifier \
      phase0_v2/data/results/meta-llama_Llama-3.1-8B-Instruct_results.jsonl \
      --conflict sentence_chaining \
      [--sample-mismatches 10]
"""

import argparse
import sys
from collections import defaultdict

from ._shared import load_records, direction_to_verify_code
from .rescore import _reverify_record
from .analyze import (
    _compute_baseline_rates,
    _compute_all_balanced_accuracy,
    _find_anomalies,
)
from ..conflicts.registry import get_conflict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test verifier for a single conflict.")
    parser.add_argument("results_file", help="Path to JSONL results file")
    parser.add_argument("--conflict", required=True, help="Conflict ID to test")
    parser.add_argument(
        "--sample-mismatches",
        type=int,
        default=0,
        help="Show N records where baseline label is unexpected",
    )
    args = parser.parse_args(argv)

    conflict = get_conflict(args.conflict)
    if conflict is None:
        print(f"Error: unknown conflict_id '{args.conflict}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading records from {args.results_file}...")
    all_records = load_records(args.results_file)

    # Filter to target conflict, skip errors
    records = [
        r for r in all_records
        if r.get("error") is None and r["conflict_id"] == args.conflict
    ]
    print(f"Found {len(records)} records for '{args.conflict}'")

    if not records:
        print("No records to process.")
        sys.exit(0)

    # Reverify in-memory
    reverified = []
    label_changes = 0
    for rec in records:
        direction_code = direction_to_verify_code(rec["direction"])
        updated = _reverify_record(rec, conflict, direction_code)
        if updated["label"] != rec["label"]:
            label_changes += 1
        reverified.append(updated)

    # Compute metrics on reverified records
    baseline_rates = _compute_baseline_rates(reverified, args.conflict)
    all_ba = _compute_all_balanced_accuracy(reverified, args.conflict)

    anomalies = _find_anomalies(reverified, args.conflict)
    anomaly_counts: dict[str, int] = defaultdict(int)
    for a in anomalies:
        anomaly_counts[a["anomaly_reason"]] += 1

    # Print compact summary
    print(f"\n{'=' * 60}")
    print(f"TEST VERIFIER: {args.conflict}")
    print(f"{'=' * 60}")

    ba = all_ba.get(args.conflict)
    print(f"  Balanced Accuracy:  {ba:.4f}" if ba is not None else "  Balanced Accuracy:  N/A")
    print(f"  Label changes:      {label_changes} / {len(records)}")

    if baseline_rates:
        row = baseline_rates[0]
        for key, label in [
            ("sbr_a", "SBR(a)"), ("ucr_a", "UCR(a)"),
            ("sbr_b", "SBR(b)"), ("ucr_b", "UCR(b)"),
        ]:
            val = row[key]
            n_key = f"n_{key}"
            n = row[n_key]
            if val is not None:
                print(f"  {label}:  {val:.4f}  (n={n})")
            else:
                print(f"  {label}:  N/A")

    if anomaly_counts:
        print(f"\n  Anomalies:")
        for reason, count in sorted(anomaly_counts.items()):
            print(f"    {reason}: {count}")
    else:
        print(f"\n  Anomalies: none")

    # Sample mismatches: baseline records with unexpected labels
    if args.sample_mismatches > 0:
        mismatches = []
        for rec in reverified:
            cond = rec["condition"]
            label = rec["label"]
            if cond == "A" and label != "followed_system":
                mismatches.append(rec)
            elif cond == "B" and label != "followed_user":
                mismatches.append(rec)

        print(f"\n  Baseline mismatches: {len(mismatches)} total")
        for rec in mismatches[: args.sample_mismatches]:
            response = rec.get("response", "")[:200]
            print(f"\n  --- cond={rec['condition']} dir={rec['direction']} label={rec['label']} ---")
            print(f"  sys_score={rec.get('verify_system_score'):.4f}  usr_score={rec.get('verify_user_score'):.4f}")
            print(f"  sys_ok={rec.get('verify_system_result')}  usr_ok={rec.get('verify_user_result')}")
            print(f"  response: {response!r}")

    print()


if __name__ == "__main__":
    main()
