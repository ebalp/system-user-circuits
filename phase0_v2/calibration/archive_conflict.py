"""Archive records for a given conflict_id from results JSONL files.

Moves matching records into an archived/ subdirectory and rewrites
the original files without them.
"""

import argparse
import json
from pathlib import Path

from ._shared import load_records


def archive_conflict(conflict_id: str, results_dir: Path) -> dict[str, int]:
    """Remove records matching conflict_id from all *_results.jsonl files.

    Returns a dict mapping filename -> number of archived records.
    """
    results_dir = Path(results_dir)
    archive_dir = results_dir / "archived"
    summary: dict[str, int] = {}

    for results_file in sorted(results_dir.glob("*_results.jsonl")):
        records = load_records(results_file)
        keep = [r for r in records if r.get("conflict_id") != conflict_id]
        archive = [r for r in records if r.get("conflict_id") == conflict_id]

        if not archive:
            continue

        # Check if these records were already archived (idempotent)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_file = archive_dir / f"{conflict_id}_from_{results_file.name}"

        # Write archived records (append if file exists to handle re-runs
        # on partially archived data, but avoid duplicates)
        existing_archived: set[str] = set()
        if archive_file.exists():
            for rec in load_records(archive_file):
                existing_archived.add(json.dumps(rec, sort_keys=True))

        new_archive = [
            r for r in archive
            if json.dumps(r, sort_keys=True) not in existing_archived
        ]

        if not new_archive and len(keep) == len(records):
            # All records already archived and source unchanged — true no-op
            continue

        if new_archive:
            with open(archive_file, "a") as f:
                for rec in new_archive:
                    f.write(json.dumps(rec) + "\n")

        # Rewrite original without archived records
        with open(results_file, "w") as f:
            for rec in keep:
                f.write(json.dumps(rec) + "\n")

        summary[results_file.name] = len(archive)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive records for a conflict_id from results JSONL files."
    )
    parser.add_argument("conflict_id", help="The conflict_id to archive")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing *_results.jsonl files",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error(f"Not a directory: {args.results_dir}")

    summary = archive_conflict(args.conflict_id, args.results_dir)

    if not summary:
        print(f"No records found for conflict_id '{args.conflict_id}' — nothing to archive.")
    else:
        total = 0
        for filename, count in summary.items():
            print(f"  {filename}: {count} records archived")
            total += count
        print(f"Total: {total} records archived to {args.results_dir / 'archived'}/")


if __name__ == "__main__":
    main()
