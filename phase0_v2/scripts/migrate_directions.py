"""Migrate JSONL direction field: baseline_a -> a_to_b, baseline_b -> b_to_a.

Also recomputes prompt_id and experiment_hash to reflect the new direction strings,
and deduplicates records (keeping the first occurrence).

Usage:
    uv run python phase0_v2/scripts/migrate_directions.py [--dry-run]
"""

import json
import shutil
import sys
from pathlib import Path

from phase0_v2.src.experiment import ExperimentKey, compute_experiment_hash

RESULTS_DIR = Path("phase0_v2/data/results")
BACKUP_DIR = RESULTS_DIR / "backup"

DIRECTION_MAP = {
    "baseline_a": "a_to_b",
    "baseline_b": "b_to_a",
}


def _recompute_prompt_id(rec: dict) -> str:
    """Recompute prompt_id using the same logic as PromptGenerator._make_id."""
    conflict_id = rec.get("conflict_id", "")
    direction = rec["direction"]
    task_id = rec.get("task_id", "")
    system_style = rec.get("system_style") or "none"
    user_style = rec.get("user_style", "")
    style_short = user_style[:3]
    parts = [conflict_id, direction, task_id, system_style, style_short, "000"]
    return "_".join(parts)


def _recompute_experiment_hash(rec: dict, temperature: float, max_tokens: int) -> str:
    """Recompute experiment_hash using the real compute_experiment_hash function."""
    key = ExperimentKey(
        model=rec.get("model", ""),
        conflict_id=rec.get("conflict_id", ""),
        instruction_args_json=json.dumps(
            rec.get("instruction_args", {}), sort_keys=True, separators=(",", ":")
        ),
        task_id=rec.get("task_id", ""),
        task_source=rec.get("task_source", ""),
        condition=rec.get("condition", ""),
        direction=rec["direction"],
        system_style=rec.get("system_style"),
        user_style=rec.get("user_style", ""),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return compute_experiment_hash(key)


def migrate_file(
    path: Path, temperature: float, max_tokens: int, dry_run: bool = False
) -> tuple[int, int, int]:
    """Migrate a single JSONL file. Returns (total, direction_changes, duplicates_removed)."""
    records = []
    dir_changed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Migrate directions and recompute IDs for ALL records
    for rec in records:
        old_dir = rec.get("direction", "")
        if old_dir in DIRECTION_MAP:
            rec["direction"] = DIRECTION_MAP[old_dir]
            dir_changed += 1
        # Store config values and recompute prompt_id and hash
        rec["temperature"] = temperature
        rec["max_tokens"] = max_tokens
        rec["prompt_id"] = _recompute_prompt_id(rec)
        rec["experiment_hash"] = _recompute_experiment_hash(rec, temperature, max_tokens)

    # Deduplicate by experiment_hash (keep first occurrence)
    seen_hashes = set()
    deduped = []
    dupes = 0
    for rec in records:
        h = rec.get("experiment_hash", "")
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)
        deduped.append(rec)

    if not dry_run:
        with open(path, "w") as f:
            for rec in deduped:
                f.write(json.dumps(rec) + "\n")

    return len(records), dir_changed, dupes


def main():
    dry_run = "--dry-run" in sys.argv

    # Load generation config for correct hash computation
    import yaml
    config_path = Path("phase0_v2/config/experiment.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    temperature = config["generation"]["temperature"]
    max_tokens = config["generation"]["max_tokens"]
    print(f"Using config: temperature={temperature}, max_tokens={max_tokens}")

    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    if not dry_run:
        # Backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Backing up {len(jsonl_files)} files to {BACKUP_DIR}/")
        for path in jsonl_files:
            shutil.copy2(path, BACKUP_DIR / path.name)

    # Migrate
    action = "Would migrate" if dry_run else "Migrating"
    print(f"\n{action} direction strings and recomputing hashes...")
    total_changed = 0
    total_dupes = 0
    for path in jsonl_files:
        n_total, n_changed, n_dupes = migrate_file(path, temperature, max_tokens, dry_run)
        total_changed += n_changed
        total_dupes += n_dupes
        print(
            f"  {path.name}: {n_total} records, "
            f"{n_changed} directions updated, {n_dupes} duplicates removed"
        )

    print(f"\nDone. {total_changed} directions updated, {total_dupes} duplicates removed.")
    if not dry_run:
        print(f"Backups in {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
