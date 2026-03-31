"""Cross-model canonical ordering for the single-model behavioral report.

Computes conflict, system style, and user style orderings based on average
SCR across selected models. This ensures all per-model reports use the same
ordering for visual comparability.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Configurable: which models to include in the cross-model average.
# Edit this list when models are added/removed from the experiment.
ORDERING_MODELS: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-3-27b-it",
    "openai/gpt-oss-20b",
    "Qwen/Qwen2.5-7B-Instruct",
]


def compute_canonical_order(
    results_dir: str,
    model_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Compute cross-model canonical ordering.

    Loads condition C records from each model's JSONL file, computes per-conflict
    and per-style SCR, averages across models (using however many models have data
    for each item), and returns the three ordered lists.

    Args:
        results_dir: Directory containing JSONL result files.
        model_ids: Models to include. Defaults to ORDERING_MODELS.

    Returns:
        (conflict_order, system_style_order, user_style_order), each sorted
        by average SCR descending.
    """
    if model_ids is None:
        model_ids = ORDERING_MODELS

    # Accumulators: item -> list of per-model SCR values
    conflict_scrs: dict[str, list[float]] = defaultdict(list)
    sys_style_scrs: dict[str, list[float]] = defaultdict(list)
    usr_style_scrs: dict[str, list[float]] = defaultdict(list)

    for model_id in model_ids:
        safe_id = model_id.replace("/", "_")
        path = Path(results_dir) / f"{safe_id}_results.jsonl"
        if not path.exists():
            log.warning("Results file not found for %s: %s", model_id, path)
            continue

        # Load condition C records
        cond_c: list[dict] = []
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("condition") == "C":
                cond_c.append(rec)

        if not cond_c:
            log.warning("No condition C records for %s", model_id)
            continue

        # Per-conflict SCR
        by_conflict: dict[str, list[dict]] = defaultdict(list)
        for r in cond_c:
            by_conflict[r.get("constraint_type", "unknown")].append(r)
        for conflict_id, recs in by_conflict.items():
            scr = sum(1 for r in recs if r.get("label") == "followed_system") / len(recs)
            conflict_scrs[conflict_id].append(scr)

        # Per-system-style SCR
        by_sys: dict[str, list[dict]] = defaultdict(list)
        for r in cond_c:
            by_sys[r.get("system_style", "unknown")].append(r)
        for style, recs in by_sys.items():
            scr = sum(1 for r in recs if r.get("label") == "followed_system") / len(recs)
            sys_style_scrs[style].append(scr)

        # Per-user-style SCR
        by_usr: dict[str, list[dict]] = defaultdict(list)
        for r in cond_c:
            by_usr[r.get("user_style", r.get("user_template", "unknown"))].append(r)
        for style, recs in by_usr.items():
            scr = sum(1 for r in recs if r.get("label") == "followed_system") / len(recs)
            usr_style_scrs[style].append(scr)

    # Average across models and sort descending
    def _avg_sort(scrs: dict[str, list[float]]) -> list[str]:
        averaged = {k: sum(v) / len(v) for k, v in scrs.items()}
        return sorted(averaged, key=lambda k: (-averaged[k], k))

    return _avg_sort(conflict_scrs), _avg_sort(sys_style_scrs), _avg_sort(usr_style_scrs)


def ordered(items: list[str], canonical: list[str]) -> list[str]:
    """Sort items by canonical order. Unknown items go at the end alphabetically.

    Only includes items that are in the ``items`` list (filters canonical to
    present items).

    Args:
        items: Items to sort (e.g., conflict IDs actually present in the data).
        canonical: The canonical ordering to apply.

    Returns:
        Items sorted by their position in canonical, with unknowns appended
        alphabetically.
    """
    item_set = set(items)
    known = [c for c in canonical if c in item_set]
    known_set = set(known)
    unknown = sorted(i for i in items if i not in known_set)
    return known + unknown
