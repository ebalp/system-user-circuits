"""Merged config loader: thresholds (from thresholds.yaml) + conflict metadata (from conflicts.yaml).

Threshold functions were originally in thresholds.py and are re-exported from there
for backward compatibility.
"""

import functools
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREPROCESSING_VALUES = frozenset({
    "extract_content",
    "refusal_prefix_only",
    "use_mention_stripping",
    "code_fence_unwrap",
    "markdown_prefix_stripping",
    "parenthetical_stripping",
})

ARCHITECTURE_VALUES = frozenset({
    "inverted_float_pair",
    "independent_float_pair",
    "complement_bool",
    "independent_bool",
})

# ---------------------------------------------------------------------------
# Section 1: Threshold functions (moved from thresholds.py)
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "thresholds.yaml"


@functools.cache
def _load_thresholds() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_threshold(conflict_id: str, model_id: str | None = None) -> float | None:
    """Return threshold for a specific model, falling back to default.

    Per-model entries can be either:
    - A plain float (simple override)
    - A dict with a 'threshold' key (rich metadata from optimizer)

    Falls back to default section if model not found or conflict not in model section.
    """
    data = _load_thresholds()
    if model_id:
        safe_id = model_id.replace("/", "_")
        model_section = data.get(safe_id, {})
        if not isinstance(model_section, dict):
            model_section = {}
        entry = model_section.get(conflict_id)
        if entry is not None:
            if isinstance(entry, dict):
                return entry.get("threshold")
            return entry
    default = data.get("default", {})
    return default.get(conflict_id)


def get_threshold_info(conflict_id: str, model_id: str) -> dict | None:
    """Return full threshold metadata for a conflict+model pair.

    Returns dict with keys: threshold, feasible, fallback, ba, d_norm, c_norm,
    distribution.  Returns None if no per-model entry exists (conflict may be
    bool or model section absent).
    """
    data = _load_thresholds()
    safe_id = model_id.replace("/", "_")
    model_section = data.get(safe_id, {})
    if not isinstance(model_section, dict):
        return None
    entry = model_section.get(conflict_id)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        return {"threshold": entry, "feasible": True, "fallback": None}
    return {
        "threshold": entry.get("threshold"),
        "feasible": entry.get("feasible", True),
        "fallback": entry.get("fallback"),
        "ba": entry.get("ba"),
        "d_norm": entry.get("d_norm"),
        "c_norm": entry.get("c_norm"),
        "distribution": entry.get("distribution"),
    }


def activate_model(model_id: str) -> None:
    """Set all conflict class thresholds for a specific model.

    Call once at experiment/rescore start for single-model pipelines.
    For multi-model analysis, use get_threshold(cid, model_id) directly.
    """
    from phase0_v2.conflicts.registry import get_all_conflicts

    for conflict in get_all_conflicts():
        t = get_threshold(conflict.conflict_id, model_id)
        if t is not None:
            type(conflict).verify_threshold = t


# ---------------------------------------------------------------------------
# Section 2: Conflict metadata functions (from conflicts.yaml)
# ---------------------------------------------------------------------------

_CONFLICTS_PATH = Path(__file__).parent / "conflicts.yaml"


@functools.cache
def _load_conflicts() -> dict:
    with open(_CONFLICTS_PATH) as f:
        return yaml.safe_load(f)


def get_conflict_meta(conflict_id: str) -> dict | None:
    """Return full metadata dict for a conflict, or None."""
    data = _load_conflicts()
    entry = data.get(conflict_id)
    if entry is None or not isinstance(entry, dict):
        return None
    return dict(entry)


def get_type(conflict_id: str) -> str | None:
    """Return 'bool' or 'float' for a conflict, or None."""
    meta = get_conflict_meta(conflict_id)
    return meta.get("type") if meta else None


def get_counterbalance_quality(conflict_id: str) -> str | None:
    meta = get_conflict_meta(conflict_id)
    return meta.get("counterbalance_quality") if meta else None


def get_preprocessing(conflict_id: str) -> list[str]:
    meta = get_conflict_meta(conflict_id)
    return list(meta.get("preprocessing", [])) if meta else []


def get_all_conflict_ids() -> list[str]:
    """Return sorted list of conflict IDs from conflicts.yaml."""
    data = _load_conflicts()
    return sorted(k for k in data if k != "_meta")


def validate_conflicts() -> list[str]:
    """Validate conflicts.yaml schema. Return list of error strings (empty = valid)."""
    data = _load_conflicts()
    errors = []
    required_fields = {"type", "constraint_a", "constraint_b", "architecture",
                       "verifier_logic", "counterbalance_quality", "exclusivity",
                       "arg_keys", "preprocessing"}
    valid_types = {"bool", "float"}
    valid_architectures = {"inverted_float_pair", "independent_float_pair",
                          "complement_bool", "independent_bool"}
    valid_cb = {"full", "partial", "none"}

    for cid, meta in data.items():
        if cid == "_meta":
            continue
        if not isinstance(meta, dict):
            errors.append(f"{cid}: entry is not a dict")
            continue
        missing = required_fields - set(meta.keys())
        if missing:
            errors.append(f"{cid}: missing fields {missing}")
        if meta.get("type") not in valid_types:
            errors.append(f"{cid}: invalid type '{meta.get('type')}'")
        if meta.get("architecture") not in valid_architectures:
            errors.append(f"{cid}: invalid architecture '{meta.get('architecture')}'")
        if meta.get("counterbalance_quality") not in valid_cb:
            errors.append(f"{cid}: invalid counterbalance_quality")
        if not isinstance(meta.get("arg_keys"), list):
            errors.append(f"{cid}: arg_keys must be a list")
        if not isinstance(meta.get("preprocessing"), list):
            errors.append(f"{cid}: preprocessing must be a list")
    return errors
