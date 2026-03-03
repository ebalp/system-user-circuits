"""Shared utilities for calibration analysis and rescoring."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..conflicts.registry import get_all_conflicts


def load_records(path: str | Path) -> list[dict]:
    """Read a JSONL file line-by-line, return list of parsed dicts. Skip blank lines."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@dataclass
class SideInfo:
    is_inverted: bool


@dataclass
class ConflictThresholdInfo:
    threshold: float
    # sides[direction_code][side] -> SideInfo
    # e.g., sides["a"]["system"].is_inverted
    sides: dict[str, dict[str, SideInfo]] = field(default_factory=dict)


def build_conflict_threshold_map() -> dict[str, ConflictThresholdInfo]:
    """Build per-conflict threshold and inversion metadata from the registry."""
    result = {}
    for conflict in get_all_conflicts():
        info = ConflictThresholdInfo(threshold=conflict.verify_threshold)

        # Direction "a"
        info.sides["a"] = {
            "system": SideInfo(
                is_inverted=getattr(conflict._verify_system_fn, "is_inverted", False)
            ),
            "user": SideInfo(
                is_inverted=getattr(conflict._verify_user_fn, "is_inverted", False)
            ),
        }

        # Direction "b" (only if counterbalancing supported)
        if conflict.supports_counterbalancing():
            info.sides["b"] = {
                "system": SideInfo(
                    is_inverted=getattr(
                        conflict._verify_inverse_system_fn, "is_inverted", False
                    )
                ),
                "user": SideInfo(
                    is_inverted=getattr(
                        conflict._verify_inverse_user_fn, "is_inverted", False
                    )
                ),
            }

        result[conflict.conflict_id] = info
    return result


def direction_to_verify_code(direction: str) -> str:
    """Map JSONL direction strings to verify codes.

    'a_to_b' -> 'a'
    'b_to_a' -> 'b'
    'none' -> 'a'
    """
    return {
        "a_to_b": "a",
        "b_to_a": "b",
        "none": "a",
    }.get(direction, "a")


def apply_threshold(score: float, threshold: float, is_inverted: bool) -> bool:
    """Replicate the asymmetric threshold logic from conflict_base._dispatch_verify.

    Direct (is_inverted=False): score >= threshold
    Inverted (is_inverted=True): score > (1.0 - threshold)
    """
    if is_inverted:
        return score > (1.0 - threshold)
    else:
        return score >= threshold


def compute_label(sys_result: bool, usr_result: bool) -> str:
    """Return classification label from boolean verify results."""
    if sys_result and not usr_result:
        return "followed_system"
    elif usr_result and not sys_result:
        return "followed_user"
    elif sys_result and usr_result:
        return "followed_both"
    else:
        return "followed_neither"
