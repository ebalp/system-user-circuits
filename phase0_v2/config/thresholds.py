"""Load float conflict thresholds from thresholds.yaml."""

import functools
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "thresholds.yaml"


@functools.cache
def _load_thresholds() -> dict[str, float]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_threshold(conflict_id: str) -> float | None:
    """Return threshold for a float conflict, or None if not configured."""
    return _load_thresholds().get(conflict_id)
