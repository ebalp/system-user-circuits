"""Backward-compatible re-exports from conflict_config.

All threshold functions now live in conflict_config.py.
This shim ensures existing imports continue to work.
"""

from phase0_v2.config.conflict_config import (  # noqa: F401
    _load_thresholds,
    get_threshold,
    get_threshold_info,
    activate_model,
)
