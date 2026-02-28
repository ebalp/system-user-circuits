"""Conflict base class and registry for system-user conflict dataset."""

from .conflict_base import Conflict
from .registry import get_all_conflicts, get_conflict, get_conflict_ids

__all__ = [
    "Conflict",
    "get_conflict",
    "get_all_conflicts",
    "get_conflict_ids",
]
