"""Validation tests for conflicts.yaml: schema, sync with Python classes, cross-file consistency."""

import pytest

from phase0_v2.config.conflict_config import (
    ARCHITECTURE_VALUES,
    PREPROCESSING_VALUES,
    get_all_conflict_ids,
    get_conflict_meta,
    get_threshold,
    get_type,
    validate_conflicts,
)
from phase0_v2.conflicts.registry import get_all_conflicts, get_conflict_ids


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


class TestConflictsYamlSchema:
    """Verify conflicts.yaml structure and valid enum values."""

    def test_validate_returns_no_errors(self):
        """validate_conflicts() should return empty list."""
        errors = validate_conflicts()
        assert errors == [], f"Validation errors: {errors}"

    def test_count_matches_41(self):
        """Exactly 41 conflict entries (excluding _meta)."""
        ids = get_all_conflict_ids()
        assert len(ids) == 41, f"Expected 41 conflicts, got {len(ids)}: {ids}"

    def test_all_required_fields_present(self):
        """Every entry has all 9 required fields."""
        required = {
            "type", "constraint_a", "constraint_b", "architecture",
            "verifier_logic", "counterbalance_quality", "exclusivity",
            "arg_keys", "preprocessing",
        }
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            missing = required - set(meta.keys())
            assert not missing, f"{cid} missing: {missing}"

    def test_type_values(self):
        """Every conflict has type 'bool' or 'float'."""
        for cid in get_all_conflict_ids():
            assert get_type(cid) in ("bool", "float"), f"{cid}: invalid type"

    def test_architecture_values(self):
        """Every conflict has a recognized architecture."""
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            assert meta["architecture"] in ARCHITECTURE_VALUES, (
                f"{cid}: unknown architecture '{meta['architecture']}'"
            )

    def test_preprocessing_values(self):
        """Every preprocessing step is from the known set."""
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            for p in meta["preprocessing"]:
                assert p in PREPROCESSING_VALUES, (
                    f"{cid}: unknown preprocessing '{p}'"
                )

    def test_counterbalance_quality_values(self):
        """Every conflict has counterbalance_quality in {full, partial, none}."""
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            assert meta["counterbalance_quality"] in ("full", "partial", "none"), (
                f"{cid}: invalid counterbalance_quality '{meta['counterbalance_quality']}'"
            )

    def test_arg_keys_is_list(self):
        """arg_keys must be a list for every conflict."""
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            assert isinstance(meta["arg_keys"], list), (
                f"{cid}: arg_keys is {type(meta['arg_keys']).__name__}, not list"
            )

    def test_preprocessing_is_list(self):
        """preprocessing must be a list for every conflict."""
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            assert isinstance(meta["preprocessing"], list), (
                f"{cid}: preprocessing is {type(meta['preprocessing']).__name__}, not list"
            )

    def test_bool_architecture_consistency(self):
        """Bool conflicts must use complement_bool or independent_bool."""
        bool_archs = {"complement_bool", "independent_bool"}
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            if meta["type"] == "bool":
                assert meta["architecture"] in bool_archs, (
                    f"{cid}: bool conflict has float architecture '{meta['architecture']}'"
                )

    def test_float_architecture_consistency(self):
        """Float conflicts must use inverted_float_pair or independent_float_pair."""
        float_archs = {"inverted_float_pair", "independent_float_pair"}
        for cid in get_all_conflict_ids():
            meta = get_conflict_meta(cid)
            if meta["type"] == "float":
                assert meta["architecture"] in float_archs, (
                    f"{cid}: float conflict has bool architecture '{meta['architecture']}'"
                )


# ---------------------------------------------------------------------------
# Sync checks (YAML matches Python code)
# ---------------------------------------------------------------------------


class TestConflictsYamlSync:
    """Verify YAML metadata stays in sync with Python conflict classes."""

    def test_yaml_ids_match_registry(self):
        """Every registered conflict has a YAML entry and vice versa."""
        yaml_ids = set(get_all_conflict_ids())
        registry_ids = set(get_conflict_ids())
        only_yaml = yaml_ids - registry_ids
        only_registry = registry_ids - yaml_ids
        assert yaml_ids == registry_ids, (
            f"Only in YAML: {only_yaml or 'none'}, "
            f"only in registry: {only_registry or 'none'}"
        )

    def test_arg_keys_match_class(self):
        """arg_keys in YAML matches class attribute."""
        for conflict in get_all_conflicts():
            cid = conflict.conflict_id
            meta = get_conflict_meta(cid)
            yaml_keys = sorted(meta["arg_keys"])
            class_keys = sorted(conflict.get_instruction_args_keys())
            assert yaml_keys == class_keys, (
                f"{cid}: YAML {yaml_keys} != class {class_keys}"
            )

    def test_counterbalance_quality_matches_class(self):
        """counterbalance_quality in YAML matches class attribute."""
        for conflict in get_all_conflicts():
            cid = conflict.conflict_id
            meta = get_conflict_meta(cid)
            assert meta["counterbalance_quality"] == conflict.counterbalance_quality, (
                f"{cid}: YAML '{meta['counterbalance_quality']}' "
                f"!= class '{conflict.counterbalance_quality}'"
            )

    def test_architecture_consistent_with_is_inverted(self):
        """Float conflicts: is_inverted on any verify fn implies inverted_float_pair."""
        for conflict in get_all_conflicts():
            cid = conflict.conflict_id
            meta = get_conflict_meta(cid)
            if meta["type"] != "float":
                continue
            fns = [conflict._verify_system_fn, conflict._verify_user_fn]
            if conflict._verify_inverse_system_fn:
                fns.append(conflict._verify_inverse_system_fn)
            if conflict._verify_inverse_user_fn:
                fns.append(conflict._verify_inverse_user_fn)
            has_inverted = any(getattr(fn, "is_inverted", False) for fn in fns)
            if has_inverted:
                assert meta["architecture"] == "inverted_float_pair", (
                    f"{cid}: has is_inverted but architecture is "
                    f"'{meta['architecture']}'"
                )
            else:
                assert meta["architecture"] == "independent_float_pair", (
                    f"{cid}: no is_inverted but architecture is "
                    f"'{meta['architecture']}'"
                )


# ---------------------------------------------------------------------------
# Cross-file checks (conflicts.yaml vs thresholds.yaml)
# ---------------------------------------------------------------------------


class TestConflictsYamlCrossFile:
    """Check consistency between conflicts.yaml and thresholds.yaml."""

    def test_float_conflicts_have_default_threshold(self):
        """Every type:float conflict has a threshold in thresholds.yaml default section."""
        missing = []
        for cid in get_all_conflict_ids():
            if get_type(cid) == "float":
                t = get_threshold(cid)
                if t is None:
                    missing.append(cid)
        assert not missing, (
            f"Float conflicts without default threshold: {missing}"
        )

    def test_bool_conflicts_have_no_threshold(self):
        """Bool conflicts should not have entries in thresholds.yaml default section."""
        unexpected = []
        for cid in get_all_conflict_ids():
            if get_type(cid) == "bool":
                t = get_threshold(cid)
                if t is not None:
                    unexpected.append(cid)
        assert not unexpected, (
            f"Bool conflicts with unexpected threshold: {unexpected}"
        )
