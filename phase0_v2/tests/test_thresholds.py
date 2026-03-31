"""Tests for threshold loading with nested YAML structure."""

from unittest.mock import patch

import pytest

from phase0_v2.config.thresholds import get_threshold, activate_model, _load_thresholds

# Sample nested data matching new YAML structure
_SAMPLE_DATA = {
    "_meta": {"last_updated": "2026-03-21"},
    "default": {
        "formal_vs_casual_tone": 0.988,
        "direct_answer_vs_hedging": 0.798,
        "address_reader_directly": 0.143,
    },
    "meta-llama_Llama-3.1-8B-Instruct": {
        "_meta": {"last_updated": "2026-03-21"},
        "formal_vs_casual_tone": {
            "threshold": 0.985,
            "baseline_range": [0.980, 0.995],
            "status": "OK",
        },
        "address_reader_directly": 0.150,
    },
}


@pytest.fixture(autouse=True)
def _mock_thresholds():
    """Patch _load_thresholds for all tests in this module."""
    _load_thresholds.cache_clear()
    with patch("phase0_v2.config.conflict_config._load_thresholds", return_value=_SAMPLE_DATA):
        yield
    _load_thresholds.cache_clear()


class TestGetThresholdDefaultOnly:
    """get_threshold with no model_id returns from default section."""

    def test_returns_default_value(self):
        assert get_threshold("formal_vs_casual_tone") == 0.988

    def test_returns_default_for_another_conflict(self):
        assert get_threshold("direct_answer_vs_hedging") == 0.798


class TestGetThresholdWithModel:
    """get_threshold with model_id returns model-specific threshold."""

    def test_dict_entry_extracts_threshold(self):
        result = get_threshold(
            "formal_vs_casual_tone", "meta-llama_Llama-3.1-8B-Instruct"
        )
        assert result == 0.985

    def test_plain_float_entry(self):
        result = get_threshold(
            "address_reader_directly", "meta-llama_Llama-3.1-8B-Instruct"
        )
        assert result == 0.150


class TestGetThresholdFallback:
    """get_threshold falls back to default when model doesn't have conflict."""

    def test_fallback_to_default(self):
        # direct_answer_vs_hedging not in model section → falls back
        result = get_threshold(
            "direct_answer_vs_hedging", "meta-llama_Llama-3.1-8B-Instruct"
        )
        assert result == 0.798

    def test_unknown_model_falls_back(self):
        result = get_threshold("formal_vs_casual_tone", "unknown_model")
        assert result == 0.988


class TestGetThresholdUnknown:
    """get_threshold returns None for unknown conflicts."""

    def test_unknown_conflict_no_model(self):
        assert get_threshold("nonexistent_conflict") is None

    def test_unknown_conflict_with_model(self):
        assert get_threshold(
            "nonexistent_conflict", "meta-llama_Llama-3.1-8B-Instruct"
        ) is None


class TestGetThresholdSlashInModelId:
    """Model IDs with slashes are converted to underscores."""

    def test_slash_converted(self):
        result = get_threshold(
            "formal_vs_casual_tone", "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert result == 0.985

    def test_slash_plain_float(self):
        result = get_threshold(
            "address_reader_directly", "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert result == 0.150


class TestActivateModel:
    """activate_model updates conflict class verify_threshold attributes."""

    def test_updates_thresholds(self):
        from phase0_v2.conflicts.registry import get_all_conflicts

        activate_model("meta-llama/Llama-3.1-8B-Instruct")

        conflicts_by_id = {c.conflict_id: c for c in get_all_conflicts()}

        # Conflicts in model section should use model threshold
        if "formal_vs_casual_tone" in conflicts_by_id:
            cls = type(conflicts_by_id["formal_vs_casual_tone"])
            assert cls.verify_threshold == 0.985

        # Conflicts NOT in model section should use default
        if "direct_answer_vs_hedging" in conflicts_by_id:
            cls = type(conflicts_by_id["direct_answer_vs_hedging"])
            assert cls.verify_threshold == 0.798
