"""Tests for phase0_v2.calibration.smoke_test."""

import pytest

from phase0_v2.calibration.smoke_test import _compute_metrics, _select_tasks
from phase0_v2.src.config import Task


# ── _compute_metrics tests ──


def _make_record(condition, direction, label, expected_label, error=None):
    return {
        "condition": condition,
        "direction": direction,
        "label": label,
        "expected_label": expected_label,
        "error": error,
    }


class TestComputeMetrics:
    def test_perfect_baselines(self):
        """A/B conditions with 100% accuracy."""
        records = [
            _make_record("A", "a_to_b", "followed_system", "followed_system"),
            _make_record("A", "b_to_a", "followed_system", "followed_system"),
            _make_record("B", "a_to_b", "followed_user", "followed_user"),
            _make_record("B", "b_to_a", "followed_user", "followed_user"),
        ]
        m = _compute_metrics(records)
        assert m["sbr_a"] == 1.0
        assert m["sbr_b"] == 1.0
        assert m["ucr_a"] == 1.0
        assert m["ucr_b"] == 1.0
        assert m["baseline_ba"] == 1.0
        assert m["conditions"]["A"]["accuracy"] == 1.0
        assert m["conditions"]["B"]["accuracy"] == 1.0

    def test_ba_computation(self):
        """BA is average of per-direction accuracy for condition C."""
        records = [
            # a_to_b: 2/2 correct
            _make_record("C", "a_to_b", "followed_system", "followed_system"),
            _make_record("C", "a_to_b", "followed_system", "followed_system"),
            # b_to_a: 1/2 correct
            _make_record("C", "b_to_a", "followed_system", "followed_system"),
            _make_record("C", "b_to_a", "followed_user", "followed_system"),
        ]
        m = _compute_metrics(records)
        # BA = (1.0 + 0.5) / 2 = 0.75
        assert m["ba"] == pytest.approx(0.75)

    def test_errors_skipped(self):
        """Records with errors are counted but not scored."""
        records = [
            _make_record("A", "a_to_b", "error", "followed_system", error="timeout"),
            _make_record("A", "a_to_b", "followed_system", "followed_system"),
        ]
        m = _compute_metrics(records)
        assert m["errors"] == 1
        assert m["conditions"]["A"]["total"] == 1
        assert m["conditions"]["A"]["correct"] == 1

    def test_no_condition_c(self):
        """BA is None when no condition C records exist."""
        records = [
            _make_record("A", "a_to_b", "followed_system", "followed_system"),
        ]
        m = _compute_metrics(records)
        assert m["ba"] is None

    def test_empty_records(self):
        m = _compute_metrics([])
        assert m["total_records"] == 0
        assert m["errors"] == 0
        assert m["ba"] is None

    def test_label_distribution(self):
        """Label counts are reported per condition."""
        records = [
            _make_record("C", "a_to_b", "followed_system", "followed_system"),
            _make_record("C", "a_to_b", "followed_user", "followed_system"),
            _make_record("C", "a_to_b", "followed_both", "followed_system"),
        ]
        m = _compute_metrics(records)
        labels = m["conditions"]["C"]["labels"]
        assert labels["followed_system"] == 1
        assert labels["followed_user"] == 1
        assert labels["followed_both"] == 1


class TestSelectTasks:
    def _make_config_with_tasks(self, task_ids):
        """Create a minimal config-like object with tasks."""
        # We only need config.tasks for _select_tasks
        class FakeConfig:
            tasks = [Task(id=tid, prompt=f"Prompt for {tid}") for tid in task_ids]
        return FakeConfig()

    def test_default_selection(self):
        """Default picks the 4 representative tasks."""
        config = self._make_config_with_tasks([
            "photosynthesis", "vaccines", "used_car", "omelette",
            "renewable_energy", "french_revolution", "gravity",
        ])
        tasks = _select_tasks(config, n_tasks=None)
        ids = [t.id for t in tasks]
        assert ids == ["photosynthesis", "used_car", "renewable_energy", "french_revolution"]

    def test_n_tasks_override(self):
        """--n-tasks takes first N from full list."""
        config = self._make_config_with_tasks(["a", "b", "c", "d", "e"])
        tasks = _select_tasks(config, n_tasks=2)
        assert [t.id for t in tasks] == ["a", "b"]

    def test_fallback_when_defaults_missing(self):
        """Falls back to first 4 if default task IDs aren't in config."""
        config = self._make_config_with_tasks(["x1", "x2", "x3", "x4", "x5"])
        tasks = _select_tasks(config, n_tasks=None)
        assert len(tasks) == 4
        assert [t.id for t in tasks] == ["x1", "x2", "x3", "x4"]

    def test_empty_tasks_raises(self):
        config = self._make_config_with_tasks([])
        with pytest.raises(ValueError, match="No synthetic tasks"):
            _select_tasks(config, n_tasks=None)
