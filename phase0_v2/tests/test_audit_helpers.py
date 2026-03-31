"""Tests for phase0_v2.calibration.audit_helpers utilities."""

import pytest

from phase0_v2.calibration.audit_helpers import (
    measure_baseline_metrics,
    reclassify_condition_c,
    load_conflict_audits,
)


def _make_record(
    conflict_id="test_conflict",
    condition="C",
    direction="a_to_b",
    response="hello world",
    label="followed_system",
    **extra,
):
    rec = {
        "conflict_id": conflict_id,
        "condition": condition,
        "direction": direction,
        "response": response,
        "label": label,
        "verify_system_result": True,
        "verify_user_result": False,
    }
    rec.update(extra)
    return rec


# ---- measure_baseline_metrics (bool mode) ----


class TestMeasureBaselineMetricsBool:
    """Test baseline metrics with bool verifiers."""

    def _build_baseline_records(self):
        """Build records where constraint_a = starts with 'A', constraint_b = starts with 'B'."""
        records = []
        # Condition A, direction=a_to_b: system says constraint_a
        # Model follows system → response starts with 'A' (verify_a=True, verify_b=False)
        for i in range(10):
            records.append(_make_record(
                condition="A", direction="a_to_b",
                response="A response here",
                label="followed_system",
            ))
        # Condition A, direction=b_to_a: system says constraint_b
        # Model follows system → response starts with 'B'
        for i in range(10):
            records.append(_make_record(
                condition="A", direction="b_to_a",
                response="B response here",
                label="followed_system",
            ))
        # Condition B, direction=a_to_b: user says constraint_b
        # Model follows user → response starts with 'B'
        for i in range(10):
            records.append(_make_record(
                condition="B", direction="a_to_b",
                response="B response here",
                label="followed_user",
            ))
        # Condition B, direction=b_to_a: user says constraint_a
        # Model follows user → response starts with 'A'
        for i in range(10):
            records.append(_make_record(
                condition="B", direction="b_to_a",
                response="A response here",
                label="followed_user",
            ))
        return records

    def test_perfect_baselines(self):
        """When verify functions match perfectly, all rates should be 1.0."""
        records = self._build_baseline_records()

        def verify_a(resp):
            return resp.startswith("A")

        def verify_b(resp):
            return resp.startswith("B")

        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b,
        )
        assert result["sbr_a"] == 1.0
        assert result["ucr_a"] == 1.0
        assert result["sbr_b"] == 1.0
        assert result["ucr_b"] == 1.0
        assert result["ba"] == 1.0
        assert result["n"] == 40

    def test_verify_a_broken(self):
        """When verify_a always returns False, SBR(a) and UCR(a) should drop."""
        records = self._build_baseline_records()

        def verify_a(resp):
            return False  # never detects constraint_a

        def verify_b(resp):
            return resp.startswith("B")

        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b,
        )
        # SBR(a): cond A, a_to_b — verify_a=False, verify_b=False → neither → fail
        assert result["sbr_a"] == 0.0
        # UCR(a): cond B, b_to_a — verify_a=False, verify_b=False → neither → fail
        assert result["ucr_a"] == 0.0
        # SBR(b): cond A, b_to_a — verify_a=False, verify_b=True → followed_system → pass
        assert result["sbr_b"] == 1.0
        # UCR(b): cond B, a_to_b — verify_a=False, verify_b=True → followed_user → pass
        assert result["ucr_b"] == 1.0
        assert result["ba"] == 0.5

    def test_both_fire_gives_followed_both(self):
        """When both verify functions fire, label is followed_both → SBR/UCR drop."""
        records = self._build_baseline_records()

        def verify_a(resp):
            return True  # always fires

        def verify_b(resp):
            return True  # always fires

        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b,
        )
        # Everything is followed_both → no followed_system or followed_user
        assert result["sbr_a"] == 0.0
        assert result["ucr_a"] == 0.0
        assert result["sbr_b"] == 0.0
        assert result["ucr_b"] == 0.0
        assert result["ba"] == 0.0

    def test_filters_by_conflict_id(self):
        """Records for other conflicts are ignored."""
        records = [
            _make_record(
                conflict_id="other_conflict",
                condition="A", direction="a_to_b",
                response="A response",
            ),
        ]
        result = measure_baseline_metrics(
            records, "test_conflict", lambda r: True, lambda r: False,
        )
        assert result["n"] == 0
        assert result["ba"] is None

    def test_skips_error_records(self):
        """Records with error field are skipped."""
        records = [
            _make_record(
                condition="A", direction="a_to_b",
                response="A response", error="timeout",
            ),
        ]
        result = measure_baseline_metrics(
            records, "test_conflict", lambda r: True, lambda r: False,
        )
        assert result["n"] == 0

    def test_skips_condition_c(self):
        """Only conditions A and B are processed."""
        records = [
            _make_record(
                condition="C", direction="a_to_b",
                response="A response",
            ),
        ]
        result = measure_baseline_metrics(
            records, "test_conflict", lambda r: True, lambda r: False,
        )
        assert result["n"] == 0


# ---- measure_baseline_metrics (float mode) ----


class TestMeasureBaselineMetricsFloat:
    """Test baseline metrics with float verifiers and threshold."""

    def test_float_thresholds(self):
        """Float scores classified via asymmetric thresholds."""
        records = []
        # Cond A, a_to_b: system says constraint_a
        # Model follows → verify_a scores high, verify_b scores low
        for i in range(10):
            records.append(_make_record(
                condition="A", direction="a_to_b",
                response="high_a",
            ))
        # Cond B, b_to_a: user says constraint_a
        for i in range(10):
            records.append(_make_record(
                condition="B", direction="b_to_a",
                response="high_a",
            ))

        def verify_a(resp):
            return 0.8  # high score

        def verify_b(resp):
            return 0.1  # low score → 1-0.1 = 0.9, NOT > 1-T

        # threshold=0.5: a_pass = 0.8 >= 0.5 = True, b_pass = 0.1 > 0.5 = False
        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b, threshold=0.5,
        )
        assert result["sbr_a"] == 1.0
        assert result["ucr_a"] == 1.0

    def test_float_near_threshold(self):
        """Score exactly at threshold boundary."""
        records = [
            _make_record(condition="A", direction="a_to_b", response="x"),
        ]

        def verify_a(resp):
            return 0.5  # exactly at threshold

        def verify_b(resp):
            return 0.1

        # a_pass = 0.5 >= 0.5 = True (direct side uses >=)
        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b, threshold=0.5,
        )
        assert result["sbr_a"] == 1.0

    def test_float_inverted_boundary(self):
        """Inverted side uses > (strict), not >=."""
        records = [
            _make_record(condition="A", direction="a_to_b", response="x"),
        ]

        def verify_a(resp):
            return 0.1

        def verify_b(resp):
            return 0.5  # b_pass = 0.5 > (1 - 0.5) = 0.5 > 0.5 → False (strict >)

        result = measure_baseline_metrics(
            records, "test_conflict", verify_a, verify_b, threshold=0.5,
        )
        # a_pass=False, b_pass=False → followed_neither → SBR(a)=0
        assert result["sbr_a"] == 0.0


# ---- load_conflict_audits ----


class TestLoadConflictAudits:
    """Test loading audit JSONs across models."""

    def test_returns_empty_for_nonexistent(self):
        result = load_conflict_audits("nonexistent_conflict_xyz_123")
        assert result == {}

    def test_loads_existing_conflict(self):
        """Smoke test: loads disclaimer_first_vs_none if audit data exists."""
        result = load_conflict_audits("disclaimer_first_vs_none")
        if not result:
            pytest.skip("No audit data available for disclaimer_first_vs_none")
        for model_label, data in result.items():
            assert "json" in data
            assert "report_paths" in data
            assert isinstance(data["json"], dict)
            assert isinstance(data["report_paths"], list)
            # JSON should have conflict_id
            assert data["json"].get("conflict_id") == "disclaimer_first_vs_none"

    def test_model_filter(self):
        """Filtering to specific models works."""
        all_results = load_conflict_audits("disclaimer_first_vs_none")
        if len(all_results) < 2:
            pytest.skip("Need at least 2 models with audit data")
        first_model = list(all_results.keys())[0]
        filtered = load_conflict_audits(
            "disclaimer_first_vs_none", model_labels=[first_model],
        )
        assert len(filtered) == 1
        assert first_model in filtered
