"""Tests for phase0_v2.src.metrics — directional baselines, new metrics, bug fix."""

import pytest

from phase0_v2.src.metrics import (
    DirectionalMetrics,
    MetricValue,
    MetricsCalculator,
    ModelMetrics,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic result records
# ---------------------------------------------------------------------------

def _rec(
    model: str = "test-model",
    condition: str = "C",
    label: str = "followed_system",
    direction: str = "a_to_b",
    constraint_type: str = "language",
    system_style: str = "bare",
    user_style: str = "with_instruction",
) -> dict:
    return dict(
        model=model,
        condition=condition,
        label=label,
        direction=direction,
        constraint_type=constraint_type,
        system_style=system_style,
        user_style=user_style,
    )


def _make_balanced_baseline_records(
    model: str = "test-model",
    condition: str = "A",
    label: str = "followed_system",
    n_per_dir: int = 10,
    success_rate_a: float = 1.0,
    success_rate_b: float = 1.0,
) -> list[dict]:
    """Make baseline records with given success rates per direction."""
    dir_key_a = "a_to_b"
    dir_key_b = "b_to_a"
    records = []
    for i in range(n_per_dir):
        lbl = label if i < int(n_per_dir * success_rate_a) else "followed_neither"
        records.append(_rec(model=model, condition=condition, label=lbl, direction=dir_key_a))
    for i in range(n_per_dir):
        lbl = label if i < int(n_per_dir * success_rate_b) else "followed_neither"
        records.append(_rec(model=model, condition=condition, label=lbl, direction=dir_key_b))
    return records


# ===========================================================================
# Directional SBR tests
# ===========================================================================

class TestDirectionalSBR:
    def test_balanced_sbr(self):
        """SBR with equal rates in both directions => balanced = that rate."""
        records = _make_balanced_baseline_records(
            condition="A", label="followed_system",
            n_per_dir=10, success_rate_a=0.8, success_rate_b=0.8,
        )
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert isinstance(m.sbr, DirectionalMetrics)
        assert m.sbr.a_to_b.value == pytest.approx(0.8, abs=0.01)
        assert m.sbr.b_to_a.value == pytest.approx(0.8, abs=0.01)
        assert m.sbr.balanced.value == pytest.approx(0.8, abs=0.01)
        assert m.sbr.asymmetry == pytest.approx(0.0, abs=0.01)

    def test_asymmetric_sbr(self):
        """SBR with different rates => asymmetry > 0."""
        records = _make_balanced_baseline_records(
            condition="A", label="followed_system",
            n_per_dir=10, success_rate_a=1.0, success_rate_b=0.5,
        )
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.sbr.a_to_b.value == pytest.approx(1.0, abs=0.01)
        assert m.sbr.b_to_a.value == pytest.approx(0.5, abs=0.01)
        assert m.sbr.balanced.value == pytest.approx(0.75, abs=0.01)
        assert m.sbr.asymmetry == pytest.approx(0.5, abs=0.01)

    def test_sbr_per_direction_values(self):
        """SBR a_to_b and b_to_a slots have correct n."""
        records = _make_balanced_baseline_records(
            condition="A", label="followed_system", n_per_dir=20,
        )
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.sbr.a_to_b.n == 20
        assert m.sbr.b_to_a.n == 20


# ===========================================================================
# Directional UCR tests
# ===========================================================================

class TestDirectionalUCR:
    def test_balanced_ucr(self):
        """UCR with equal rates in both directions."""
        records = _make_balanced_baseline_records(
            condition="B", label="followed_user",
            n_per_dir=10, success_rate_a=0.9, success_rate_b=0.9,
        )
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert isinstance(m.ucr, DirectionalMetrics)
        assert m.ucr.a_to_b.value == pytest.approx(0.9, abs=0.01)
        assert m.ucr.b_to_a.value == pytest.approx(0.9, abs=0.01)
        assert m.ucr.balanced.value == pytest.approx(0.9, abs=0.01)

    def test_asymmetric_ucr(self):
        """UCR with different rates across directions."""
        records = _make_balanced_baseline_records(
            condition="B", label="followed_user",
            n_per_dir=10, success_rate_a=1.0, success_rate_b=0.6,
        )
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.ucr.a_to_b.value == pytest.approx(1.0, abs=0.01)
        assert m.ucr.b_to_a.value == pytest.approx(0.6, abs=0.01)
        assert m.ucr.asymmetry == pytest.approx(0.4, abs=0.01)


# ===========================================================================
# c_bare_scr tests
# ===========================================================================

class TestCBareSCR:
    def test_filters_correctly(self):
        """c_bare_scr only includes bare+with_instruction records."""
        bare_records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="bare", user_style="with_instruction"),
            _rec(condition="C", label="followed_system", direction="b_to_a",
                 system_style="bare", user_style="with_instruction"),
        ]
        # These should be excluded
        other_records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="with_identity", user_style="with_instruction"),
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="bare", user_style="bare"),
        ]
        calc = MetricsCalculator(bare_records + other_records)
        m = calc.compute_for_model("test-model")

        assert m.c_bare_scr.a_to_b.n == 1
        assert m.c_bare_scr.b_to_a.n == 1
        assert m.c_bare_scr.balanced.value == pytest.approx(1.0)

    def test_empty_c_bare_scr(self):
        """c_bare_scr with no matching records => zeros."""
        records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="with_identity", user_style="with_instruction"),
        ]
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.c_bare_scr.a_to_b.n == 0
        assert m.c_bare_scr.b_to_a.n == 0


# ===========================================================================
# d_first_rate tests
# ===========================================================================

class TestDFirstRate:
    def test_d_first_rate_basic(self):
        """D first rate counts followed_system in condition D."""
        records = [
            _rec(condition="D", label="followed_system", direction="a_to_b"),
            _rec(condition="D", label="followed_system", direction="a_to_b"),
            _rec(condition="D", label="followed_user", direction="a_to_b"),
            _rec(condition="D", label="followed_system", direction="b_to_a"),
            _rec(condition="D", label="followed_user", direction="b_to_a"),
            _rec(condition="D", label="followed_user", direction="b_to_a"),
        ]
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.d_first_rate.a_to_b.value == pytest.approx(2 / 3, abs=0.01)
        assert m.d_first_rate.b_to_a.value == pytest.approx(1 / 3, abs=0.01)


# ===========================================================================
# system_authority_delta tests
# ===========================================================================

class TestSystemAuthorityDelta:
    def test_positive_delta(self):
        """Delta > 0 when c_bare_scr > d_first_rate."""
        # C bare: all followed_system
        c_records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="bare", user_style="with_instruction"),
            _rec(condition="C", label="followed_system", direction="b_to_a",
                 system_style="bare", user_style="with_instruction"),
        ]
        # D: none followed_system
        d_records = [
            _rec(condition="D", label="followed_user", direction="a_to_b"),
            _rec(condition="D", label="followed_user", direction="b_to_a"),
        ]
        calc = MetricsCalculator(c_records + d_records)
        m = calc.compute_for_model("test-model")

        assert m.system_authority_delta.a_to_b.value == pytest.approx(1.0)
        assert m.system_authority_delta.b_to_a.value == pytest.approx(1.0)
        assert m.system_authority_delta.balanced.value == pytest.approx(1.0)

    def test_zero_delta(self):
        """Delta = 0 when c_bare_scr == d_first_rate."""
        c_records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="bare", user_style="with_instruction"),
            _rec(condition="C", label="followed_system", direction="b_to_a",
                 system_style="bare", user_style="with_instruction"),
        ]
        d_records = [
            _rec(condition="D", label="followed_system", direction="a_to_b"),
            _rec(condition="D", label="followed_system", direction="b_to_a"),
        ]
        calc = MetricsCalculator(c_records + d_records)
        m = calc.compute_for_model("test-model")

        assert m.system_authority_delta.a_to_b.value == pytest.approx(0.0)
        assert m.system_authority_delta.b_to_a.value == pytest.approx(0.0)


# ===========================================================================
# Adjusted asymmetry bug fix verification
# ===========================================================================

class TestAdjustedAsymmetryBugFix:
    """Verify that adjusted asymmetry uses actual directional SBR values
    (a_to_b/b_to_a), not zero."""

    def test_adjusted_asymmetry_uses_real_sbr(self):
        """With asymmetric SBR, adjusted asymmetry should subtract baseline asymmetry."""
        # Condition A: a_to_b has 100% rate, b_to_a has 50% rate
        a_records = _make_balanced_baseline_records(
            condition="A", label="followed_system",
            n_per_dir=10, success_rate_a=1.0, success_rate_b=0.5,
        )
        # Condition C: a_to_b has 90%, b_to_a has 40%
        # Raw SCR asymmetry = |0.9 - 0.4| = 0.5
        # SBR asymmetry = |1.0 - 0.5| = 0.5
        # Adjusted = max(0, 0.5 - 0.5) = 0.0
        c_records = []
        for i in range(10):
            label = "followed_system" if i < 9 else "followed_user"
            c_records.append(_rec(condition="C", label=label, direction="a_to_b"))
        for i in range(10):
            label = "followed_system" if i < 4 else "followed_user"
            c_records.append(_rec(condition="C", label=label, direction="b_to_a"))

        calc = MetricsCalculator(a_records + c_records)
        m = calc.compute_for_model("test-model")

        # The bug was that sbr_by_direction always returned 0 for both directions,
        # so adjusted asymmetry equaled raw asymmetry. Now it should be 0.
        assert m.scr.adjusted_asymmetry == pytest.approx(0.0, abs=0.01)

    def test_adjusted_asymmetry_nonzero_when_beyond_baseline(self):
        """Adjusted asymmetry > 0 when SCR asymmetry exceeds SBR asymmetry."""
        # Condition A: symmetric baseline
        a_records = _make_balanced_baseline_records(
            condition="A", label="followed_system",
            n_per_dir=10, success_rate_a=0.8, success_rate_b=0.8,
        )
        # Condition C: highly asymmetric SCR
        # a_to_b: 100% => 10/10, b_to_a: 50% => 5/10
        # Raw asymmetry = 0.5, SBR asymmetry = 0.0
        # Adjusted = 0.5
        c_records = []
        for _ in range(10):
            c_records.append(_rec(condition="C", label="followed_system", direction="a_to_b"))
        for i in range(10):
            label = "followed_system" if i < 5 else "followed_user"
            c_records.append(_rec(condition="C", label=label, direction="b_to_a"))

        calc = MetricsCalculator(a_records + c_records)
        m = calc.compute_for_model("test-model")

        assert m.scr.adjusted_asymmetry == pytest.approx(0.5, abs=0.01)


# ===========================================================================
# Per-constraint breakdown tests
# ===========================================================================

class TestPerConstraintBreakdowns:
    def test_by_constraint_includes_directional_sbr_ucr(self):
        """Per-constraint breakdown should have directional SBR and UCR."""
        records = (
            _make_balanced_baseline_records(
                condition="A", label="followed_system",
                n_per_dir=5, success_rate_a=0.8, success_rate_b=0.6,
            )
            + _make_balanced_baseline_records(
                condition="B", label="followed_user",
                n_per_dir=5, success_rate_a=0.9, success_rate_b=0.7,
            )
        )
        # Add some C records so compute_for_model works
        records.append(_rec(condition="C", label="followed_system", direction="a_to_b"))
        records.append(_rec(condition="C", label="followed_user", direction="b_to_a"))

        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert "language" in m.by_constraint
        lang = m.by_constraint["language"]
        assert isinstance(lang.sbr, DirectionalMetrics)
        assert isinstance(lang.ucr, DirectionalMetrics)

    def test_by_constraint_has_c_bare_scr(self):
        """Per-constraint breakdown includes c_bare_scr."""
        records = [
            _rec(condition="C", label="followed_system", direction="a_to_b",
                 system_style="bare", user_style="with_instruction"),
            _rec(condition="C", label="followed_user", direction="b_to_a",
                 system_style="bare", user_style="with_instruction"),
        ]
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        lang = m.by_constraint["language"]
        assert isinstance(lang.c_bare_scr, DirectionalMetrics)
        assert lang.c_bare_scr.a_to_b.n == 1


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_results(self):
        """Empty results => all zeros, no errors."""
        calc = MetricsCalculator([])
        result = calc.compute_all()
        assert result == {}

    def test_single_direction_only(self):
        """Records with only one direction => balanced = that direction."""
        records = [
            _rec(condition="A", label="followed_system", direction="a_to_b"),
            _rec(condition="C", label="followed_system", direction="a_to_b"),
        ]
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert m.sbr.a_to_b.n == 1
        assert m.sbr.b_to_a.n == 0
        assert m.sbr.balanced.value == pytest.approx(1.0)

    def test_n_zero_for_condition(self):
        """Model with no records for a given condition => n=0 metrics."""
        records = [
            _rec(condition="C", label="followed_system", direction="a_to_b"),
        ]
        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        # No condition A records => SBR is zero
        assert m.sbr.a_to_b.n == 0
        assert m.sbr.b_to_a.n == 0
        assert m.sbr.balanced.value == 0.0

        # No condition B records => UCR is zero
        assert m.ucr.balanced.value == 0.0

    def test_capability_bias_warnings_sbr(self):
        """High SBR asymmetry triggers a warning."""
        records = _make_balanced_baseline_records(
            condition="A", label="followed_system",
            n_per_dir=10, success_rate_a=1.0, success_rate_b=0.5,
        )
        # Need some C records for compute_for_model
        records.append(_rec(condition="C", label="followed_system", direction="a_to_b"))
        records.append(_rec(condition="C", label="followed_system", direction="b_to_a"))

        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert any("SBR" in w for w in m.capability_bias_warnings)

    def test_capability_bias_warnings_ucr(self):
        """High UCR asymmetry triggers a warning."""
        records = _make_balanced_baseline_records(
            condition="B", label="followed_user",
            n_per_dir=10, success_rate_a=1.0, success_rate_b=0.5,
        )
        records.append(_rec(condition="C", label="followed_system", direction="a_to_b"))
        records.append(_rec(condition="C", label="followed_system", direction="b_to_a"))

        calc = MetricsCalculator(records)
        m = calc.compute_for_model("test-model")

        assert any("UCR" in w for w in m.capability_bias_warnings)
