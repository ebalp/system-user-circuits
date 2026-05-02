"""Tests for phase0_v2.calibration.audit_helpers utilities."""

import json

import pytest
import yaml

from phase0_v2.calibration.audit_helpers import (
    apply_audit_recommendation,
    measure_baseline_metrics,
    reclassify_condition_c,
    load_conflict_audits,
    revert_audit_recommendation,
    sample_by_score_band,
)
from phase0_v2.calibration.per_model_thresholds import is_ambiguous


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


# ---- sample_by_score_band ----


def _band_record(score, direction="a_to_b", condition="C", conflict_id="test"):
    return {
        "conflict_id": conflict_id,
        "condition": condition,
        "direction": direction,
        "response": f"resp@{score}",
        "verify_system_score": score,
        "error": None,
    }


class TestSampleByScoreBand:
    def test_uniform_bands_assign_correctly(self):
        # Scores at 0.05, 0.15, 0.25, ..., 0.95 — one per default 8-band split.
        # n_bands=4 → bands [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0].
        records = [_band_record(s) for s in [0.05, 0.20, 0.30, 0.55, 0.80, 0.99]]
        out = sample_by_score_band(records, "test", n_bands=4, samples_per_band=10)
        assert len(out) == 4
        scores_per_band = {k: sorted(it["score"] for it in v) for k, v in out.items()}
        assert scores_per_band[(0.0, 0.25)] == [0.05, 0.20]
        assert scores_per_band[(0.25, 0.5)] == [0.30]
        assert scores_per_band[(0.5, 0.75)] == [0.55]
        assert scores_per_band[(0.75, 1.0)] == [0.80, 0.99]

    def test_score_1_0_falls_in_last_band(self):
        records = [_band_record(1.0)]
        out = sample_by_score_band(records, "test", n_bands=4, samples_per_band=10)
        assert len(out[(0.75, 1.0)]) == 1

    def test_b_to_a_direction_normalizes_score(self):
        # b_to_a: raw score is on constraint_b scale → invert to 1 - s.
        # Raw 0.1 → normalized 0.9 → lands in last band.
        records = [_band_record(0.1, direction="b_to_a")]
        out = sample_by_score_band(records, "test", n_bands=4, samples_per_band=10)
        assert out[(0.0, 0.25)] == []
        assert len(out[(0.75, 1.0)]) == 1
        assert out[(0.75, 1.0)][0]["score"] == pytest.approx(0.9)

    def test_subsamples_when_band_overflows(self):
        records = [_band_record(0.10) for _ in range(20)]
        out = sample_by_score_band(records, "test", n_bands=4, samples_per_band=3, seed=7)
        assert len(out[(0.0, 0.25)]) == 3
        # Deterministic with same seed
        out2 = sample_by_score_band(records, "test", n_bands=4, samples_per_band=3, seed=7)
        assert [it["score"] for it in out[(0.0, 0.25)]] == [
            it["score"] for it in out2[(0.0, 0.25)]
        ]

    def test_drill_down_with_explicit_bands(self):
        records = [_band_record(s) for s in [0.40, 0.42, 0.44, 0.46, 0.48, 0.55]]
        out = sample_by_score_band(
            records, "test",
            bands=[(0.40, 0.45), (0.45, 0.50)],
            samples_per_band=10,
        )
        assert len(out) == 2
        assert sorted(it["score"] for it in out[(0.40, 0.45)]) == [0.40, 0.42, 0.44]
        assert sorted(it["score"] for it in out[(0.45, 0.50)]) == [0.46, 0.48]

    def test_filters_error_records_and_other_conflicts(self):
        records = [
            _band_record(0.5),
            {**_band_record(0.5), "error": "x"},
            {**_band_record(0.5), "conflict_id": "other"},
            {**_band_record(0.5), "condition": "A"},
        ]
        out = sample_by_score_band(records, "test", n_bands=4, samples_per_band=10)
        assert sum(len(v) for v in out.values()) == 1


# ---- is_ambiguous ----


class TestIsAmbiguous:
    def test_infeasible_is_ambiguous(self):
        r = {"feasible": False, "d_norm": 0.0, "c_norm": 0.0, "ba": 1.0}
        assert is_ambiguous(r, max_ba=1.0) is True

    def test_high_d_norm_triggers(self):
        r = {"feasible": True, "d_norm": 0.02, "c_norm": 0.001, "ba": 1.0}
        assert is_ambiguous(r, max_ba=1.0) is True

    def test_high_c_norm_triggers(self):
        r = {"feasible": True, "d_norm": 0.001, "c_norm": 0.05, "ba": 1.0}
        assert is_ambiguous(r, max_ba=1.0) is True

    def test_ba_below_max_triggers(self):
        r = {"feasible": True, "d_norm": 0.0, "c_norm": 0.0, "ba": 0.99}
        assert is_ambiguous(r, max_ba=1.0) is True

    def test_clean_pick_not_ambiguous(self):
        r = {"feasible": True, "d_norm": 0.005, "c_norm": 0.005, "ba": 1.0}
        assert is_ambiguous(r, max_ba=1.0) is False

    def test_missing_max_ba_does_not_trigger_ba_rule(self):
        r = {"feasible": True, "d_norm": 0.005, "c_norm": 0.005, "ba": 0.97}
        assert is_ambiguous(r, max_ba=None) is False


# ---- apply_audit_recommendation / revert_audit_recommendation ----


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _audit_json_payload(
    *, conflict_id="vowel_omission", model="google_gemma-4-E2B-it",
    T_pareto=0.978, T_recommended=0.75, confidence="high",
):
    return {
        "conflict_id": conflict_id,
        "model": model,
        "timestamp": "2026-05-01T18:11:00Z",
        "verifier": {"type": "float", "threshold": T_pareto},
        "pareto": {
            "threshold": T_pareto, "ba": 0.965, "max_ba": 1.0,
            "d_norm": 0.000238, "c_norm": 0.012116,
            "distribution": "bimodal", "feasible": True,
            "ambiguous": True,
        },
        "semantic_threshold": {
            "ran": True,
            "trigger_reason": "ambiguous=true",
            "T_pareto": T_pareto,
            "T_recommended": T_recommended,
            "recommended_agent_ba": 1.0,
            "recommendation_confidence": confidence,
            "rationale": "test fixture",
            "delta_vs_pareto": T_recommended - T_pareto,
        },
    }


class TestApplyAuditRecommendation:
    """End-to-end apply/revert without touching the real thresholds.yaml."""

    @pytest.fixture
    def yaml_setup(self, tmp_path):
        """Write a minimal thresholds.yaml with one model section + one conflict."""
        yaml_path = tmp_path / "thresholds.yaml"
        _write_yaml(yaml_path, {
            "default": {"vowel_omission": 0.5},
            "google_gemma-4-E2B-it": {
                "_meta": {"last_updated": "2026-05-01"},
                "vowel_omission": {
                    "threshold": 0.978,
                    "source": "pareto",
                    "ba": 0.965,
                    "max_ba": 1.0,
                    "d_norm": 0.000238,
                    "c_norm": 0.012116,
                    "distribution": "bimodal",
                    "feasible": True,
                    "ambiguous": True,
                },
            },
        })
        return yaml_path

    def test_skip_when_not_ran(self, yaml_setup, tmp_path):
        audit = _audit_json_payload()
        audit["semantic_threshold"]["ran"] = False
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)
        result = apply_audit_recommendation(
            json_path, yaml_path=yaml_setup, rescore=False,
        )
        assert result["applied"] is False
        assert "ran is false" in result["reason"]

    def test_skip_when_T_unchanged(self, yaml_setup, tmp_path):
        audit = _audit_json_payload(T_pareto=0.978, T_recommended=0.978)
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)
        result = apply_audit_recommendation(
            json_path, yaml_path=yaml_setup, rescore=False,
        )
        assert result["applied"] is False
        assert "T_recommended == T_pareto" in result["reason"]

    def test_skip_when_confidence_low(self, yaml_setup, tmp_path):
        audit = _audit_json_payload(confidence="medium")
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)
        result = apply_audit_recommendation(
            json_path, yaml_path=yaml_setup, rescore=False,
        )
        assert result["applied"] is False
        assert "recommendation_confidence" in result["reason"]

    def test_apply_writes_provenance(self, yaml_setup, tmp_path):
        audit = _audit_json_payload(T_pareto=0.978, T_recommended=0.75, confidence="high")
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)

        result = apply_audit_recommendation(
            json_path, yaml_path=yaml_setup, rescore=False,
        )
        assert result["applied"] is True
        assert result["T_old"] == 0.978
        assert result["T_new"] == 0.75
        assert result["model"] == "google_gemma-4-E2B-it"

        # Check the YAML write
        with open(yaml_setup) as f:
            data = yaml.safe_load(f)
        entry = data["google_gemma-4-E2B-it"]["vowel_omission"]
        assert entry["threshold"] == 0.75
        assert entry["source"] == "audit_0501_1811"
        assert entry["audit_run"].endswith("audit_0501_1811.json")
        # Provenance preserves prior state
        assert entry["previous"]["threshold"] == 0.978
        assert entry["previous"]["source"] == "pareto"
        assert entry["previous"]["ba"] == 0.965
        assert entry["previous"]["d_norm"] == 0.000238

    def test_apply_is_idempotent(self, yaml_setup, tmp_path):
        """Re-applying the same audit JSON must not nest provenance."""
        audit = _audit_json_payload()
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)

        apply_audit_recommendation(json_path, yaml_path=yaml_setup, rescore=False)
        apply_audit_recommendation(json_path, yaml_path=yaml_setup, rescore=False)

        with open(yaml_setup) as f:
            data = yaml.safe_load(f)
        entry = data["google_gemma-4-E2B-it"]["vowel_omission"]
        # `previous` does not contain a nested previous (we strip it on re-apply).
        assert "previous" in entry
        assert "previous" not in entry["previous"]
        # Threshold still equals the recommendation.
        assert entry["threshold"] == 0.75

    def test_revert_restores_previous(self, yaml_setup, tmp_path):
        audit = _audit_json_payload()
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)
        apply_audit_recommendation(json_path, yaml_path=yaml_setup, rescore=False)

        result = revert_audit_recommendation(
            model_label="google_gemma-4-E2B-it",
            conflict_id="vowel_omission",
            yaml_path=yaml_setup,
            rescore=False,
        )
        assert result["reverted"] is True
        assert result["T_new"] == 0.978  # restored

        with open(yaml_setup) as f:
            data = yaml.safe_load(f)
        entry = data["google_gemma-4-E2B-it"]["vowel_omission"]
        assert entry["threshold"] == 0.978
        assert entry["source"] == "pareto"
        # `previous` block removed after revert
        assert "previous" not in entry

    def test_revert_noop_when_no_previous(self, yaml_setup):
        # No previous block exists yet — revert should refuse cleanly.
        result = revert_audit_recommendation(
            model_label="google_gemma-4-E2B-it",
            conflict_id="vowel_omission",
            yaml_path=yaml_setup,
            rescore=False,
        )
        assert result["reverted"] is False
        assert "previous" in result["reason"]

    def test_apply_rejects_missing_section(self, yaml_setup, tmp_path):
        audit = _audit_json_payload(model="nonexistent_model")
        json_path = tmp_path / "audit_0501_1811.json"
        _write_json(json_path, audit)
        result = apply_audit_recommendation(
            json_path, yaml_path=yaml_setup, rescore=False,
        )
        assert result["applied"] is False
        assert "no per-model section" in result["reason"]


# ---- Audit-lock protection (is_ambiguous, update_thresholds_yaml) ----


class TestAuditLockProtection:
    """Audit-set entries (source: audit_*) must not be silently overwritten."""

    def test_is_ambiguous_locked_for_audit_source(self):
        """is_ambiguous returns False for audit-set entries even if metrics say ambiguous."""
        # Metrics that would normally trigger ambiguous (high d_norm + low ba)
        result = {
            "feasible": True, "d_norm": 0.05, "c_norm": 0.05, "ba": 0.90,
            "source": "audit_0501_1811",
        }
        assert is_ambiguous(result, max_ba=1.0) is False

    def test_is_ambiguous_unlocked_for_pareto_source(self):
        """Same metrics, source=pareto → ambiguous=True."""
        result = {
            "feasible": True, "d_norm": 0.05, "c_norm": 0.05, "ba": 0.90,
            "source": "pareto",
        }
        assert is_ambiguous(result, max_ba=1.0) is True

    def test_get_threshold_info_exposes_audit_lock(self, tmp_path, monkeypatch):
        """get_threshold_info must return is_audit_locked=True when source=audit_*."""
        from phase0_v2.config import conflict_config

        yaml_path = tmp_path / "thresholds.yaml"
        _write_yaml(yaml_path, {
            "default": {"vowel_omission": 0.5},
            "google_gemma-4-E2B-it": {
                "vowel_omission": {
                    "threshold": 0.75,
                    "source": "audit_0501_1811",
                    "audit_run": "/path/to/audit.json",
                },
                "other_conflict": {
                    "threshold": 0.50,
                    "source": "pareto",
                },
            },
        })

        # Monkeypatch the cached loader to use our temp YAML
        monkeypatch.setattr(conflict_config, "_CONFIG_PATH", yaml_path)
        conflict_config._load_thresholds.cache_clear()

        info_locked = conflict_config.get_threshold_info(
            "vowel_omission", "google/gemma-4-E2B-it"
        )
        assert info_locked["is_audit_locked"] is True
        assert info_locked["source"] == "audit_0501_1811"
        assert info_locked["audit_run"] == "/path/to/audit.json"

        info_unlocked = conflict_config.get_threshold_info(
            "other_conflict", "google/gemma-4-E2B-it"
        )
        assert info_unlocked["is_audit_locked"] is False
        assert info_unlocked["source"] == "pareto"

        conflict_config._load_thresholds.cache_clear()

    def test_update_yaml_preserves_audit_locked_entry(self, tmp_path):
        """update_thresholds_yaml without --force-overwrite-audit must skip audit_* entries."""
        from phase0_v2.calibration.per_model_thresholds import update_thresholds_yaml

        yaml_path = tmp_path / "thresholds.yaml"
        _write_yaml(yaml_path, {
            "default": {},
            "google_gemma-4-E2B-it": {
                "_meta": {"last_updated": "2026-05-01"},
                "vowel_omission": {
                    "threshold": 0.75,
                    "source": "audit_0501_1811",
                    "ba": 1.0,
                },
                "first_vs_third_person": {
                    "threshold": 0.337,
                    "source": "pareto",
                },
            },
        })

        # Pareto wants to write new picks for both conflicts
        results = [
            {
                "conflict_id": "vowel_omission",
                "threshold": 0.978,  # would clobber audit's 0.75
                "feasible": True,
                "d_norm": 0.001, "c_norm": 0.012, "ba": 0.965,
                "max_ba": 1.0, "distribution": "bimodal",
                "ambiguous": True,
            },
            {
                "conflict_id": "first_vs_third_person",
                "threshold": 0.420,  # ok to overwrite
                "feasible": True,
                "d_norm": 0.0, "c_norm": 0.0, "ba": 1.0,
                "max_ba": 1.0, "distribution": "bimodal",
                "ambiguous": False,
            },
        ]
        result = update_thresholds_yaml(
            yaml_path, "google/gemma-4-E2B-it", results,
            pareto_caps={"max_d": 0.02, "max_c": 0.02, "min_ba": 0.95},
        )
        assert "vowel_omission" in result["skipped_locked"]
        assert "first_vs_third_person" not in result["skipped_locked"]

        # YAML state: vowel_omission preserved, first_vs_third_person updated
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        section = data["google_gemma-4-E2B-it"]
        assert section["vowel_omission"]["threshold"] == 0.75  # unchanged
        assert section["vowel_omission"]["source"] == "audit_0501_1811"
        assert section["first_vs_third_person"]["threshold"] == 0.420  # updated

    def test_collect_threshold_provenance(self, tmp_path, monkeypatch):
        """_collect_threshold_provenance classifies entries as audit/pareto/legacy."""
        from phase0_v2.calibration import audit_helpers
        from phase0_v2.calibration.audit_helpers import _collect_threshold_provenance

        yaml_path = tmp_path / "thresholds.yaml"
        _write_yaml(yaml_path, {
            "default": {},
            "google_gemma-4-E2B-it": {
                "_meta": {"last_updated": "2026-05-01"},
                "vowel_omission": {
                    "threshold": 0.75,
                    "source": "audit_0501_1811",
                    "audit_run": "/tmp/audit.json",
                },
                "first_vs_third_person": {
                    "threshold": 0.337,
                    "source": "pareto",
                },
                "legacy_dict_no_source": {
                    "threshold": 0.5,
                },
                "legacy_scalar": 0.42,
            },
        })
        monkeypatch.setattr(audit_helpers, "_THRESHOLDS_YAML", yaml_path)

        rows = _collect_threshold_provenance("google_gemma-4-E2B-it")
        by_id = {r["id"]: r for r in rows}

        assert by_id["vowel_omission"]["source_kind"] == "audit"
        assert by_id["vowel_omission"]["set_on"] == "0501_1811"
        assert by_id["vowel_omission"]["audit_run"] == "/tmp/audit.json"

        assert by_id["first_vs_third_person"]["source_kind"] == "pareto"
        assert by_id["legacy_dict_no_source"]["source_kind"] == "legacy"
        assert by_id["legacy_scalar"]["source_kind"] == "legacy"
        assert by_id["legacy_scalar"]["threshold"] == 0.42

    def test_update_yaml_force_overwrite_audit(self, tmp_path):
        """With force_overwrite_audit=True, audit_* entries are overwritten."""
        from phase0_v2.calibration.per_model_thresholds import update_thresholds_yaml

        yaml_path = tmp_path / "thresholds.yaml"
        _write_yaml(yaml_path, {
            "default": {},
            "google_gemma-4-E2B-it": {
                "_meta": {"last_updated": "2026-05-01"},
                "vowel_omission": {
                    "threshold": 0.75, "source": "audit_0501_1811",
                },
            },
        })
        results = [{
            "conflict_id": "vowel_omission",
            "threshold": 0.978,
            "feasible": True,
            "d_norm": 0.001, "c_norm": 0.012, "ba": 0.965,
            "max_ba": 1.0, "distribution": "bimodal",
            "ambiguous": True,
        }]
        result = update_thresholds_yaml(
            yaml_path, "google/gemma-4-E2B-it", results,
            pareto_caps={"max_d": 0.02, "max_c": 0.02, "min_ba": 0.95},
            force_overwrite_audit=True,
        )
        assert result["skipped_locked"] == []
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        entry = data["google_gemma-4-E2B-it"]["vowel_omission"]
        assert entry["threshold"] == 0.978  # overwritten
        # source field gone — Pareto wrote a fresh entry
        assert "source" not in entry or entry.get("source") != "audit_0501_1811"
