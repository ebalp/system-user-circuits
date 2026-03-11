"""Tests for phase0_v2.calibration.audit_conflict."""

import json
import random

import pytest

from phase0_v2.calibration.audit_conflict import (
    _condition_c_breakdown,
    _score_distribution,
    compute_summary,
    load_and_filter,
    print_samples,
    print_summary,
    query_records,
    sample_records,
)


# ── Helpers ──


def _make_record(
    conflict_id="forbidden_words",
    condition="C",
    direction="a_to_b",
    label="followed_system",
    verify_system_score=1.0,
    verify_user_score=0.0,
    verify_system_result=True,
    verify_user_result=False,
    response="Test response text",
    task_id="photosynthesis",
    system_style="direct",
    user_style="direct",
    system_prompt="System constraint",
    user_prompt="User constraint",
    model="meta-llama/Llama-3.1-8B-Instruct",
    error=None,
):
    return {
        "conflict_id": conflict_id,
        "condition": condition,
        "direction": direction,
        "label": label,
        "verify_system_score": verify_system_score,
        "verify_user_score": verify_user_score,
        "verify_system_result": verify_system_result,
        "verify_user_result": verify_user_result,
        "response": response,
        "task_id": task_id,
        "system_style": system_style,
        "user_style": user_style,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model": model,
        "error": error,
    }


def _make_results_file(tmp_path, records, filename="model_results.jsonl"):
    path = tmp_path / filename
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


# ── load_and_filter ──


class TestLoadAndFilter:
    def test_filters_to_conflict(self, tmp_path):
        records = [
            _make_record(conflict_id="forbidden_words"),
            _make_record(conflict_id="other_conflict"),
            _make_record(conflict_id="forbidden_words"),
        ]
        path = _make_results_file(tmp_path, records)
        result = load_and_filter([path], "forbidden_words")
        assert len(result) == 1
        _, _, filtered = result[0]
        assert len(filtered) == 2
        assert all(r["conflict_id"] == "forbidden_words" for r in filtered)

    def test_multiple_files(self, tmp_path):
        path1 = _make_results_file(
            tmp_path,
            [_make_record(model="meta-llama/Llama-3.1-8B-Instruct")],
            "8b_results.jsonl",
        )
        path2 = _make_results_file(
            tmp_path,
            [_make_record(model="meta-llama/Llama-3.3-70B-Instruct")],
            "70b_results.jsonl",
        )
        result = load_and_filter([path1, path2], "forbidden_words")
        assert len(result) == 2
        assert result[0][0] == "8B"
        assert result[1][0] == "70B"

    def test_no_match_returns_empty(self, tmp_path):
        path = _make_results_file(tmp_path, [_make_record(conflict_id="other")])
        result = load_and_filter([path], "forbidden_words")
        _, _, filtered = result[0]
        assert filtered == []

    def test_error_records_excluded(self, tmp_path):
        records = [
            _make_record(error="timeout"),
            _make_record(error=None),
        ]
        path = _make_results_file(tmp_path, records)
        _, _, filtered = load_and_filter([path], "forbidden_words")[0]
        assert len(filtered) == 1


# ── _condition_c_breakdown ──


class TestConditionCBreakdown:
    def test_counts_by_direction(self):
        records = [
            _make_record(condition="C", direction="a_to_b", label="followed_system"),
            _make_record(condition="C", direction="a_to_b", label="followed_user"),
            _make_record(condition="C", direction="b_to_a", label="followed_system"),
            _make_record(condition="C", direction="b_to_a", label="followed_system"),
            _make_record(condition="C", direction="b_to_a", label="followed_system"),
        ]
        result = _condition_c_breakdown(records)
        assert result["a_to_b"]["sys"] == 1
        assert result["a_to_b"]["usr"] == 1
        assert result["a_to_b"]["n"] == 2
        assert result["b_to_a"]["sys"] == 3
        assert result["b_to_a"]["n"] == 3
        assert result["total"]["sys"] == 4
        assert result["total"]["usr"] == 1
        assert result["total"]["n"] == 5

    def test_followed_both_and_neither(self):
        records = [
            _make_record(condition="C", direction="a_to_b", label="followed_both"),
            _make_record(condition="C", direction="a_to_b", label="followed_neither"),
        ]
        result = _condition_c_breakdown(records)
        assert result["a_to_b"]["both"] == 1
        assert result["a_to_b"]["neither"] == 1
        assert result["total"]["both"] == 1
        assert result["total"]["neither"] == 1

    def test_ignores_non_c_conditions(self):
        records = [
            _make_record(condition="A", direction="a_to_b", label="followed_system"),
            _make_record(condition="C", direction="a_to_b", label="followed_system"),
        ]
        result = _condition_c_breakdown(records)
        assert result["total"]["n"] == 1

    def test_zero_c_records(self):
        records = [_make_record(condition="A")]
        result = _condition_c_breakdown(records)
        assert result["total"]["n"] == 0
        assert result["a_to_b"]["n"] == 0


# ── _score_distribution ──


class TestScoreDistribution:
    def test_bins_correctly(self):
        records = [
            _make_record(condition="C", verify_system_score=0.05),
            _make_record(condition="C", verify_system_score=0.2),
            _make_record(condition="C", verify_system_score=0.4),
            _make_record(condition="C", verify_system_score=0.6),
            _make_record(condition="C", verify_system_score=0.8),
            _make_record(condition="C", verify_system_score=1.0),
        ]
        dist = _score_distribution(records)
        assert dist["[0,.1)"] == 1
        assert dist["[.1,.3)"] == 1
        assert dist["[.3,.5)"] == 1
        assert dist["[.5,.7)"] == 1
        assert dist["[.7,.9)"] == 1
        assert dist["[.9,1]"] == 1

    def test_ignores_non_c(self):
        records = [
            _make_record(condition="A", verify_system_score=0.5),
            _make_record(condition="C", verify_system_score=0.5),
        ]
        dist = _score_distribution(records)
        assert dist["[.5,.7)"] == 1
        assert sum(dist.values()) == 1


# ── sample_records ──


class TestSampleRecords:
    def _pool(self):
        return [
            _make_record(condition="C", direction="a_to_b", label="followed_system"),
            _make_record(condition="C", direction="a_to_b", label="followed_user"),
            _make_record(condition="C", direction="b_to_a", label="followed_system"),
            _make_record(condition="C", direction="b_to_a", label="followed_user"),
        ]

    def test_filter_by_label(self):
        result = sample_records(self._pool(), "forbidden_words", label="followed_system", n=100)
        assert len(result) == 2
        assert all(r["label"] == "followed_system" for r in result)

    def test_filter_by_direction(self):
        result = sample_records(self._pool(), "forbidden_words", direction="b_to_a", n=100)
        assert len(result) == 2
        assert all(r["direction"] == "b_to_a" for r in result)

    def test_combined_label_direction(self):
        result = sample_records(
            self._pool(), "forbidden_words",
            label="followed_system", direction="a_to_b", n=100,
        )
        assert len(result) == 1

    def test_n_limits_count(self):
        random.seed(42)
        result = sample_records(self._pool(), "forbidden_words", n=2)
        assert len(result) == 2

    def test_fewer_than_n(self):
        result = sample_records(self._pool(), "forbidden_words", label="followed_system", n=100)
        assert len(result) == 2  # only 2 match, no error

    def test_empty_when_no_match(self):
        result = sample_records(self._pool(), "forbidden_words", label="followed_both", n=100)
        assert result == []

    def test_near_threshold_float(self):
        records = [
            _make_record(condition="C", verify_system_score=0.1),
            _make_record(condition="C", verify_system_score=0.5),
            _make_record(condition="C", verify_system_score=0.9),
        ]
        # forbidden_words is bool, so use a real float conflict
        # Instead, test with records directly — near_threshold calls get_threshold
        # Use "past_vs_present_tense" which is a float conflict
        for r in records:
            r["conflict_id"] = "past_vs_present_tense"
        result = sample_records(records, "past_vs_present_tense", near_threshold=True, n=2)
        assert len(result) == 2
        # First record should be closest to threshold

    def test_near_threshold_bool_warns(self, capsys):
        result = sample_records([], "forbidden_words", near_threshold=True)
        assert result == []
        captured = capsys.readouterr()
        assert "no-op" in captured.err.lower() or "WARNING" in captured.err

    def test_anomalies_returns_baseline_mismatches(self):
        records = [
            # Anomaly: condition A, followed_user
            _make_record(condition="A", direction="a_to_b", label="followed_user"),
            # Normal: condition A, followed_system
            _make_record(condition="A", direction="a_to_b", label="followed_system"),
        ]
        result = sample_records(records, "forbidden_words", anomalies=True, n=100)
        assert len(result) == 1
        assert result[0]["anomaly_reason"] == "cond_A_followed_user"


# ── query_records ──


class TestQueryRecords:
    def _pool(self):
        return [
            _make_record(condition="A", direction="a_to_b", response="Hello World"),
            _make_record(condition="C", direction="a_to_b", response="hello there",
                         verify_system_score=0.5),
            _make_record(condition="C", direction="b_to_a", response="Goodbye",
                         verify_system_score=0.8),
            _make_record(condition="B", direction="b_to_a", response="Testing"),
        ]

    def test_filter_by_condition(self):
        result = query_records(self._pool(), "forbidden_words", condition="C", n=100)
        assert len(result) == 2

    def test_filter_by_direction(self):
        result = query_records(self._pool(), "forbidden_words", direction="a_to_b", n=100)
        assert len(result) == 2

    def test_response_contains_case_insensitive(self):
        result = query_records(
            self._pool(), "forbidden_words", response_contains="hello", n=100,
        )
        assert len(result) == 2  # "Hello World" and "hello there"

    def test_score_range(self):
        result = query_records(
            self._pool(), "forbidden_words", score_range=(0.3, 0.7), n=100,
        )
        assert len(result) == 1
        assert result[0]["verify_system_score"] == 0.5

    def test_count_returns_int(self):
        result = query_records(self._pool(), "forbidden_words", condition="C", count=True)
        assert result == 2

    def test_combined_filters(self):
        result = query_records(
            self._pool(), "forbidden_words",
            condition="C", direction="a_to_b", response_contains="hello", n=100,
        )
        assert len(result) == 1
        assert result[0]["response"] == "hello there"

    def test_empty_result(self):
        result = query_records(
            self._pool(), "forbidden_words",
            condition="D", n=100,
        )
        assert result == []


# ── Print formatting ──


class TestPrintFormatting:
    def _make_summary(self, is_float=False, n_models=1):
        models = []
        for i in range(n_models):
            name = ["8B", "70B", "Gemma-27B"][i]
            cond_c = {
                "a_to_b": {"sys": 5, "usr": 3, "both": 0, "neither": 2, "n": 10},
                "b_to_a": {"sys": 4, "usr": 4, "both": 1, "neither": 1, "n": 10},
                "total": {"sys": 9, "usr": 7, "both": 1, "neither": 3, "n": 20},
            }
            models.append({
                "name": name,
                "baseline": {"sbr_a": 1.0, "ucr_a": 1.0, "sbr_b": 0.95, "ucr_b": 0.98},
                "ba": 0.990,
                "cond_c": cond_c,
                "score_dist": {"[0,.1)": 5, "[.1,.3)": 0, "[.3,.5)": 0,
                                "[.5,.7)": 0, "[.7,.9)": 0, "[.9,1]": 15} if is_float else None,
                "optimal": {"t_low": 0.4, "t_high": 0.8} if is_float else None,
                "n_anomalies": 2,
            })
        return {
            "conflict_id": "test_conflict",
            "is_float": is_float,
            "threshold": 0.6 if is_float else None,
            "constraint_a": "constraint A",
            "constraint_b": "constraint B",
            "counterbalance": "full",
            "models": models,
        }

    def test_summary_has_section_headers(self, capsys):
        print_summary(self._make_summary())
        out = capsys.readouterr().out
        assert "AUDIT:" in out
        assert "BASELINES" in out
        assert "CONDITION C" in out
        assert "ANOMALIES" in out

    def test_float_summary_has_threshold_section(self, capsys):
        print_summary(self._make_summary(is_float=True))
        out = capsys.readouterr().out
        assert "THRESHOLD ANALYSIS" in out
        assert "SCORE DISTRIBUTION" in out

    def test_bool_summary_no_threshold(self, capsys):
        print_summary(self._make_summary(is_float=False))
        out = capsys.readouterr().out
        assert "THRESHOLD ANALYSIS" not in out
        assert "SCORE DISTRIBUTION" not in out

    def test_multi_model_shows_all(self, capsys):
        print_summary(self._make_summary(n_models=2))
        out = capsys.readouterr().out
        assert "8B" in out
        assert "70B" in out

    def test_print_samples_includes_response(self, capsys):
        samples = [_make_record(response="Unique response text here")]
        print_samples(samples, "forbidden_words")
        out = capsys.readouterr().out
        assert "Unique response text here" in out
        assert "sys_score=" in out
