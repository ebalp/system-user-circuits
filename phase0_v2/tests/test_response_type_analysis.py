"""Tests for phase0_v2.calibration.response_type_analysis."""

import json
from pathlib import Path

import pytest
import yaml

from phase0_v2.calibration.response_type_analysis import (
    classify_response,
    compute_conflict_structure,
    compute_structure_by_constraint,
    compute_structure_stats,
    get_system_constraint_type,
    load_constraint_type_map,
    main,
)


# ---------------------------------------------------------------------------
# classify_response
# ---------------------------------------------------------------------------


class TestClassifyResponse:
    def test_clean_no_tags(self):
        assert classify_response({"response": "Hello world"}) == "clean"

    def test_bare_refusal(self):
        rec = {"refusal_tags": {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}}
        assert classify_response(rec) == "bare_refusal"

    def test_refusal_content(self):
        rec = {"refusal_tags": {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}}
        assert classify_response(rec) == "refusal_content"

    def test_meta_content(self):
        rec = {"refusal_tags": {"structure": ["metacommentary", "content"], "word_counts": [5, 20], "char_spans": [[0, 30], [30, 100]]}}
        assert classify_response(rec) == "meta_content"


# ---------------------------------------------------------------------------
# get_system_constraint_type
# ---------------------------------------------------------------------------


class TestGetSystemConstraintType:
    def setup_method(self):
        self.type_map = {
            "keyword_avoidance": {"a": "presence", "b": "avoidance"},
            "formal_vs_casual_tone": {"a": "ambiguous", "b": "ambiguous"},
        }

    def test_a_to_b_returns_a_type(self):
        rec = {"conflict_id": "keyword_avoidance", "direction": "a_to_b"}
        assert get_system_constraint_type(rec, self.type_map) == "presence"

    def test_b_to_a_returns_b_type(self):
        rec = {"conflict_id": "keyword_avoidance", "direction": "b_to_a"}
        assert get_system_constraint_type(rec, self.type_map) == "avoidance"

    def test_unknown_conflict(self):
        rec = {"conflict_id": "nonexistent", "direction": "a_to_b"}
        assert get_system_constraint_type(rec, self.type_map) == "unknown"

    def test_none_direction(self):
        rec = {"conflict_id": "keyword_avoidance", "direction": "none"}
        assert get_system_constraint_type(rec, self.type_map) == "unknown"


# ---------------------------------------------------------------------------
# compute_structure_stats
# ---------------------------------------------------------------------------


class TestComputeStructureStats:
    def test_basic(self):
        records = [
            {"condition": "C", "label": "followed_system", "refusal_tags": {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}},
            {"condition": "C", "label": "followed_user"},
            {"condition": "C", "label": "followed_user"},
            {"condition": "A", "label": "followed_system"},  # excluded
        ]
        stats = compute_structure_stats(records)
        assert stats["total"] == 3
        assert len(stats["patterns"]) == 2
        # content should be most common
        assert stats["patterns"][0]["structure"] == ("content",)
        assert stats["patterns"][0]["count"] == 2

    def test_with_type_map(self):
        records = [
            {"condition": "C", "label": "followed_system", "conflict_id": "foo", "direction": "b_to_a",
             "refusal_tags": {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}},
        ]
        type_map = {"foo": {"a": "presence", "b": "avoidance"}}
        stats = compute_structure_stats(records, type_map)
        assert "constraint_types" in stats["patterns"][0]
        assert stats["patterns"][0]["constraint_types"]["avoidance"] == 1


# ---------------------------------------------------------------------------
# compute_conflict_structure
# ---------------------------------------------------------------------------


class TestComputeConflictStructure:
    def test_per_direction(self):
        records = [
            {"condition": "C", "direction": "a_to_b", "label": "followed_system", "conflict_id": "foo"},
            {"condition": "C", "direction": "a_to_b", "label": "followed_user", "conflict_id": "foo",
             "refusal_tags": {"structure": ["refusal", "content"], "word_counts": [4, 20], "char_spans": [[0, 20], [20, 100]]}},
            {"condition": "C", "direction": "b_to_a", "label": "followed_system", "conflict_id": "foo",
             "refusal_tags": {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 20]]}},
        ]
        type_map = {"foo": {"a": "presence", "b": "avoidance"}}
        stats = compute_conflict_structure(records, type_map)
        assert stats["a_to_b"]["total"] == 2
        assert stats["a_to_b"]["sys_constraint"] == "presence"
        assert stats["b_to_a"]["total"] == 1
        assert stats["b_to_a"]["sys_constraint"] == "avoidance"


# ---------------------------------------------------------------------------
# load_constraint_type_map
# ---------------------------------------------------------------------------


class TestLoadConstraintTypeMap:
    def test_from_fixture(self, tmp_path):
        data = {
            "_meta": {"last_updated": "2026-03-25"},
            "foo_conflict": {"constraint_a_type": "presence", "constraint_b_type": "avoidance"},
        }
        yaml_path = tmp_path / "conflicts.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        result = load_constraint_type_map(yaml_path)
        assert "_meta" not in result
        assert result["foo_conflict"] == {"a": "presence", "b": "avoidance"}

    def test_loads_real_conflicts_yaml(self):
        real_yaml = Path(__file__).resolve().parent.parent / "config" / "conflicts.yaml"
        if not real_yaml.exists():
            pytest.skip("conflicts.yaml not found")
        result = load_constraint_type_map(real_yaml)
        assert len(result) >= 40


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    @pytest.fixture
    def sample_data(self, tmp_path):
        conflicts = {
            "_meta": {"last_updated": "2026-03-25"},
            "test_conflict": {"constraint_a_type": "presence", "constraint_b_type": "avoidance"},
            "ambig_conflict": {"constraint_a_type": "ambiguous", "constraint_b_type": "ambiguous"},
        }
        yaml_path = tmp_path / "conflicts.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(conflicts, f)

        records = [
            {"model": "test-model", "condition": "C", "conflict_id": "test_conflict",
             "direction": "b_to_a", "label": "followed_system", "response": "I cannot do that.",
             "refusal_tags": {"structure": ["refusal"], "word_counts": [4], "char_spans": [[0, 17]]}},
            {"model": "test-model", "condition": "C", "conflict_id": "test_conflict",
             "direction": "a_to_b", "label": "followed_user", "response": "The answer is 42."},
            {"model": "test-model", "condition": "C", "conflict_id": "ambig_conflict",
             "direction": "a_to_b", "label": "followed_system", "response": "As instructed, here is the answer.",
             "refusal_tags": {"structure": ["metacommentary", "content"], "word_counts": [2, 5], "char_spans": [[0, 15], [15, 34]]}},
            {"model": "test-model", "condition": "A", "conflict_id": "test_conflict",
             "direction": "none", "label": "followed_system", "response": "The answer is 42."},
        ]
        jsonl_path = tmp_path / "results.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        return str(jsonl_path), str(yaml_path)

    def test_full_output(self, sample_data, capsys):
        jsonl_path, yaml_path = sample_data
        main([jsonl_path, "--conflicts-yaml", yaml_path])
        out = capsys.readouterr().out
        assert "RESPONSE STRUCTURE ANALYSIS" in out
        assert "test-model" in out

    def test_per_conflict(self, sample_data, capsys):
        jsonl_path, yaml_path = sample_data
        main([jsonl_path, "--conflict", "test_conflict", "--conflicts-yaml", yaml_path])
        out = capsys.readouterr().out
        assert "test_conflict" in out
        assert "a_to_b" in out
        assert "b_to_a" in out

    def test_by_constraint(self, sample_data, capsys):
        jsonl_path, yaml_path = sample_data
        main([jsonl_path, "--focus", "by-constraint", "--conflicts-yaml", yaml_path])
        out = capsys.readouterr().out
        assert "STRUCTURE BY CONSTRAINT TYPE" in out

    def test_no_condition_c(self, tmp_path, capsys):
        conflicts = {"_meta": {"last_updated": "2026-03-25"}, "foo": {"constraint_a_type": "presence", "constraint_b_type": "avoidance"}}
        yaml_path = tmp_path / "conflicts.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(conflicts, f)
        records = [{"model": "m", "condition": "A", "conflict_id": "foo", "response": "hi"}]
        jsonl_path = tmp_path / "results.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        main([str(jsonl_path), "--conflicts-yaml", str(yaml_path)])
        out = capsys.readouterr().out
        assert "no condition C records" in out
