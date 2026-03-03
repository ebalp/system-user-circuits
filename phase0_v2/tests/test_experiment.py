"""Test experiment hashing, dedup, and record building."""

import pytest
from phase0_v2.src.experiment import ExperimentKey, compute_experiment_hash


class TestHashComputation:
    def test_deterministic(self):
        key = ExperimentKey(
            model="test", conflict_id="forbidden_words",
            instruction_args_json='{"word1": "a"}',
            task_id="t1", task_source="synthetic",
            condition="C", direction="a_to_b", system_style="compliance",
            user_style="jailbreak", temperature=0.0, max_tokens=512,
        )
        assert compute_experiment_hash(key) == compute_experiment_hash(key)

    def test_length_16(self):
        key = ExperimentKey(
            model="x", conflict_id="x", instruction_args_json="{}",
            task_id="x", task_source="synthetic",
            condition="A", direction="none", system_style=None,
            user_style="task_only", temperature=0.0, max_tokens=512,
        )
        assert len(compute_experiment_hash(key)) == 16

    def test_differs_on_direction_change(self):
        base = dict(
            model="test", conflict_id="fw",
            instruction_args_json='{"word1": "a"}',
            task_id="t1", task_source="synthetic",
            condition="C", direction="a_to_b", system_style="compliance",
            user_style="jailbreak", temperature=0.0, max_tokens=512,
        )
        h1 = compute_experiment_hash(ExperimentKey(**base))
        h2 = compute_experiment_hash(ExperimentKey(**{**base, "direction": "b_to_a"}))
        assert h1 != h2

    def test_differs_on_system_style_change(self):
        base = dict(
            model="test", conflict_id="fw",
            instruction_args_json='{"word1": "a"}',
            task_id="t1", task_source="synthetic",
            condition="C", direction="a_to_b", system_style="bare",
            user_style="jailbreak", temperature=0.0, max_tokens=512,
        )
        h1 = compute_experiment_hash(ExperimentKey(**base))
        h2 = compute_experiment_hash(ExperimentKey(**{**base, "system_style": "authority"}))
        assert h1 != h2

    def test_differs_on_model_change(self):
        base = dict(
            model="model_a", conflict_id="fw",
            instruction_args_json='{}',
            task_id="t1", task_source="synthetic",
            condition="C", direction="a_to_b", system_style="compliance",
            user_style="jailbreak", temperature=0.0, max_tokens=512,
        )
        h1 = compute_experiment_hash(ExperimentKey(**base))
        h2 = compute_experiment_hash(ExperimentKey(**{**base, "model": "model_b"}))
        assert h1 != h2

    def test_hex_chars_only(self):
        key = ExperimentKey(
            model="test", conflict_id="x", instruction_args_json="{}",
            task_id="x", task_source="synthetic",
            condition="A", direction="none", system_style=None,
            user_style="task_only", temperature=0.0, max_tokens=512,
        )
        h = compute_experiment_hash(key)
        assert all(c in "0123456789abcdef" for c in h)
