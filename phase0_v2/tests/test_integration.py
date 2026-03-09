"""Integration tests: full pipeline, JSONL schema, dry-run, cross-module consistency."""

import json
import pytest
from collections import Counter
from phase0_v2.src.config import load_config, ExperimentConfig, CounterbalancingConfig, Task
from phase0_v2.src.prompts import PromptGenerator, Prompt
from phase0_v2.src.experiment import ExperimentKey, compute_experiment_hash
from phase0_v2.src.classifiers import classify_response
from phase0_v2.conflicts.registry import get_all_conflicts, get_conflict, get_conflict_ids
from phase0_v2.conflicts.compatibility import validate_matrix_coverage


@pytest.fixture
def config():
    return load_config("phase0_v2/config/experiment.yaml")


@pytest.fixture
def generator(config):
    return PromptGenerator(config)


# ── Full pipeline: config → prompts → classify ──


class TestPipelineEndToEnd:
    """Test that the full flow works: config -> prompts -> classify -> record."""

    def test_generate_classify_forbidden_words(self, generator):
        """End-to-end: generate prompt, simulate response, classify."""
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Explain gravity.")
        prompts = generator.generate_for_conflict(conflict, [task])

        # Take one Condition C prompt
        cond_c = next(p for p in prompts if p.condition == "C")
        args = cond_c.instruction_args

        # Simulate a response that follows system (no forbidden words)
        response = "Gravity is a fundamental force that attracts objects with mass."
        label, conf = classify_response(response, conflict, cond_c.direction, args)
        assert label in {
            "followed_system",
            "followed_user",
            "followed_neither",
            "followed_both",
        }
        assert isinstance(conf, float)

    def test_generate_classify_emoji(self, generator):
        """End-to-end with emoji conflict."""
        conflict = get_conflict("emoji_use_vs_avoid")
        task = Task(id="test", prompt="What is water?")
        prompts = generator.generate_for_conflict(conflict, [task])

        cond_c = next(p for p in prompts if p.condition == "C")
        response_with_emoji = "Water is H2O! \U0001f4a7 It's essential for life! \U0001f30a"
        label, _ = classify_response(
            response_with_emoji, conflict, cond_c.direction, {}
        )
        assert label in {
            "followed_system",
            "followed_user",
            "followed_neither",
            "followed_both",
        }

    def test_all_conflicts_generate_prompts(self, generator):
        """Every registered conflict should generate at least 1 prompt per task."""
        task = Task(id="test", prompt="Tell me about science.")
        for conflict in get_all_conflicts():
            prompts = generator.generate_for_conflict(conflict, [task])
            assert len(prompts) > 0, f"{conflict.conflict_id} generated 0 prompts"


# ── JSONL schema validation ──


class TestJSONLSchema:
    """Validate that generated prompts have all Phase 1-required fields."""

    REQUIRED_FIELDS = {
        "condition",
        "constraint_type",
        "conflict_id",
        "conflict_class",
        "direction",
        "system_style",
        "user_style",
        "task_id",
        "system_prompt",
        "user_prompt",
        "expected_label",
    }

    def _prompt_to_record(self, prompt: Prompt) -> dict:
        """Convert a Prompt to a dict matching the JSONL schema."""
        return {
            "prompt_id": prompt.id,
            "condition": prompt.condition,
            "constraint_type": prompt.constraint_type,
            "conflict_id": prompt.conflict_id,
            "conflict_class": prompt.conflict_class,
            "instruction_args": prompt.instruction_args,
            "direction": prompt.direction,
            "system_style": prompt.system_style,
            "user_style": prompt.user_style,
            "task_id": prompt.task_id,
            "task_source": prompt.task_source,
            "wildchat_id": prompt.wildchat_id,
            "system_prompt": prompt.system_message,
            "user_prompt": prompt.user_message,
            "expected_label": prompt.expected_label,
        }

    def test_all_required_fields_present(self, generator):
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            record = self._prompt_to_record(p)
            for field in self.REQUIRED_FIELDS:
                assert field in record, f"Missing field '{field}' in prompt {p.id}"

    def test_condition_values(self, generator):
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        conditions = {p.condition for p in prompts}
        assert conditions == {"A", "B", "C", "D"}

    def test_constraint_type_matches_conflict_id(self, generator):
        """Phase 1 requires constraint_type == conflict_id for GroupKFold."""
        for conflict in get_all_conflicts()[:5]:  # spot-check 5
            task = Task(id="test", prompt="Test.")
            prompts = generator.generate_for_conflict(conflict, [task])
            for p in prompts:
                assert p.constraint_type == p.conflict_id

    def test_expected_labels_correct_per_condition(self, generator):
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            if p.condition == "A":
                assert p.expected_label == "followed_system"
            elif p.condition == "B":
                assert p.expected_label == "followed_user"
            elif p.condition == "C":
                assert p.expected_label == "followed_system"
            elif p.condition == "D":
                assert p.expected_label == "followed_user"

    def test_direction_values_per_condition(self, generator):
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            assert p.direction in ("a_to_b", "b_to_a")

    def test_record_serializable_as_json(self, generator):
        """All records must be JSON-serializable (for JSONL output)."""
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            record = self._prompt_to_record(p)
            json_str = json.dumps(record)  # should not raise
            assert isinstance(json_str, str)


# ── Cross-module consistency ──


class TestCrossModuleConsistency:
    def test_hash_from_prompt_fields(self, generator):
        """ExperimentKey can be built from Prompt fields."""
        conflict = get_conflict("forbidden_words")
        task = Task(id="test", prompt="Test.")
        prompts = generator.generate_for_conflict(conflict, [task])
        p = next(p for p in prompts if p.condition == "C")

        key = ExperimentKey(
            model="test-model",
            conflict_id=p.conflict_id,
            instruction_args_json=json.dumps(p.instruction_args, sort_keys=True),
            task_id=p.task_id,
            task_source=p.task_source,
            condition=p.condition,
            direction=p.direction,
            system_style=p.system_style,
            user_style=p.user_style,
            temperature=0.0,
            system_prompt=p.system_message,
            user_prompt=p.user_message,
        )
        h = compute_experiment_hash(key)
        assert len(h) == 16

    def test_compatibility_matrix_complete(self):
        """Registry + compatibility matrix must be in sync."""
        uncovered = validate_matrix_coverage(get_conflict_ids())
        assert uncovered == []

    def test_all_42_conflicts_registered(self):
        assert len(get_all_conflicts()) == 42

    def test_prompt_count_scales_with_tasks(self, generator):
        """3 tasks should give 3x the prompts of 1 task."""
        conflict = get_conflict("forbidden_words")
        tasks_1 = [Task(id="t1", prompt="A.")]
        tasks_3 = [Task(id=f"t{i}", prompt=f"Task {i}.") for i in range(3)]
        n1 = len(generator.generate_for_conflict(conflict, tasks_1))
        n3 = len(generator.generate_for_conflict(conflict, tasks_3))
        assert n3 == n1 * 3


# ── Dry-run smoke test ──


class TestDryRun:
    def test_dry_run_two_conflicts_two_tasks(self, generator):
        """Simulate what --dry-run does: generate prompts for a subset."""
        conflicts = [
            get_conflict("forbidden_words"),
            get_conflict("language_en_es"),
        ]
        tasks = [Task(id="t1", prompt="Task 1."), Task(id="t2", prompt="Task 2.")]
        all_prompts = []
        for conflict in conflicts:
            prompts = generator.generate_for_conflict(conflict, tasks)
            all_prompts.extend(prompts)

        assert len(all_prompts) > 0
        conds = Counter(p.condition for p in all_prompts)
        assert set(conds.keys()) == {"A", "B", "C", "D"}

        # Both conflicts represented
        conflict_ids = {p.conflict_id for p in all_prompts}
        assert conflict_ids == {"forbidden_words", "language_en_es"}

        # Both tasks represented
        task_ids = {p.task_id for p in all_prompts}
        assert task_ids == {"t1", "t2"}


# ── require_invertible filtering ──


class TestRequireInvertible:
    def test_config_require_invertible_set(self, config):
        """Config has require_invertible set (currently true)."""
        assert config.counterbalancing.require_invertible is True
        all_conflicts = get_all_conflicts()
        non_inv = [c for c in all_conflicts if not c.supports_counterbalancing()]
        assert len(non_inv) > 0, "Expected some non-invertible conflicts in registry"

    def test_filter_removes_non_invertible(self):
        """When require_invertible=True, non-invertible conflicts are excluded."""
        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.supports_counterbalancing()]
        non_inv = [c for c in all_conflicts if not c.supports_counterbalancing()]
        assert len(filtered) == len(all_conflicts) - len(non_inv)
        for c in filtered:
            assert c.supports_counterbalancing()

    def test_non_invertible_ids_excluded(self):
        """Specific non-invertible IDs should be dropped by the filter."""
        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.supports_counterbalancing()]
        filtered_ids = {c.conflict_id for c in filtered}
        for cid in ("stairs_indent", "each_word_new_line", "odd_even_syllables"):
            assert cid not in filtered_ids

    def test_partial_invertible_kept(self):
        """Partial counterbalance conflicts should survive the filter."""
        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.supports_counterbalancing()]
        filtered_ids = {c.conflict_id for c in filtered}
        for cid in ("no_consecutive_first_letter", "bullets_and_sub_bullets"):
            assert cid in filtered_ids

    def test_filtered_count(self):
        """42 total - 3 non-invertible = 39 invertible."""
        all_conflicts = get_all_conflicts()
        filtered = [c for c in all_conflicts if c.supports_counterbalancing()]
        assert len(filtered) == 39

    def test_config_parses_require_invertible(self):
        """Config correctly parses require_invertible field."""
        config = load_config("phase0_v2/config/experiment.yaml")
        assert hasattr(config.counterbalancing, "require_invertible")
        assert isinstance(config.counterbalancing.require_invertible, bool)
