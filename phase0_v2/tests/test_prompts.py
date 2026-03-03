"""Test PromptGenerator produces correct prompts for all conditions."""

import pytest
from phase0_v2.src.config import load_config, Task
from phase0_v2.src.prompts import PromptGenerator, _deterministic_seed
from phase0_v2.conflicts.registry import get_conflict, get_all_conflicts


@pytest.fixture
def config():
    return load_config("phase0_v2/config/experiment.yaml")


@pytest.fixture
def generator(config):
    return PromptGenerator(config)


@pytest.fixture
def task():
    return Task(id="test_task", prompt="Explain how photosynthesis works.")


class TestConditionA:
    def test_generates_two_for_invertible(self, generator, task):
        """Invertible conflict: 2 prompts (side a + side b baselines)."""
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_a = [p for p in prompts if p.condition == "A"]
        assert len(cond_a) == 2
        directions = {p.direction for p in cond_a}
        assert directions == {"a_to_b", "b_to_a"}

    def test_generates_one_for_non_invertible(self, generator, task):
        """Non-invertible conflict: 1 prompt (side a only)."""
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_a = [p for p in prompts if p.condition == "A"]
        assert len(cond_a) == 1
        assert cond_a[0].direction == "a_to_b"

    def test_fields(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in [p for p in prompts if p.condition == "A"]:
            assert p.direction in ("a_to_b", "b_to_a")
            assert p.system_style == "bare"
            assert p.user_style == "task_only"
            assert p.expected_label == "followed_system"
            assert p.system_message  # non-empty
            assert p.user_message == task.prompt  # raw task, no wrapping


class TestConditionB:
    def test_generates_two_for_invertible(self, generator, config, task):
        """Invertible conflict: 2 prompts (side a + side b baselines)."""
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_b = [p for p in prompts if p.condition == "B"]
        assert len(cond_b) == 2
        directions = {p.direction for p in cond_b}
        assert directions == {"a_to_b", "b_to_a"}
        for p in cond_b:
            assert p.user_style == config.default_user_style

    def test_generates_one_for_non_invertible(self, generator, task):
        """Non-invertible conflict: 1 prompt (side a only)."""
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_b = [p for p in prompts if p.condition == "B"]
        assert len(cond_b) == 1
        assert cond_b[0].direction == "a_to_b"

    def test_fields(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in [p for p in prompts if p.condition == "B"]:
            assert p.direction in ("a_to_b", "b_to_a")
            assert p.system_style is None
            assert p.expected_label == "followed_user"
            assert p.system_message == ""


class TestConditionC:
    def test_invertible_has_both_directions(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_c = [p for p in prompts if p.condition == "C"]
        directions = {p.direction for p in cond_c}
        assert "a_to_b" in directions
        assert "b_to_a" in directions

    def test_non_invertible_only_a_to_b(self, generator, task):
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_c = [p for p in prompts if p.condition == "C"]
        directions = {p.direction for p in cond_c}
        assert "a_to_b" in directions
        assert "b_to_a" not in directions

    def test_all_system_style_user_style_combos(self, generator, config, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_c = [p for p in prompts if p.condition == "C"]
        system_styles = {p.system_style for p in cond_c}
        styles = {p.user_style for p in cond_c}
        assert system_styles == set(config.condition_c_system_styles)
        assert styles == set(config.user_styles_to_test)

    def test_expected_count_invertible(self, generator, config, task):
        """2 dirs x 5 system styles x 5 user styles = 50."""
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_c = [p for p in prompts if p.condition == "C"]
        assert len(cond_c) == 2 * 5 * 5

    def test_expected_count_non_invertible(self, generator, config, task):
        """1 dir x 5 system styles x 5 user styles = 25."""
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_c = [p for p in prompts if p.condition == "C"]
        assert len(cond_c) == 1 * 5 * 5

    def test_expected_label(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in [p for p in prompts if p.condition == "C"]:
            assert p.expected_label == "followed_system"

    def test_system_message_wrapped_with_system_style(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        authority = [p for p in prompts if p.condition == "C" and p.system_style == "authority"]
        assert len(authority) > 0
        # Authority wrapping should include the preamble text
        assert "absolute priority" in authority[0].system_message.lower() or \
               "locked" in authority[0].system_message.lower()


class TestConditionD:
    def test_only_for_invertible(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_d = [p for p in prompts if p.condition == "D"]
        assert len(cond_d) > 0

    def test_not_for_non_invertible(self, generator, task):
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_d = [p for p in prompts if p.condition == "D"]
        assert len(cond_d) == 0

    def test_fields(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in [p for p in prompts if p.condition == "D"]:
            assert p.system_message == ""
            assert p.system_style == "bare"
            assert p.user_style == "recency"
            assert p.expected_label == "followed_user"

    def test_expected_count_invertible(self, generator, task):
        """2 directions for invertible conflict."""
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        cond_d = [p for p in prompts if p.condition == "D"]
        assert len(cond_d) == 2


class TestPromptMetadata:
    def test_ids_unique(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found"

    def test_constraint_type_equals_conflict_id(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            assert p.constraint_type == p.conflict_id == "forbidden_words"

    def test_conflict_class_set(self, generator, task):
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        for p in prompts:
            assert p.conflict_class == "ForbiddenWordsConflict"

    def test_total_count_invertible(self, generator, task):
        """A:2 + B:2 + C:50 + D:2 = 56 per task."""
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, [task])
        assert len(prompts) == 56

    def test_total_count_non_invertible(self, generator, task):
        """A:1 + B:1 + C:25 + D:0 = 27 per task."""
        conflict = get_conflict("odd_even_syllables")
        prompts = generator.generate_for_conflict(conflict, [task])
        assert len(prompts) == 27

    def test_multiple_tasks(self, generator):
        """Prompt count scales linearly with task count."""
        tasks = [Task(id=f"t{i}", prompt=f"Task {i}") for i in range(3)]
        conflict = get_conflict("forbidden_words")
        prompts = generator.generate_for_conflict(conflict, tasks)
        assert len(prompts) == 56 * 3


class TestSeedIsolation:
    """Per-(conflict, task) seeding produces stable args regardless of execution context."""

    def test_same_args_regardless_of_other_conflicts(self, config):
        """Args for forbidden_words+task are identical whether generated alone or after other conflicts."""
        task = Task(id="test_task", prompt="Explain photosynthesis.")

        # Run forbidden_words alone
        gen1 = PromptGenerator(config)
        conflict_fw = get_conflict("forbidden_words")
        prompts_alone = gen1.generate_for_conflict(conflict_fw, [task])
        args_alone = prompts_alone[0].instruction_args

        # Run language_en_es first, then forbidden_words
        gen2 = PromptGenerator(config)
        conflict_lang = get_conflict("language_en_es")
        gen2.generate_for_conflict(conflict_lang, [task])
        prompts_after = gen2.generate_for_conflict(conflict_fw, [task])
        args_after = prompts_after[0].instruction_args

        assert args_alone == args_after

    def test_different_seed_different_args(self, config):
        """Different global seed produces different sampled args for parameterized conflicts."""
        task = Task(id="test_task", prompt="Explain photosynthesis.")
        conflict = get_conflict("forbidden_words")

        gen1 = PromptGenerator(config)
        prompts1 = gen1.generate_for_conflict(conflict, [task])
        args1 = prompts1[0].instruction_args

        # Modify seed
        import copy
        config2 = copy.deepcopy(config)
        config2.seed = 999
        gen2 = PromptGenerator(config2)
        prompts2 = gen2.generate_for_conflict(conflict, [task])
        args2 = prompts2[0].instruction_args

        assert args1 != args2

    def test_deterministic_seed_helper(self):
        """_deterministic_seed is pure: same inputs -> same output."""
        s1 = _deterministic_seed(42, "forbidden_words", "test_task")
        s2 = _deterministic_seed(42, "forbidden_words", "test_task")
        assert s1 == s2

        # Different key -> different seed
        s3 = _deterministic_seed(42, "language_en_es", "test_task")
        assert s1 != s3
