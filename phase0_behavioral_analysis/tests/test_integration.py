"""
Integration tests for Phase 0 Behavioral Analysis.

Tests the full pipeline from configuration loading through prompt generation,
classification, labeling, and metrics computation.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.config import load_config, ExperimentConfig
from src.prompts import PromptGenerator, Prompt
from src.classifiers import (
    ClassificationResult,
    LanguageClassifier,
    FormatClassifier,
    get_classifier,
    compute_label,
)
from src.experiment import ExperimentResult
from src.metrics import MetricsCalculator, ModelMetrics


# Path to the actual config file
CONFIG_PATH = Path(__file__).parent.parent / "config" / "experiment.yaml"


class TestConfigToPromptsIntegration:
    """Integration tests: config → prompts → export → verify counts and structure."""

    def test_load_real_config(self):
        config = load_config(CONFIG_PATH)
        assert isinstance(config, ExperimentConfig)
        assert len(config.models) > 0
        assert len(config.constraint_types) > 0
        assert len(config.tasks) > 0
        assert len(config.conditions) == 4

    def test_generate_prompts_from_real_config(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        assert len(prompts) > 0
        assert all(isinstance(p, Prompt) for p in prompts)

    def test_all_conditions_present(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        conditions = {p.condition for p in prompts}
        assert conditions == {'A', 'B', 'C', 'D'}

    def test_all_constraint_types_present(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        constraint_types = {p.constraint_type for p in prompts}
        expected_types = {pair.constraint_type for pair in config.experiment_pairs}
        assert constraint_types == expected_types

    def test_counterbalancing_directions_present(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        c_directions = {p.direction for p in prompts if p.condition == 'C'}
        assert c_directions == {'a_to_b', 'b_to_a'}
        d_directions = {p.direction for p in prompts if p.condition == 'D'}
        assert d_directions == {'a_to_b', 'b_to_a'}
        a_directions = {p.direction for p in prompts if p.condition == 'A'}
        assert a_directions == {'option_a', 'option_b'}

    def test_all_strength_levels_in_condition_c(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        c_strengths = {p.strength for p in prompts if p.condition == 'C'}
        expected_strengths = set(config.condition_c_strengths)
        assert c_strengths == expected_strengths

    def test_unique_prompt_ids(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids)), "Duplicate prompt IDs found"

    def test_export_to_json_and_reload(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            generator.export_to_json(prompts, path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) == len(prompts)
            for item in data:
                assert 'id' in item
                assert 'condition' in item
                assert 'direction' in item
                assert 'expected_label' in item
        finally:
            os.unlink(path)

    def test_export_to_csv_and_verify_structure(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            generator.export_to_csv(prompts, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == len(prompts) + 1
            header = lines[0].strip()
            assert 'id' in header
            assert 'direction' in header
        finally:
            os.unlink(path)

    def test_prompt_system_messages_rendered_correctly(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        weak_prompts = [p for p in prompts if p.strength == 'weak' and p.condition == 'C']
        for p in weak_prompts[:5]:
            assert 'Please' in p.system_message or 'please' in p.system_message.lower()
        strong_prompts = [p for p in prompts if p.strength == 'strong' and p.condition == 'C']
        for p in strong_prompts[:5]:
            assert 'CRITICAL' in p.system_message

    def test_prompt_user_messages_contain_task(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        task_prompts = {t.prompt for t in config.tasks}
        for p in prompts:
            assert any(task in p.user_message for task in task_prompts), \
                f"User message doesn't contain task: {p.user_message}"


class TestClassificationToMetricsIntegration:
    """Integration tests: mock API responses → classification → labeling → metrics."""

    @pytest.fixture
    def sample_responses(self):
        return {
            'english': "The capital of France is Paris. It is a beautiful city known for the Eiffel Tower.",
            'spanish': "La capital de Francia es París. Es una ciudad hermosa conocida por la Torre Eiffel.",
            'json': '{"capital": "Paris", "country": "France", "population": 2161000}',
            'plain': "Paris is the capital of France with a population of about 2 million people.",
        }

    def test_language_classification_to_label(self, sample_responses):
        classifier = LanguageClassifier()
        result = classifier.classify(sample_responses['english'])
        label, confidence = compute_label(result, 'english', 'spanish')
        assert label == 'followed_system'
        assert confidence > 0.7

        result = classifier.classify(sample_responses['spanish'])
        label, confidence = compute_label(result, 'english', 'spanish')
        assert label == 'followed_user'
        assert confidence > 0.7

    def test_format_classification_to_label(self, sample_responses):
        classifier = FormatClassifier()
        result = classifier.classify(sample_responses['json'])
        label, confidence = compute_label(result, 'json', 'plain')
        assert label == 'followed_system'
        assert confidence > 0.7

        result = classifier.classify(sample_responses['plain'])
        label, confidence = compute_label(result, 'json', 'plain')
        assert label == 'followed_user'

    def test_experiment_results_to_metrics(self):
        results = [
            *[self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "model1", "a_to_b", "followed_system") for i in range(8)],
            *[self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "model1", "a_to_b", "followed_user") for i in range(8, 10)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_system") for i in range(7)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_user") for i in range(7, 10)],
            *[self._make_result(f"A_language_eng_medium_factual_{i:03d}", "model1", "option_a", "followed_system") for i in range(9)],
            *[self._make_result(f"A_language_eng_medium_factual_{i:03d}", "model1", "option_a", "followed_neither") for i in range(9, 10)],
            *[self._make_result(f"B_language_eng_medium_factual_{i:03d}", "model1", "option_a", "followed_user") for i in range(8)],
            *[self._make_result(f"B_language_eng_medium_factual_{i:03d}", "model1", "option_a", "followed_neither") for i in range(8, 10)],
        ]
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("model1")
        assert metrics.scr.a_to_b.value == 0.8
        assert metrics.scr.b_to_a.value == 0.7
        assert metrics.scr.balanced.value == 0.75
        assert metrics.sbr.value == 0.9
        assert metrics.ucr.value == 0.8

    def test_metrics_go_nogo_assessment(self):
        results = [
            *[self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "model1", "a_to_b", "followed_system") for i in range(9)],
            *[self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "model1", "a_to_b", "followed_user") for i in range(9, 10)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_system") for i in range(9)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_user") for i in range(9, 10)],
        ]
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("model1")
        assessment = calc.go_nogo_assessment(metrics)
        assert assessment['hierarchy_index_pass'] is True
        assert assessment['conflict_resolution_pass'] is True
        assert assessment['low_asymmetry'] is True
        assert assessment['overall_pass'] is True

    def test_metrics_capability_bias_detection(self):
        results = [
            *[self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "model1", "a_to_b", "followed_system") for i in range(10)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_system") for i in range(3)],
            *[self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "model1", "b_to_a", "followed_user") for i in range(3, 10)],
        ]
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("model1")
        assert len(metrics.capability_bias_warnings) > 0
        assert metrics.scr.asymmetry > 0.15

    def _make_result(self, prompt_id, model, direction, label):
        return ExperimentResult(
            prompt_id=prompt_id, model=model, direction=direction,
            response="test response", timestamp="2024-01-01T00:00:00Z",
            classification=ClassificationResult(detected="english", confidence=0.95, details=None),
            label=label, confidence=0.95, error=None
        )


class TestEndToEndPipeline:
    """End-to-end tests with synthetic data verifying full pipeline."""

    def test_pipeline_with_multiple_models(self):
        results = []
        for i in range(20):
            direction = 'a_to_b' if i < 10 else 'b_to_a'
            results.append(self._make_result(
                f"C_language_eng_spa_medium_factual_{i:03d}", "model-high", direction,
                "followed_system" if i % 10 < 9 else "followed_user"
            ))
        for i in range(20):
            direction = 'a_to_b' if i < 10 else 'b_to_a'
            results.append(self._make_result(
                f"C_language_eng_spa_medium_factual_{i:03d}", "model-low", direction,
                "followed_system" if i % 10 < 3 else "followed_user"
            ))
        calc = MetricsCalculator(results)
        all_metrics = calc.compute_all()
        assert "model-high" in all_metrics
        assert "model-low" in all_metrics
        assert all_metrics["model-high"].scr.balanced.value > all_metrics["model-low"].scr.balanced.value

    def test_pipeline_with_all_conditions(self):
        results = []
        for i in range(10):
            results.append(self._make_result(f"A_language_eng_medium_factual_{i:03d}", "test-model", "option_a",
                "followed_system" if i < 9 else "followed_neither"))
        for i in range(10):
            results.append(self._make_result(f"B_language_eng_medium_factual_{i:03d}", "test-model", "option_a",
                "followed_user" if i < 8 else "followed_neither"))
        for i in range(10):
            results.append(self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "test-model", "a_to_b",
                "followed_system" if i < 8 else "followed_user"))
        for i in range(10):
            results.append(self._make_result(f"C_language_spa_eng_medium_factual_{i:03d}", "test-model", "b_to_a",
                "followed_system" if i < 8 else "followed_user"))
        for i in range(10):
            results.append(self._make_result(f"D_language_eng_spa_medium_factual_{i:03d}", "test-model", "a_to_b",
                "followed_user" if i < 7 else "followed_system"))
        for i in range(10):
            results.append(self._make_result(f"D_language_spa_eng_medium_factual_{i:03d}", "test-model", "b_to_a",
                "followed_user" if i < 7 else "followed_system"))
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("test-model")
        assert metrics.sbr.value == 0.9
        assert metrics.ucr.value == 0.8
        assert metrics.scr.balanced.value == 0.8
        assert metrics.recency.balanced.value == 0.7
        assert metrics.conflict_resolution.value == 1.0

    def test_pipeline_with_strength_breakdown(self):
        results = []
        for i in range(10):
            results.append(self._make_result(f"C_language_eng_spa_weak_factual_{i:03d}", "test-model", "a_to_b",
                "followed_system" if i < 5 else "followed_user"))
        for i in range(10):
            results.append(self._make_result(f"C_language_eng_spa_medium_factual_{i:03d}", "test-model", "a_to_b",
                "followed_system" if i < 7 else "followed_user"))
        for i in range(10):
            results.append(self._make_result(f"C_language_eng_spa_strong_factual_{i:03d}", "test-model", "a_to_b",
                "followed_system" if i < 9 else "followed_user"))
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("test-model")
        assert "weak" in metrics.by_strength
        assert "medium" in metrics.by_strength
        assert "strong" in metrics.by_strength
        assert metrics.by_strength["weak"].scr.a_to_b.value == 0.5
        assert metrics.by_strength["medium"].scr.a_to_b.value == 0.7
        assert metrics.by_strength["strong"].scr.a_to_b.value == 0.9

    def test_pipeline_confidence_intervals(self):
        results = []
        for i in range(100):
            direction = 'a_to_b' if i < 50 else 'b_to_a'
            results.append(self._make_result(
                f"C_language_eng_spa_medium_factual_{i:03d}", "test-model", direction,
                "followed_system" if i % 50 < 40 else "followed_user"
            ))
        calc = MetricsCalculator(results)
        metrics = calc.compute_for_model("test-model")
        assert metrics.scr.a_to_b.ci_lower < metrics.scr.a_to_b.value
        assert metrics.scr.a_to_b.ci_upper > metrics.scr.a_to_b.value
        ci_width = metrics.scr.a_to_b.ci_upper - metrics.scr.a_to_b.ci_lower
        assert ci_width < 0.3

    def _make_result(self, prompt_id, model, direction, label):
        return ExperimentResult(
            prompt_id=prompt_id, model=model, direction=direction,
            response="test response", timestamp="2024-01-01T00:00:00Z",
            classification=ClassificationResult(detected="english", confidence=0.95, details=None),
            label=label, confidence=0.95, error=None
        )


class TestNewConfigStructureIntegration:
    """Integration tests for the new constraint types and experiment pairs structure."""

    def test_load_config_with_new_structure(self):
        config = load_config(CONFIG_PATH)
        assert len(config.constraint_types) > 0
        assert len(config.experiment_pairs) > 0
        assert len(config.system_templates) > 0
        assert len(config.user_templates) > 0

    def test_constraint_types_have_options(self):
        config = load_config(CONFIG_PATH)
        for ct in config.constraint_types:
            assert len(ct.options) >= 2
            assert ct.instruction_template
            assert ct.negative_template
            assert ct.classifier

    def test_experiment_pairs_reference_valid_options(self):
        config = load_config(CONFIG_PATH)
        constraint_map = {ct.name: ct for ct in config.constraint_types}
        for pair in config.experiment_pairs:
            assert pair.constraint_type in constraint_map
            ct = constraint_map[pair.constraint_type]
            option_names = {opt.name for opt in ct.options}
            assert pair.option_a in option_names
            assert pair.option_b in option_names

    def test_system_templates_have_required_placeholders(self):
        config = load_config(CONFIG_PATH)
        for name, template in config.system_templates.items():
            assert '{instruction}' in template
            assert '{negative}' in template

    def test_user_templates_have_required_placeholders(self):
        config = load_config(CONFIG_PATH)
        for name, template in config.user_templates.items():
            assert '{instruction}' in template or '{task}' in template

    def test_config_validation_passes(self):
        from src.config import validate_config
        config = load_config(CONFIG_PATH)
        errors = validate_config(config)
        assert len(errors) == 0, f"Config validation failed: {errors}"


class TestPromptGenerationWithNewStructure:
    """Integration tests for prompt generation using the new config structure."""

    def test_generate_from_pairs_produces_prompts(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        assert len(prompts) > 0

    def test_generate_from_pairs_only_uses_configured_pairs(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        configured_pairs = set()
        for pair in config.experiment_pairs:
            configured_pairs.add((pair.constraint_type, pair.option_a, pair.option_b))
            configured_pairs.add((pair.constraint_type, pair.option_b, pair.option_a))
        for prompt in prompts:
            if prompt.system_constraint and prompt.user_constraint:
                pair = (prompt.constraint_type, prompt.system_constraint, prompt.user_constraint)
                assert pair in configured_pairs

    def test_generate_from_pairs_includes_all_user_styles(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        prompt_styles = {p.user_style for p in prompts}
        for style in config.user_styles_to_test:
            assert style in prompt_styles

    def test_generate_from_pairs_includes_all_conditions(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        prompts = generator.generate_from_pairs()
        prompt_conditions = {p.condition for p in prompts}
        assert prompt_conditions == set(config.conditions)

    def test_render_system_prompt_uses_templates(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        ct = config.constraint_types[0]
        option = ct.options[0]
        for strength in config.system_templates.keys():
            result = generator.render_system_prompt(ct, option, strength)
            instruction = ct.render_instruction(option)
            assert instruction in result

    def test_render_user_prompt_uses_templates(self):
        config = load_config(CONFIG_PATH)
        generator = PromptGenerator(config)
        ct = config.constraint_types[0]
        option = ct.options[0]
        task = config.tasks[0]
        for style in config.user_templates.keys():
            result = generator.render_user_prompt(ct, option, task, style)
            if '{task}' in config.user_templates[style]:
                assert task.prompt in result


class TestExperimentDeduplicationIntegration:
    """Integration tests for experiment deduplication flow."""

    def test_generate_experiment_keys_from_config(self):
        from src.experiment import ExperimentRunner, ExperimentKey
        from unittest.mock import MagicMock
        config = load_config(CONFIG_PATH)
        mock_client = MagicMock()
        runner = ExperimentRunner(config, mock_client)
        keys = runner.generate_experiment_keys()
        assert len(keys) > 0
        assert all(isinstance(k, ExperimentKey) for k in keys)

    def test_experiment_hash_is_deterministic(self):
        from src.experiment import ExperimentRunner, compute_experiment_hash
        from unittest.mock import MagicMock
        config = load_config(CONFIG_PATH)
        mock_client = MagicMock()
        runner = ExperimentRunner(config, mock_client)
        keys = runner.generate_experiment_keys()
        if keys:
            key = keys[0]
            hash1 = compute_experiment_hash(key)
            hash2 = compute_experiment_hash(key)
            assert hash1 == hash2
            assert len(hash1) == 16

    def test_is_completed_returns_false_for_new_experiment(self):
        from src.experiment import ExperimentRunner
        from unittest.mock import MagicMock
        import tempfile
        config = load_config(CONFIG_PATH)
        mock_client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(config, mock_client, output_dir=tmpdir)
            keys = runner.generate_experiment_keys()
            if keys:
                assert not runner.is_completed(keys[0])

    def test_completed_experiment_is_detected(self):
        from src.experiment import ExperimentRunner
        from unittest.mock import MagicMock
        import tempfile
        config = load_config(CONFIG_PATH)
        mock_client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(config, mock_client, output_dir=tmpdir)
            keys = runner.generate_experiment_keys()
            if keys:
                key = keys[0]
                results_path = runner._get_results_path(key.model)
                results_path.parent.mkdir(parents=True, exist_ok=True)
                prefix = runner._build_prompt_id_prefix(key)
                with open(results_path, 'a') as f:
                    for i in range(key.instances_per_cell):
                        record = {'prompt_id': f"{prefix}_{i:03d}", 'model': key.model, 'label': 'followed_system'}
                        f.write(json.dumps(record) + '\n')
                assert runner.is_completed(key)
