"""Test configuration loading and validation."""

import os
import tempfile
import pytest
import yaml
from pathlib import Path
from phase0_v2.src.config import load_config, ModelConfig


@pytest.fixture
def config():
    return load_config("phase0_v2/config/experiment.yaml")


class TestConfigLoading:
    def test_loads_without_error(self, config):
        assert config is not None

    def test_models_present(self, config):
        assert len(config.models) >= 1
        assert isinstance(config.models[0], ModelConfig)
        assert isinstance(config.models[0].id, str)

    def test_tasks_loaded(self, config):
        assert len(config.tasks) >= 50, f"Expected ~50 tasks, got {len(config.tasks)}"

    def test_task_ids_unique(self, config):
        ids = [t.id for t in config.tasks]
        assert len(ids) == len(set(ids)), f"Duplicate task IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_task_prompts_nonempty(self, config):
        for t in config.tasks:
            assert t.prompt.strip(), f"Task {t.id} has empty prompt"


class TestTemplateValidation:
    def test_system_templates_have_placeholder(self, config):
        for name, tmpl in config.system_templates.items():
            assert "{system_instruction}" in tmpl.template, \
                f"System template '{name}' missing {{system_instruction}} placeholder"

    def test_user_templates_have_placeholders(self, config):
        for name, tmpl in config.user_templates.items():
            assert "{user_instruction}" in tmpl.template, \
                f"User template '{name}' missing {{user_instruction}} placeholder"
            assert "{task}" in tmpl.template, \
                f"User template '{name}' missing {{task}} placeholder"

    def test_system_styles_exist_in_system_templates(self, config):
        for style in config.condition_c_system_styles:
            assert style in config.system_templates, \
                f"System style '{style}' not in system_templates"

    def test_styles_exist_in_user_templates(self, config):
        for style in config.user_styles_to_test:
            assert style in config.user_templates, \
                f"Style '{style}' not in user_templates"

    def test_default_system_style_valid(self, config):
        assert config.default_system_style in config.system_templates

    def test_default_user_style_valid(self, config):
        assert config.default_user_style in config.user_templates


class TestGenerationConfig:
    def test_temperature_range(self, config):
        assert 0.0 <= config.generation.temperature <= 2.0

    def test_max_tokens_positive(self, config):
        assert config.generation.max_tokens > 0

    def test_instances_per_cell_positive(self, config):
        assert config.generation.instances_per_cell >= 1


class TestConditions:
    def test_all_four_conditions(self, config):
        assert set(config.conditions) == {"A", "B", "C", "D"}

    def test_five_system_styles(self, config):
        assert set(config.condition_c_system_styles) == {"bare", "compliance", "authority", "persona", "safety"}

    def test_five_user_styles(self, config):
        assert len(config.user_styles_to_test) == 5


def _make_test_config(models_section, tmpdir):
    """Create a minimal experiment YAML with the given models section.

    Returns the path to the config file.
    """
    # Create a tasks file
    tasks_file = os.path.join(tmpdir, "tasks.yaml")
    with open(tasks_file, "w") as f:
        yaml.dump({"tasks": [{"id": "t1", "prompt": "Hello."}]}, f)

    config_data = {
        "models": models_section,
        "api": {"timeout": 60, "max_retries": 3, "concurrent_per_model": 1},
        "system_templates": {"bare": "{system_instruction}"},
        "user_templates": {"with_instruction": "{user_instruction}\n\n{task}"},
        "conditions": ["A", "B", "C", "D"],
        "condition_c_system_styles": ["bare"],
        "user_styles_to_test": ["with_instruction"],
        "default_system_style": "bare",
        "default_user_style": "with_instruction",
        "counterbalancing": {"enabled": False},
        "generation": {"temperature": 0.0, "max_tokens": 512, "instances_per_cell": 1},
        "task_sources": {"synthetic": {"file": "tasks.yaml"}},
        "thresholds": {"hierarchy_index": 0.7, "conflict_resolution": 0.8, "asymmetry_warning": 0.15},
        "seed": 42,
    }
    config_path = os.path.join(tmpdir, "experiment.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestModelConfig:
    def test_string_format_backward_compatible(self, tmp_path):
        """Plain string model ID creates ModelConfig with defaults."""
        config_path = _make_test_config(["model-a", "model-b"], str(tmp_path))
        config = load_config(config_path)
        assert len(config.models) == 2
        assert isinstance(config.models[0], ModelConfig)
        assert config.models[0].id == "model-a"
        assert config.models[0].exclude_conflicts == []

    def test_dict_format(self, tmp_path):
        """Dict format with exclude_conflicts."""
        models = [
            {
                "id": "model-a",
                "exclude_conflicts": ["word_repetition_density"],
            }
        ]
        config_path = _make_test_config(models, str(tmp_path))
        config = load_config(config_path)
        assert config.models[0].id == "model-a"
        assert config.models[0].exclude_conflicts == ["word_repetition_density"]

    def test_dict_format_minimal(self, tmp_path):
        """Dict with only id, no excludes."""
        models = [{"id": "model-a"}]
        config_path = _make_test_config(models, str(tmp_path))
        config = load_config(config_path)
        assert config.models[0].id == "model-a"
        assert config.models[0].exclude_conflicts == []

    def test_mixed_format(self, tmp_path):
        """Mix of string and dict model entries."""
        models = [
            "model-a",
            {"id": "model-b", "exclude_conflicts": ["emoji_use_vs_avoid"]},
            "model-c",
        ]
        config_path = _make_test_config(models, str(tmp_path))
        config = load_config(config_path)
        assert len(config.models) == 3
        assert config.models[0].id == "model-a"
        assert config.models[1].id == "model-b"
        assert config.models[1].exclude_conflicts == ["emoji_use_vs_avoid"]
        assert config.models[2].id == "model-c"

    def test_invalid_model_entry(self, tmp_path):
        """Non-string, non-dict raises ValueError."""
        models = [42]
        config_path = _make_test_config(models, str(tmp_path))
        with pytest.raises(ValueError, match="Invalid model entry type"):
            load_config(config_path)

    def test_dict_missing_id_raises(self, tmp_path):
        """Dict without 'id' key raises ValueError."""
        models = [{"thresholds": {"foo": 0.5}}]
        config_path = _make_test_config(models, str(tmp_path))
        with pytest.raises(ValueError, match="Model entry missing 'id'"):
            load_config(config_path)

    def test_dict_model_has_excludes(self, tmp_path):
        """Dict model entry with exclude_conflicts is parsed correctly."""
        models = [
            {
                "id": "my-model",
                "exclude_conflicts": ["word_repetition_density"],
            }
        ]
        config_path = _make_test_config(models, str(tmp_path))
        config = load_config(config_path)
        mc = config.models[0]
        assert mc.id == "my-model"
        assert "word_repetition_density" in mc.exclude_conflicts

    def test_plain_string_model_has_empty_defaults(self, tmp_path):
        """Models without explicit config should have empty defaults."""
        models = [
            {"id": "model-with-config", "exclude_conflicts": ["foo"]},
            "model-without-config",
        ]
        config_path = _make_test_config(models, str(tmp_path))
        config = load_config(config_path)
        mc = next(m for m in config.models if m.id == "model-without-config")
        assert mc.exclude_conflicts == []
