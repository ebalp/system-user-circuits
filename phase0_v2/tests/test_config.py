"""Test configuration loading and validation."""

import pytest
from phase0_v2.src.config import load_config


@pytest.fixture
def config():
    return load_config("phase0_v2/config/experiment.yaml")


class TestConfigLoading:
    def test_loads_without_error(self, config):
        assert config is not None

    def test_models_present(self, config):
        assert len(config.models) >= 1

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
