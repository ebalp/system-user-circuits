"""
Unit tests for sample count derivation in _render_experiment_design().

Feature: fix-condition-ab-keys
Requirements: 7.1, 7.2, 7.3, 7.4
"""

import re

from src.config import (
    ExperimentConfig,
    ConstraintType,
    ConstraintOption,
    ExperimentPair,
    Task,
    ApiConfig,
    GenerationConfig,
    CounterbalancingConfig,
)
from src.report.layout import _render_experiment_design


def _make_config(n_pairs: int, n_strengths: int, n_user_styles: int, n_tasks: int) -> ExperimentConfig:
    """Build a minimal ExperimentConfig with the given combinatorial dimensions."""
    # Constraint options and pairs
    constraint_types = []
    pairs = []
    for i in range(n_pairs):
        opt_a = ConstraintOption(name=f"opt_a_{i}", value=f"val_a_{i}", expected_value=f"exp_a_{i}")
        opt_b = ConstraintOption(name=f"opt_b_{i}", value=f"val_b_{i}", expected_value=f"exp_b_{i}")
        ct = ConstraintType(
            name=f"ct_{i}",
            instruction_template="Please {value}",
            negative_template="Do not {value}",
            classifier="language",
            options=[opt_a, opt_b],
        )
        constraint_types.append(ct)
        pairs.append(ExperimentPair(constraint_type=f"ct_{i}", option_a=f"opt_a_{i}", option_b=f"opt_b_{i}"))

    # Strength templates
    strength_names = [f"strength_{i}" for i in range(n_strengths)]
    system_templates = {name: f"System {name} {{instruction}} {{negative}}" for name in strength_names}

    # User style templates
    style_names = [f"style_{i}" for i in range(n_user_styles)]
    user_templates = {name: f"User {name} {{instruction}} {{task}}" for name in style_names}

    tasks = [Task(id=f"task_{i}", prompt=f"Do task {i}") for i in range(n_tasks)]

    return ExperimentConfig(
        api=ApiConfig(token_file="tok.txt", key_env_var="KEY", timeout=10, max_retries=1),
        models=["model_a"],
        constraint_types=constraint_types,
        experiment_pairs=pairs,
        system_templates=system_templates,
        user_templates=user_templates,
        tasks=tasks,
        conditions=["A", "B", "C", "D"],
        counterbalancing=CounterbalancingConfig(enabled=True),
        generation=GenerationConfig(temperature=0.0, max_tokens=512, instances_per_cell=1),
        condition_c_strengths=strength_names,
        default_strength=strength_names[0],
        default_user_style=style_names[0],
        user_styles_to_test=style_names,
    )


def _extract_condition_count(html: str, condition: str) -> int:
    """Extract the sample count for a given condition from the rendered HTML table."""
    # Pattern: <td>A</td><td>42</td><td>derivation text</td>
    pattern = rf"<td>{condition}</td>\s*<td>(\d+)</td>"
    match = re.search(pattern, html)
    assert match, f"Could not find count for condition {condition} in HTML"
    return int(match.group(1))


class TestSampleCountFormulas:
    """Verify computed sample counts match expected formulas for all four conditions."""

    def test_condition_a_count(self):
        """Condition A: n_pairs × 2 × n_tasks"""
        config = _make_config(n_pairs=3, n_strengths=2, n_user_styles=3, n_tasks=4)
        html = _render_experiment_design(config, ["model_a"])
        assert _extract_condition_count(html, "A") == 3 * 2 * 4

    def test_condition_b_count(self):
        """Condition B: n_pairs × 2 × n_tasks"""
        config = _make_config(n_pairs=3, n_strengths=2, n_user_styles=3, n_tasks=4)
        html = _render_experiment_design(config, ["model_a"])
        assert _extract_condition_count(html, "B") == 3 * 2 * 4

    def test_condition_c_count(self):
        """Condition C: n_pairs × 2 × n_strengths × n_user_styles × n_tasks (unchanged)"""
        config = _make_config(n_pairs=3, n_strengths=2, n_user_styles=3, n_tasks=4)
        html = _render_experiment_design(config, ["model_a"])
        assert _extract_condition_count(html, "C") == 3 * 2 * 2 * 3 * 4

    def test_condition_d_count(self):
        """Condition D: n_pairs × 2 × n_user_styles × n_tasks (unchanged)"""
        config = _make_config(n_pairs=3, n_strengths=2, n_user_styles=3, n_tasks=4)
        html = _render_experiment_design(config, ["model_a"])
        assert _extract_condition_count(html, "D") == 3 * 2 * 3 * 4


class TestSampleCountDerivationText:
    """Verify the derivation text in the HTML table uses the correct formula descriptions."""

    def test_condition_a_derivation_mentions_options(self):
        """Condition A derivation should say '2 options', not 'user styles'."""
        config = _make_config(n_pairs=2, n_strengths=1, n_user_styles=2, n_tasks=3)
        html = _render_experiment_design(config, ["model_a"])
        # Find the A row derivation
        match = re.search(r"<td>A</td>\s*<td>\d+</td>\s*<td>(.*?)</td>", html)
        assert match
        derivation = match.group(1)
        assert "2 options" in derivation
        assert "user style" not in derivation.lower()

    def test_condition_b_derivation_mentions_options(self):
        """Condition B derivation should say '2 options', not 'user styles'."""
        config = _make_config(n_pairs=2, n_strengths=1, n_user_styles=2, n_tasks=3)
        html = _render_experiment_design(config, ["model_a"])
        match = re.search(r"<td>B</td>\s*<td>\d+</td>\s*<td>(.*?)</td>", html)
        assert match
        derivation = match.group(1)
        assert "2 options" in derivation
        assert "user style" not in derivation.lower()

    def test_condition_c_derivation_mentions_directions_and_styles(self):
        """Condition C derivation should mention directions, strengths, and user styles."""
        config = _make_config(n_pairs=2, n_strengths=3, n_user_styles=2, n_tasks=1)
        html = _render_experiment_design(config, ["model_a"])
        match = re.search(r"<td>C</td>\s*<td>\d+</td>\s*<td>(.*?)</td>", html)
        assert match
        derivation = match.group(1)
        assert "2 directions" in derivation
        assert "3 strengths" in derivation
        assert "2 user styles" in derivation
