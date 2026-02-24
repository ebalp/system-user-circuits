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
        api=ApiConfig(timeout=10, max_retries=1),
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
    """Extract the sample count for a given condition from the rendered HTML table.

    Table columns are: Condition | Formula | Count
    """
    # Match the last <td> in the row: <td>A</td><td>formula</td><td>COUNT</td>
    pattern = rf"<td>{condition}</td>\s*<td>.*?</td>\s*<td>(\d+)</td>"
    match = re.search(pattern, html, re.DOTALL)
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
        """Condition D: n_pairs × 2 × n_tasks (fixed style, no user_styles multiplier)"""
        config = _make_config(n_pairs=3, n_strengths=2, n_user_styles=3, n_tasks=4)
        html = _render_experiment_design(config, ["model_a"])
        assert _extract_condition_count(html, "D") == 3 * 2 * 4


class TestSampleCountDerivationText:
    """Verify the derivation text in the HTML table uses the correct formula descriptions."""

    def test_condition_a_derivation_mentions_dirs_not_styles(self):
        """Condition A formula should say '2 dirs' and not mention user styles."""
        config = _make_config(n_pairs=2, n_strengths=1, n_user_styles=2, n_tasks=3)
        html = _render_experiment_design(config, ["model_a"])
        match = re.search(r"<td>A</td>\s*<td>(.*?)</td>", html, re.DOTALL)
        assert match
        formula = match.group(1)
        assert "2 dirs" in formula
        assert "user style" not in formula.lower()

    def test_condition_b_derivation_mentions_dirs_not_styles(self):
        """Condition B formula should say '2 dirs' and not mention user styles."""
        config = _make_config(n_pairs=2, n_strengths=1, n_user_styles=2, n_tasks=3)
        html = _render_experiment_design(config, ["model_a"])
        match = re.search(r"<td>B</td>\s*<td>(.*?)</td>", html, re.DOTALL)
        assert match
        formula = match.group(1)
        assert "2 dirs" in formula
        assert "user style" not in formula.lower()

    def test_condition_c_derivation_mentions_directions_and_styles(self):
        """Condition C formula should mention directions, strengths, and user styles."""
        config = _make_config(n_pairs=2, n_strengths=3, n_user_styles=2, n_tasks=1)
        html = _render_experiment_design(config, ["model_a"])
        match = re.search(r"<td>C</td>\s*<td>(.*?)</td>", html, re.DOTALL)
        assert match
        formula = match.group(1)
        assert "2 dirs" in formula
        assert "3 strengths" in formula
        assert "2 styles" in formula
