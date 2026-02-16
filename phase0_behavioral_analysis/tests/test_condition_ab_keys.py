"""
Property-based tests for Condition A/B key generation fixes.

Feature: fix-condition-ab-keys
"""

import tempfile
from dataclasses import dataclass

from hypothesis import given, strategies as st, settings

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
from src.experiment import ExperimentRunner


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Simple identifier strategy for names/ids
_ident = st.from_regex(r"[a-z][a-z0-9]{0,9}", fullmatch=True)

# Strategy for non-empty printable text (option values, prompts, etc.)
_text = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=["L", "Nd", "Zs"]),
).filter(lambda s: s.strip())


def _user_styles_strategy():
    """Generate a list of 2-4 distinct user style names, always including a default."""
    return st.lists(
        _ident,
        min_size=2,
        max_size=4,
        unique=True,
    )


@st.composite
def experiment_config_strategy(draw):
    """Generate a valid ExperimentConfig with at least one model, pair, task, and multiple user styles."""
    # User styles: pick 2-4 distinct names; first one is the default
    user_style_names = draw(_user_styles_strategy())
    default_user_style = user_style_names[0]

    # Build user templates dict (style_name -> template content)
    user_templates = {name: f"Template for {name}: {{instruction}} {{task}}" for name in user_style_names}

    # Constraint type with two options
    opt_a_name = draw(_ident)
    opt_b_name = draw(_ident.filter(lambda s: s != opt_a_name))
    opt_a = ConstraintOption(name=opt_a_name, value=draw(_text), expected_value=draw(_text))
    opt_b = ConstraintOption(name=opt_b_name, value=draw(_text), expected_value=draw(_text))
    ct_name = draw(_ident)
    ct = ConstraintType(
        name=ct_name,
        instruction_template="Please {value}",
        negative_template="Do not {value}",
        classifier="language",
        options=[opt_a, opt_b],
    )

    pair = ExperimentPair(constraint_type=ct_name, option_a=opt_a_name, option_b=opt_b_name)

    # 1-2 models, 1-2 tasks
    models = draw(st.lists(_ident, min_size=1, max_size=2, unique=True))
    tasks = [Task(id=draw(_ident), prompt=draw(_text)) for _ in range(draw(st.integers(min_value=1, max_value=2)))]

    # Strengths
    default_strength = "medium"
    system_templates = {"medium": "System template {instruction} {negative}"}

    config = ExperimentConfig(
        api=ApiConfig(token_file="tok.txt", key_env_var="KEY", timeout=10, max_retries=1),
        models=models,
        constraint_types=[ct],
        experiment_pairs=[pair],
        system_templates=system_templates,
        user_templates=user_templates,
        tasks=tasks,
        conditions=["A", "B", "C", "D"],
        counterbalancing=CounterbalancingConfig(enabled=True),
        generation=GenerationConfig(temperature=0.0, max_tokens=512, instances_per_cell=1),
        condition_c_strengths=["medium"],
        default_strength=default_strength,
        default_user_style=default_user_style,
        user_styles_to_test=user_style_names,
    )
    return config


# =============================================================================
# Property 1: Condition A/B keys use only default user style
# Validates: Requirements 3.1, 3.2
# =============================================================================


class TestConditionABDefaultUserStyle:
    """Property 1: Condition A/B keys use only default user style."""

    @settings(max_examples=100)
    @given(config=experiment_config_strategy())
    def test_condition_ab_keys_use_default_user_style(self, config: ExperimentConfig):
        """
        **Validates: Requirements 3.1, 3.2**

        For any valid config, all keys with condition A or B must have
        user_template_name equal to config.default_user_style.
        """
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MagicMock()
            runner = ExperimentRunner(config, client, output_dir=tmp_dir)
            keys = runner.generate_experiment_keys()

        ab_keys = [k for k in keys if k.condition in ("A", "B")]
        assert len(ab_keys) > 0, "Expected at least one A/B key"

        for key in ab_keys:
            assert key.user_template_name == config.default_user_style, (
                f"Condition {key.condition} key has user_template_name="
                f"'{key.user_template_name}', expected '{config.default_user_style}'"
            )


# =============================================================================
# Property 2: Condition C/D keys cover all configured user styles
# Validates: Requirements 3.3, 3.4
# =============================================================================


class TestConditionCDAllUserStyles:
    """Property 2: Condition C/D keys cover all configured user styles."""

    @settings(max_examples=100)
    @given(config=experiment_config_strategy())
    def test_condition_cd_keys_cover_all_user_styles(self, config: ExperimentConfig):
        """
        **Validates: Requirements 3.3, 3.4**

        For any valid config with at least one model, pair, and task,
        the set of user_template_name values across all C (or D) keys
        must equal the configured user_styles_to_test set.
        """
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MagicMock()
            runner = ExperimentRunner(config, client, output_dir=tmp_dir)
            keys = runner.generate_experiment_keys()

        expected_styles = set(config.user_styles_to_test)

        for cond in ("C", "D"):
            cond_keys = [k for k in keys if k.condition == cond]
            assert len(cond_keys) > 0, f"Expected at least one {cond} key"
            actual_styles = {k.user_template_name for k in cond_keys}
            assert actual_styles == expected_styles, (
                f"Condition {cond} keys have user styles {actual_styles}, "
                f"expected {expected_styles}"
            )


# =============================================================================
# Property 3: Condition A system message uses direction-correct option value
# Validates: Requirements 4.1, 4.2
# =============================================================================


class TestConditionASystemMessageDirection:
    """Property 3: Condition A system message uses direction-correct option value."""

    @settings(max_examples=100)
    @given(config=experiment_config_strategy())
    def test_condition_a_system_message_uses_correct_option(self, config: ExperimentConfig):
        """
        **Validates: Requirements 4.1, 4.2**

        For any ExperimentKey with condition A, the rendered system message
        should contain option_a_value when direction is option_a, and
        option_b_value when direction is option_b.
        """
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MagicMock()
            runner = ExperimentRunner(config, client, output_dir=tmp_dir)
            keys = runner.generate_experiment_keys()

        cond_a_keys = [k for k in keys if k.condition == "A"]
        assert len(cond_a_keys) > 0, "Expected at least one Condition A key"

        for key in cond_a_keys:
            msg = runner._render_system_message(key)

            if key.direction == "option_a":
                assert key.option_a_value in msg, (
                    f"Condition A with direction=option_a: system message should "
                    f"contain '{key.option_a_value}', got: {msg}"
                )
            elif key.direction == "option_b":
                assert key.option_b_value in msg, (
                    f"Condition A with direction=option_b: system message should "
                    f"contain '{key.option_b_value}', got: {msg}"
                )


    # =============================================================================
    # Property 4: Condition B user message uses direction-correct option value
    # Validates: Requirements 5.1, 5.2
    # =============================================================================


    class TestConditionBUserMessageDirection:
        """Property 4: Condition B user message uses direction-correct option value."""

        @settings(max_examples=100)
        @given(config=experiment_config_strategy())
        def test_condition_b_user_message_uses_correct_option(self, config: ExperimentConfig):
            """
            **Validates: Requirements 5.1, 5.2**

            For any ExperimentKey with condition B, the rendered user message
            should contain option_a_value when direction is option_a, and
            option_b_value when direction is option_b.
            """
            from unittest.mock import MagicMock

            with tempfile.TemporaryDirectory() as tmp_dir:
                client = MagicMock()
                runner = ExperimentRunner(config, client, output_dir=tmp_dir)
                keys = runner.generate_experiment_keys()

            cond_b_keys = [k for k in keys if k.condition == "B"]
            assert len(cond_b_keys) > 0, "Expected at least one Condition B key"

            for key in cond_b_keys:
                msg = runner._render_user_message(key)

                if key.direction == "option_a":
                    assert key.option_a_value in msg, (
                        f"Condition B with direction=option_a: user message should "
                        f"contain '{key.option_a_value}', got: {msg}"
                    )
                elif key.direction == "option_b":
                    assert key.option_b_value in msg, (
                        f"Condition B with direction=option_b: user message should "
                        f"contain '{key.option_b_value}', got: {msg}"
                    )


        # =============================================================================
        # Property 5: Condition B user message applies user template
        # Validates: Requirements 5.3
        # =============================================================================


        class TestConditionBUserMessageTemplate:
            """Property 5: Condition B user message applies user template."""

            @settings(max_examples=100)
            @given(config=experiment_config_strategy())
            def test_condition_b_user_message_applies_template(self, config: ExperimentConfig):
                """
                **Validates: Requirements 5.3**

                For any ExperimentKey with condition B that has non-empty
                user_template_content, the rendered user message should match
                the template applied with the constraint instruction and task prompt.
                """
                from unittest.mock import MagicMock

                with tempfile.TemporaryDirectory() as tmp_dir:
                    client = MagicMock()
                    runner = ExperimentRunner(config, client, output_dir=tmp_dir)
                    keys = runner.generate_experiment_keys()

                cond_b_keys = [k for k in keys if k.condition == "B"]
                assert len(cond_b_keys) > 0, "Expected at least one Condition B key"

                for key in cond_b_keys:
                    msg = runner._render_user_message(key)

                    # Determine expected instruction value based on direction
                    if key.direction == "option_a":
                        instruction_value = key.option_a_value
                    elif key.direction == "option_b":
                        instruction_value = key.option_b_value
                    else:
                        instruction_value = key.option_a_value

                    instruction = key.instruction_template.format(value=instruction_value)

                    if key.user_template_content:
                        expected = key.user_template_content.format(
                            instruction=instruction,
                            task=key.task_prompt,
                        )
                        assert msg == expected, (
                            f"Condition B user message does not match template.\n"
                            f"Expected: {expected}\nGot: {msg}"
                        )



# =============================================================================
# Property 6: Key count formula correctness
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# =============================================================================


class TestKeyCountFormulaCorrectness:
    """Property 6: Key count formula correctness."""

    @settings(max_examples=100)
    @given(config=experiment_config_strategy())
    def test_key_count_formula_correctness(self, config: ExperimentConfig):
        """
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

        For any valid config, the number of generated keys per condition
        must match the combinatorial formulas:
        - A: n_models × n_pairs × 2 × n_tasks
        - B: n_models × n_pairs × 2 × n_tasks
        - C: n_models × n_pairs × 2 × n_strengths × n_user_styles × n_tasks
        - D: n_models × n_pairs × 2 × n_user_styles × n_tasks
        """
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = MagicMock()
            runner = ExperimentRunner(config, client, output_dir=tmp_dir)
            keys = runner.generate_experiment_keys()

        n_models = len(config.models)
        n_pairs = len(config.experiment_pairs)
        n_tasks = len(config.tasks)
        n_strengths = len(config.condition_c_strengths)
        n_user_styles = len(config.user_styles_to_test)

        keys_by_condition = {}
        for k in keys:
            keys_by_condition.setdefault(k.condition, []).append(k)

        expected_a = n_models * n_pairs * 2 * n_tasks
        expected_b = n_models * n_pairs * 2 * n_tasks
        expected_c = n_models * n_pairs * 2 * n_strengths * n_user_styles * n_tasks
        expected_d = n_models * n_pairs * 2 * n_user_styles * n_tasks

        actual_a = len(keys_by_condition.get("A", []))
        actual_b = len(keys_by_condition.get("B", []))
        actual_c = len(keys_by_condition.get("C", []))
        actual_d = len(keys_by_condition.get("D", []))

        assert actual_a == expected_a, (
            f"Condition A: expected {expected_a} keys, got {actual_a}"
        )
        assert actual_b == expected_b, (
            f"Condition B: expected {expected_b} keys, got {actual_b}"
        )
        assert actual_c == expected_c, (
            f"Condition C: expected {expected_c} keys, got {actual_c}"
        )
        assert actual_d == expected_d, (
            f"Condition D: expected {expected_d} keys, got {actual_d}"
        )

