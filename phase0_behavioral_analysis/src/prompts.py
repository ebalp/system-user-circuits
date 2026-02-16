"""
Prompt generation for Phase 0 Behavioral Analysis.

Generates all prompt combinations from configuration, including counterbalancing
for conflict conditions (C and D) to isolate true hierarchy effects from
model capability biases.
"""

from dataclasses import dataclass, asdict
from typing import Literal
import json
import csv

from .config import (
    ExperimentConfig, ConstraintOption, Task,
    ConstraintType, ExperimentPair
)


@dataclass
class Prompt:
    """A single prompt for the experiment."""
    id: str                           # Unique identifier
    condition: str                    # 'A', 'B', 'C', 'D'
    constraint_type: str              # 'language', 'format'
    system_constraint: str | None     # 'english', 'spanish', etc. (None for B, D)
    user_constraint: str | None       # Constraint in user message (None for A)
    direction: str                    # 'a_to_b', 'b_to_a', or 'none' (for A, B)
    strength: str                     # 'weak', 'medium', 'strong'
    user_style: str                   # 'standard', 'jailbreak', 'polite' - which user template was used
    task_id: str                      # Task identifier
    system_message: str               # Rendered system prompt
    user_message: str                 # Rendered user message
    expected_label: str               # What we expect model to follow


class PromptGenerator:
    """Generates all prompt combinations from configuration.

    Uses constraint_types, experiment_pairs, system_templates, and
    user_templates to generate prompts for configured experiment pairs.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._instance_counters: dict[str, int] = {}
        self._constraint_map: dict[str, ConstraintType] = {
            ct.name: ct for ct in config.constraint_types
        }

    def render_system_prompt(
        self,
        constraint_type: ConstraintType,
        option: ConstraintOption,
        strength: str
    ) -> str:
        """Render system prompt using template system."""
        if strength not in self.config.system_templates:
            available = sorted(self.config.system_templates.keys())
            raise ValueError(
                f"Unknown strength level '{strength}'. "
                f"Available strength levels: {available}"
            )
        template = self.config.system_templates[strength]
        instruction = constraint_type.render_instruction(option)
        negative = constraint_type.negative_template
        return template.format(instruction=instruction, negative=negative)

    def render_user_prompt(
        self,
        constraint_type: ConstraintType,
        option: ConstraintOption,
        task: Task,
        style: str
    ) -> str:
        """Render user prompt using template system."""
        if style not in self.config.user_templates:
            available = sorted(self.config.user_templates.keys())
            raise ValueError(
                f"Unknown user style '{style}'. "
                f"Available user styles: {available}"
            )
        template = self.config.user_templates[style]
        instruction = constraint_type.render_instruction(option)
        if instruction:
            instruction = instruction[0].upper() + instruction[1:]
        return template.format(instruction=instruction, task=task.prompt)

    def get_constraint_type(self, name: str) -> ConstraintType:
        """Get a constraint type by name with O(1) lookup."""
        if name not in self._constraint_map:
            available = sorted(self._constraint_map.keys())
            raise ValueError(
                f"Unknown constraint type '{name}'. "
                f"Available constraint types: {available}"
            )
        return self._constraint_map[name]

    def generate_from_pairs(self) -> list[Prompt]:
        """Generate prompts only for configured experiment pairs.

        For each experiment pair, generates prompts for:
        - All conditions (A, B, C, D)
        - All user styles in user_styles_to_test
        - All strength levels (for condition C)
        - Both directions (if counterbalancing enabled)
        """
        prompts = []
        for pair in self.config.experiment_pairs:
            prompts.extend(self._generate_prompts_for_pair(pair))
        return prompts

    def _generate_prompts_for_pair(self, pair: ExperimentPair) -> list[Prompt]:
        """Generate all prompts for a single experiment pair."""
        prompts = []
        ct = self.get_constraint_type(pair.constraint_type)
        option_a = ct.get_option(pair.option_a)
        option_b = ct.get_option(pair.option_b)
        user_styles = self.config.user_styles_to_test if self.config.user_styles_to_test else ['standard']
        pair_key = f"{pair.option_a[:3]}{pair.option_b[:3]}"

        for condition in self.config.conditions:
            for user_style in user_styles:
                prompts.extend(self._generate_pair_condition(
                    condition=condition,
                    constraint_type=ct,
                    option_a=option_a,
                    option_b=option_b,
                    user_style=user_style,
                    pair_key=pair_key,
                ))
        return prompts

    def _generate_pair_condition(
        self,
        condition: str,
        constraint_type: ConstraintType,
        option_a: ConstraintOption,
        option_b: ConstraintOption,
        user_style: str,
        pair_key: str,
    ) -> list[Prompt]:
        """Generate prompts for a specific condition and user style."""
        if condition == 'A':
            return self._generate_pair_condition_a(constraint_type, option_a, option_b, user_style, pair_key)
        elif condition == 'B':
            return self._generate_pair_condition_b(constraint_type, option_a, option_b, user_style, pair_key)
        elif condition == 'C':
            return self._generate_pair_condition_c(constraint_type, option_a, option_b, user_style, pair_key)
        elif condition == 'D':
            return self._generate_pair_condition_d(constraint_type, option_a, option_b, user_style, pair_key)
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def _generate_pair_condition_a(
        self, constraint_type: ConstraintType,
        option_a: ConstraintOption, option_b: ConstraintOption,
        user_style: str, pair_key: str,
    ) -> list[Prompt]:
        """Generate Condition A prompts (system-only constraint)."""
        prompts = []
        strength = self.config.default_strength
        for option, direction in [(option_a, 'option_a'), (option_b, 'option_b')]:
            for task in self.config.tasks:
                for i in range(self.config.generation.instances_per_cell):
                    prompt_id = self._make_id(
                        'A', constraint_type.name, option.name,
                        None, strength, user_style, task.id, i,
                        pair_key=pair_key,
                    )
                    system_msg = self.render_system_prompt(constraint_type, option, strength)
                    prompts.append(Prompt(
                        id=prompt_id, condition='A',
                        constraint_type=constraint_type.name,
                        system_constraint=option.name, user_constraint=None,
                        direction=direction, strength=strength, user_style=user_style,
                        task_id=task.id, system_message=system_msg,
                        user_message=task.prompt, expected_label='followed_system'
                    ))
        return prompts

    def _generate_pair_condition_b(
        self, constraint_type: ConstraintType,
        option_a: ConstraintOption, option_b: ConstraintOption,
        user_style: str, pair_key: str,
    ) -> list[Prompt]:
        """Generate Condition B prompts (user-only constraint)."""
        prompts = []
        strength = self.config.default_strength
        generic_system = "You are a helpful assistant."
        for option, direction in [(option_a, 'option_a'), (option_b, 'option_b')]:
            for task in self.config.tasks:
                for i in range(self.config.generation.instances_per_cell):
                    prompt_id = self._make_id(
                        'B', constraint_type.name, None,
                        option.name, strength, user_style, task.id, i,
                        pair_key=pair_key,
                    )
                    user_msg = self.render_user_prompt(constraint_type, option, task, user_style)
                    prompts.append(Prompt(
                        id=prompt_id, condition='B',
                        constraint_type=constraint_type.name,
                        system_constraint=None, user_constraint=option.name,
                        direction=direction, strength=strength, user_style=user_style,
                        task_id=task.id, system_message=generic_system,
                        user_message=user_msg, expected_label='followed_user'
                    ))
        return prompts

    def _generate_pair_condition_c(
        self, constraint_type: ConstraintType,
        option_a: ConstraintOption, option_b: ConstraintOption,
        user_style: str, pair_key: str,
    ) -> list[Prompt]:
        """Generate Condition C prompts (hierarchy conflict)."""
        prompts = []
        for strength in self.config.condition_c_strengths:
            for task in self.config.tasks:
                prompts.extend(self._generate_pair_conflict(
                    condition='C', constraint_type=constraint_type,
                    system_option=option_a, user_option=option_b,
                    direction='a_to_b', strength=strength,
                    user_style=user_style, task=task, pair_key=pair_key,
                ))
                if self.config.counterbalancing and self.config.counterbalancing.enabled:
                    prompts.extend(self._generate_pair_conflict(
                        condition='C', constraint_type=constraint_type,
                        system_option=option_b, user_option=option_a,
                        direction='b_to_a', strength=strength,
                        user_style=user_style, task=task, pair_key=pair_key,
                    ))
        return prompts

    def _generate_pair_condition_d(
        self, constraint_type: ConstraintType,
        option_a: ConstraintOption, option_b: ConstraintOption,
        user_style: str, pair_key: str,
    ) -> list[Prompt]:
        """Generate Condition D prompts (user-user conflict / recency)."""
        prompts = []
        strength = self.config.default_strength
        generic_system = "You are a helpful assistant."
        for task in self.config.tasks:
            prompts.extend(self._generate_pair_recency(
                constraint_type=constraint_type,
                first_option=option_a, second_option=option_b,
                direction='a_to_b', strength=strength,
                user_style=user_style, task=task,
                generic_system=generic_system, pair_key=pair_key,
            ))
            if self.config.counterbalancing and self.config.counterbalancing.enabled:
                prompts.extend(self._generate_pair_recency(
                    constraint_type=constraint_type,
                    first_option=option_b, second_option=option_a,
                    direction='b_to_a', strength=strength,
                    user_style=user_style, task=task,
                    generic_system=generic_system, pair_key=pair_key,
                ))
        return prompts

    def _generate_pair_conflict(
        self, condition: str, constraint_type: ConstraintType,
        system_option: ConstraintOption, user_option: ConstraintOption,
        direction: str, strength: str, user_style: str, task: Task,
        pair_key: str,
    ) -> list[Prompt]:
        """Generate prompts for a single conflict direction."""
        prompts = []
        for i in range(self.config.generation.instances_per_cell):
            prompt_id = self._make_id(
                condition, constraint_type.name, system_option.name,
                user_option.name, strength, user_style, task.id, i, direction,
                pair_key=pair_key,
            )
            system_msg = self.render_system_prompt(constraint_type, system_option, strength)
            user_msg = self.render_user_prompt(constraint_type, user_option, task, user_style)
            prompts.append(Prompt(
                id=prompt_id, condition=condition,
                constraint_type=constraint_type.name,
                system_constraint=system_option.name,
                user_constraint=user_option.name,
                direction=direction, strength=strength, user_style=user_style,
                task_id=task.id, system_message=system_msg,
                user_message=user_msg, expected_label='followed_system'
            ))
        return prompts

    def _generate_pair_recency(
        self, constraint_type: ConstraintType,
        first_option: ConstraintOption, second_option: ConstraintOption,
        direction: str, strength: str, user_style: str,
        task: Task, generic_system: str, pair_key: str,
    ) -> list[Prompt]:
        """Generate prompts for a single recency direction."""
        prompts = []
        for i in range(self.config.generation.instances_per_cell):
            prompt_id = self._make_id(
                'D', constraint_type.name, first_option.name,
                second_option.name, strength, user_style, task.id, i, direction,
                pair_key=pair_key,
            )
            first_instr = constraint_type.render_instruction(first_option)
            second_instr = constraint_type.render_instruction(second_option)
            if first_instr:
                first_instr = first_instr[0].upper() + first_instr[1:]
            if second_instr:
                second_instr = second_instr[0].upper() + second_instr[1:]
            user_msg = f"{first_instr}. Actually, {second_instr}. {task.prompt}"
            prompts.append(Prompt(
                id=prompt_id, condition='D',
                constraint_type=constraint_type.name,
                system_constraint=None,
                user_constraint=second_option.name,
                direction=direction, strength=strength, user_style=user_style,
                task_id=task.id, system_message=generic_system,
                user_message=user_msg, expected_label='followed_user'
            ))
        return prompts

    def _make_id(
        self, condition: str, constraint_type: str,
        system_opt: str | None, user_opt: str | None,
        strength: str, user_style: str, task_id: str,
        instance: int, direction: str | None = None,
        pair_key: str | None = None
    ) -> str:
        """Generate a unique prompt ID including user style and pair context."""
        parts = [condition, constraint_type]
        if pair_key:
            parts.append(pair_key)
        if system_opt:
            parts.append(system_opt[:3])
        if user_opt:
            parts.append(user_opt[:3])
        if direction and direction != 'none':
            parts.append(direction)
        parts.extend([strength, user_style[:3], task_id, f"{instance:03d}"])
        return '_'.join(parts)

    def export_to_json(self, prompts: list[Prompt], path: str) -> None:
        """Export prompts to JSON for review."""
        data = [asdict(p) for p in prompts]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_csv(self, prompts: list[Prompt], path: str) -> None:
        """Export prompts to CSV for review."""
        if not prompts:
            return
        fieldnames = list(asdict(prompts[0]).keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for prompt in prompts:
                writer.writerow(asdict(prompt))
