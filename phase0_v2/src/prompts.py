"""Prompt generation for Phase 0 v2 experiments.

Generates all prompt combinations using Conflict instances and experiment config
templates. Produces prompts for all 4 conditions (A/B/C/D) with appropriate
counterbalancing, system styles, and user styles.
"""

import hashlib
import random
from dataclasses import dataclass, asdict
from typing import Union

from ..conflicts.conflict_base import Conflict
from .config import ExperimentConfig, Task


def _deterministic_seed(global_seed: int, *keys: str) -> int:
    """Derive a deterministic seed from global seed + identifying keys."""
    raw = f"{global_seed}:{'|'.join(keys)}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


@dataclass
class WildChatTaskProtocol:
    """Minimal interface for WildChat tasks used in prompt generation."""
    wildchat_id: str
    prompt: str
    category: str
    task_id: str  # computed property in real WildChatTask


@dataclass
class Prompt:
    """A single prompt for the experiment."""
    id: str                        # unique prompt identifier
    condition: str                 # "A", "B", "C", "D"
    constraint_type: str           # = conflict_id (for Phase 1 compatibility)
    conflict_id: str               # descriptive ID, e.g. "forbidden_words"
    conflict_class: str            # class name, e.g. "ForbiddenWordsConflict"
    instruction_args: dict         # sampled args (empty dict if no args)
    direction: str                 # "a_to_b" or "b_to_a" (all conditions); "none" for single-direction
    system_style: str | None        # None for condition B; "bare"/"compliance"/"authority"/"persona"/"safety"
    user_style: str                # template name: "with_instruction", "task_only", "recency", etc.
    task_id: str                   # e.g. "photosynthesis" or "wc_a3f7b2c1d4e5"
    task_source: str               # "synthetic" or "wildchat"
    wildchat_id: str | None        # WildChat ID if applicable
    system_message: str            # fully rendered system prompt
    user_message: str              # fully rendered user prompt
    expected_label: str            # "followed_system" or "followed_user"


class PromptGenerator:
    """Generates all prompt combinations from configuration using Conflict instances."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def generate_for_conflict(
        self, conflict: Conflict, tasks: list
    ) -> list[Prompt]:
        """Generate prompts for all conditions x tasks for one conflict."""
        prompts = []
        for task in tasks:
            task_id = self._extract_task_info(task)[0]
            random.seed(_deterministic_seed(self.config.seed, conflict.conflict_id, task_id))
            args = conflict.sample_args()
            prompts.extend(self._gen_condition_a(conflict, task, args))
            prompts.extend(self._gen_condition_b(conflict, task, args))
            prompts.extend(self._gen_condition_c(conflict, task, args))
            prompts.extend(self._gen_condition_d(conflict, task, args))
        return prompts

    # ------------------------------------------------------------------
    # Condition A: System-only baseline
    # ------------------------------------------------------------------

    def _gen_condition_a(
        self, conflict: Conflict, task, args: dict
    ) -> list[Prompt]:
        """Condition A: system instruction only, no user constraint.

        Generates one prompt per side of the conflict:
        - Side "a": system prompt uses the primary constraint
        - Side "b": system prompt uses the inverse constraint (invertible only)

        This provides per-side baselines needed to interpret Condition C asymmetry.
        """
        prompts = []
        bare_tmpl = self.config.system_templates["bare"].template
        task_id, task_prompt, task_source, wildchat_id = self._extract_task_info(task)

        sides = ["a"]
        if conflict.supports_counterbalancing():
            sides.append("b")

        for side in sides:
            system_instruction = conflict.build_system_prompt(direction=side, **args)
            system_message = bare_tmpl.format(system_instruction=system_instruction)

            direction_label = "a_to_b" if side == "a" else "b_to_a"
            prompt_id = self._make_id(
                conflict.conflict_id, direction_label, task_id, "bare", "task_only", 0
            )

            prompts.append(Prompt(
                id=prompt_id,
                condition="A",
                constraint_type=conflict.conflict_id,
                conflict_id=conflict.conflict_id,
                conflict_class=conflict.__class__.__name__,
                instruction_args=dict(args),
                direction=direction_label,
                system_style="bare",
                user_style="task_only",
                task_id=task_id,
                task_source=task_source,
                wildchat_id=wildchat_id,
                system_message=system_message,
                user_message=task_prompt,
                expected_label="followed_system",
            ))

        return prompts

    # ------------------------------------------------------------------
    # Condition B: User-only baseline
    # ------------------------------------------------------------------

    def _gen_condition_b(
        self, conflict: Conflict, task, args: dict
    ) -> list[Prompt]:
        """Condition B: user instruction only, no system constraint.

        Generates one prompt per side of the conflict using default user style:
        - Side "a": user asks for the primary constraint
        - Side "b": user asks for the inverse constraint (invertible only)

        This provides per-side baselines needed to interpret Condition C asymmetry.
        """
        prompts = []
        task_id, task_prompt, task_source, wildchat_id = self._extract_task_info(task)

        style_name = self.config.default_user_style
        user_tmpl = self.config.user_templates[style_name].template

        sides = ["a"]
        if conflict.supports_counterbalancing():
            sides.append("b")

        for side in sides:
            user_instruction = conflict.build_user_conflict_prompt(direction=side, **args)
            user_message = user_tmpl.format(
                user_instruction=user_instruction, task=task_prompt
            )

            direction_label = "a_to_b" if side == "a" else "b_to_a"
            prompt_id = self._make_id(
                conflict.conflict_id, direction_label, task_id, "none", style_name, 0
            )

            prompts.append(Prompt(
                id=prompt_id,
                condition="B",
                constraint_type=conflict.conflict_id,
                conflict_id=conflict.conflict_id,
                conflict_class=conflict.__class__.__name__,
                instruction_args=dict(args),
                direction=direction_label,
                system_style=None,
                user_style=style_name,
                task_id=task_id,
                task_source=task_source,
                wildchat_id=wildchat_id,
                system_message="",
                user_message=user_message,
                expected_label="followed_user",
            ))

        return prompts

    # ------------------------------------------------------------------
    # Condition C: Hierarchy conflict (main test)
    # ------------------------------------------------------------------

    def _gen_condition_c(
        self, conflict: Conflict, task, args: dict
    ) -> list[Prompt]:
        """Condition C: system vs user conflict.

        Generates prompts for each combination of direction x system_style x user_style.
        """
        prompts = []
        task_id, task_prompt, task_source, wildchat_id = self._extract_task_info(task)

        directions = ["a_to_b"]
        if conflict.supports_counterbalancing():
            directions.append("b_to_a")

        for direction in directions:
            dir_code = "a" if direction == "a_to_b" else "b"
            system_instruction = conflict.build_system_prompt(direction=dir_code, **args)
            user_instruction = conflict.build_user_conflict_prompt(direction=dir_code, **args)

            for system_style in self.config.condition_c_system_styles:
                style_tmpl = self.config.system_templates[system_style].template
                system_message = style_tmpl.format(
                    system_instruction=system_instruction
                )

                for style_name in self.config.user_styles_to_test:
                    user_tmpl = self.config.user_templates[style_name].template
                    user_message = user_tmpl.format(
                        user_instruction=user_instruction, task=task_prompt
                    )

                    prompt_id = self._make_id(
                        conflict.conflict_id, direction, task_id,
                        system_style, style_name, 0
                    )

                    prompts.append(Prompt(
                        id=prompt_id,
                        condition="C",
                        constraint_type=conflict.conflict_id,
                        conflict_id=conflict.conflict_id,
                        conflict_class=conflict.__class__.__name__,
                        instruction_args=dict(args),
                        direction=direction,
                        system_style=system_style,
                        user_style=style_name,
                        task_id=task_id,
                        task_source=task_source,
                        wildchat_id=wildchat_id,
                        system_message=system_message,
                        user_message=user_message,
                        expected_label="followed_system",
                    ))

        return prompts

    # ------------------------------------------------------------------
    # Condition D: Recency control
    # ------------------------------------------------------------------

    def _gen_condition_d(
        self, conflict: Conflict, task, args: dict
    ) -> list[Prompt]:
        """Condition D: recency control (both constraints in user message).

        Only generated for conflicts that support counterbalancing.
        """
        if not conflict.supports_counterbalancing():
            return []

        prompts = []
        task_id, task_prompt, task_source, wildchat_id = self._extract_task_info(task)

        for direction in ["a_to_b", "b_to_a"]:
            if direction == "a_to_b":
                # first_instruction = system-side constraint rephrased as user request
                # second_instruction = user-side constraint
                # Use inverse_user_template for the system-side constraint as user request
                # (inverse_user is what asks for the system's constraint from user perspective)
                first_instruction = conflict.build_user_conflict_prompt(direction="b", **args)
                second_instruction = conflict.build_user_conflict_prompt(direction="a", **args)
            else:
                # b_to_a: instructions swap
                first_instruction = conflict.build_user_conflict_prompt(direction="a", **args)
                second_instruction = conflict.build_user_conflict_prompt(direction="b", **args)

            user_message = f"{first_instruction} {second_instruction}\n\n{task_prompt}"

            prompt_id = self._make_id(
                conflict.conflict_id, direction, task_id,
                "bare", "recency", 0
            )

            prompts.append(Prompt(
                id=prompt_id,
                condition="D",
                constraint_type=conflict.conflict_id,
                conflict_id=conflict.conflict_id,
                conflict_class=conflict.__class__.__name__,
                instruction_args=dict(args),
                direction=direction,
                system_style="bare",
                user_style="recency",
                task_id=task_id,
                task_source=task_source,
                wildchat_id=wildchat_id,
                system_message="",
                user_message=user_message,
                expected_label="followed_user",
            ))

        return prompts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task_info(task) -> tuple[str, str, str, str | None]:
        """Extract (task_id, prompt, task_source, wildchat_id) from a task object.

        Supports both Task (synthetic) and WildChatTask objects.
        """
        if hasattr(task, "wildchat_id"):
            # WildChatTask
            return (
                task.task_id if hasattr(task, "task_id") else f"wc_{task.wildchat_id[:12]}",
                task.prompt,
                "wildchat",
                task.wildchat_id,
            )
        else:
            # Synthetic Task
            return task.id, task.prompt, "synthetic", None

    @staticmethod
    def _make_id(
        conflict_id: str,
        direction: str,
        task_id: str,
        system_style: str,
        user_style: str,
        instance: int,
    ) -> str:
        """Generate a unique prompt ID.

        Format: {conflict_id}_{direction}_{task_id}_{system_style}_{user_style:3}_{instance:03d}
        """
        style_short = user_style[:3]
        parts = [
            conflict_id,
            direction,
            task_id,
            system_style,
            style_short,
            f"{instance:03d}",
        ]
        return "_".join(parts)
