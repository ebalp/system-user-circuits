"""Experiment runner for Phase 0 v2.

Orchestrates experiment execution with resume capability, progress tracking,
and result storage in JSONL format. Includes hash-based experiment tracking
for deduplication and reproducibility.
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .prompts import Prompt
from .classifiers import classify_response
from ..conflicts.conflict_base import Conflict


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentKey:
    """Unique identifier for a single experiment cell.

    All fields that define a unique experiment are included so that
    any change produces a different hash.
    """
    model: str
    conflict_id: str
    instruction_args_json: str   # canonical JSON of instruction_args (sorted keys)
    task_id: str
    task_source: str
    condition: str
    direction: str
    system_style: str | None
    user_style: str
    temperature: float
    max_tokens: int
    system_prompt: str
    user_prompt: str


def compute_experiment_hash(key: ExperimentKey) -> str:
    """Compute a deterministic 16-char hex hash for an experiment key.

    Uses SHA-256 of canonical JSON (sorted keys) of all ExperimentKey fields.
    """
    canonical = {
        "model": key.model,
        "conflict_id": key.conflict_id,
        "instruction_args_json": key.instruction_args_json,
        "task_id": key.task_id,
        "task_source": key.task_source,
        "condition": key.condition,
        "direction": key.direction,
        "system_style": key.system_style,
        "user_style": key.user_style,
        "temperature": key.temperature,
        "max_tokens": key.max_tokens,
        "system_prompt": key.system_prompt,
        "user_prompt": key.user_prompt,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


def recompute_hash_from_record(rec: dict) -> str:
    """Recompute experiment hash from a JSONL record's stored fields.

    Builds an ExperimentKey from the record and returns the hash using
    the current formula. Useful for migrating old records after the hash
    formula changes (e.g. adding prompt text).
    """
    key = ExperimentKey(
        model=rec["model"],
        conflict_id=rec["conflict_id"],
        instruction_args_json=json.dumps(
            rec.get("instruction_args", {}), sort_keys=True, separators=(",", ":")
        ),
        task_id=rec["task_id"],
        task_source=rec.get("task_source", "synthetic"),
        condition=rec["condition"],
        direction=rec["direction"],
        system_style=rec.get("system_style"),
        user_style=rec.get("user_style", "with_instruction"),
        temperature=rec.get("temperature", 0.0),
        max_tokens=rec.get("max_tokens", 512),
        system_prompt=rec.get("system_prompt", ""),
        user_prompt=rec.get("user_prompt", ""),
    )
    return compute_experiment_hash(key)


def _make_experiment_key(
    prompt: Prompt, model: str, config: ExperimentConfig
) -> ExperimentKey:
    """Build an ExperimentKey from a Prompt and run configuration."""
    return ExperimentKey(
        model=model,
        conflict_id=prompt.conflict_id,
        instruction_args_json=json.dumps(
            prompt.instruction_args, sort_keys=True, separators=(",", ":")
        ),
        task_id=prompt.task_id,
        task_source=prompt.task_source,
        condition=prompt.condition,
        direction=prompt.direction,
        system_style=prompt.system_style,
        user_style=prompt.user_style,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
        system_prompt=prompt.system_message,
        user_prompt=prompt.user_message,
    )


def _direction_to_verify_code(direction: str) -> str:
    """Map prompt direction string to the verify function direction code.

    'a_to_b' -> 'a'
    'b_to_a' -> 'b'
    'none' -> 'a'
    """
    return {
        "a_to_b": "a",
        "b_to_a": "b",
        "none": "a",
    }.get(direction, "a")


def build_record(
    prompt: Prompt,
    response: str,
    conflict: Conflict,
    model: str,
    config: ExperimentConfig,
    error: str | None = None,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the full JSONL record for one experiment result.

    Args:
        prompt: The Prompt that was sent to the model.
        response: The raw model response text.
        conflict: The Conflict instance used for verification.
        model: The model identifier.
        config: Experiment configuration (for temperature/max_tokens).
        error: Error message if the API call failed.
        threshold_overrides: Optional per-conflict threshold overrides from
            model config. Maps conflict_id to threshold float.

    Returns:
        A flat dictionary matching the output JSONL schema.
    """
    key = _make_experiment_key(prompt, model, config)
    experiment_hash = compute_experiment_hash(key)

    if error:
        return {
            "prompt_id": prompt.id,
            "experiment_hash": experiment_hash,
            "model": model,
            "condition": prompt.condition,
            "constraint_type": prompt.conflict_id,
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
            "response": "",
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expected_label": prompt.expected_label,
            "verify_system_result": False,
            "verify_user_result": False,
            "label": "error",
            "confidence": 0.0,
            "error": error,
        }

    dir_code = _direction_to_verify_code(prompt.direction)

    # Store args so verify functions can access them
    conflict.build_system_prompt(direction=dir_code, **prompt.instruction_args)

    # Look up per-conflict threshold override if provided
    threshold = None
    if threshold_overrides:
        threshold = threshold_overrides.get(conflict.conflict_id)

    sys_ok = conflict.verify_followed_system(response, direction=dir_code, threshold=threshold)
    usr_ok = conflict.verify_followed_user(response, direction=dir_code, threshold=threshold)
    sys_score = conflict.score_system(response, direction=dir_code)
    usr_score = conflict.score_user(response, direction=dir_code)

    if sys_ok and not usr_ok:
        label, confidence = "followed_system", 1.0
    elif usr_ok and not sys_ok:
        label, confidence = "followed_user", 1.0
    elif not sys_ok and not usr_ok:
        label, confidence = "followed_neither", 1.0
    else:
        label, confidence = "followed_both", 0.5

    return {
        "prompt_id": prompt.id,
        "experiment_hash": experiment_hash,
        "model": model,
        "condition": prompt.condition,
        "constraint_type": prompt.conflict_id,
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
        "response": response,
        "temperature": config.generation.temperature,
        "max_tokens": config.generation.max_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "expected_label": prompt.expected_label,
        "verify_system_result": sys_ok,
        "verify_user_result": usr_ok,
        "verify_system_score": sys_score,
        "verify_user_score": usr_score,
        "label": label,
        "confidence": confidence,
        "error": None,
    }


class ExperimentRunner:
    """Orchestrates experiment execution with resume capability.

    Runs prompts through models, classifies responses using conflict verify
    functions, and stores results in JSONL format with full metadata.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client: Any = None,
        output_dir: str = "data/results",
        per_model_concurrency: int = 3,
    ):
        self.config = config
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_model_concurrency = per_model_concurrency

    def _get_results_path(self, model_id: str) -> Path:
        """Get the path to the results file for a model."""
        safe_name = model_id.replace("/", "_").replace("\\", "_")
        return self.output_dir / f"{safe_name}_results.jsonl"

    def load_completed_hash_counts(self, model_id: str) -> dict[str, int]:
        """Read the model's JSONL once and return {experiment_hash: count}.

        Used at startup to build an in-memory completion index so that
        dedup checks are O(1) per experiment key.
        """
        results_path = self._get_results_path(model_id)
        counts: dict[str, int] = {}
        if not results_path.exists():
            return counts
        try:
            with open(results_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get("experiment_hash", "")
                        if h:
                            counts[h] = counts.get(h, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        return counts

    def run_single(
        self,
        prompt: Prompt,
        conflict: Conflict,
        model: str,
        threshold_overrides: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run a single prompt through the model and return the result record.

        Builds chat messages from the Prompt, calls the API client, classifies
        the response using the conflict's verify functions, and returns the
        enriched record dict.

        Args:
            threshold_overrides: Optional per-conflict threshold overrides from
                model config. Maps conflict_id to threshold float.
        """
        if self.client is None:
            raise RuntimeError("No API client configured")

        # Build chat messages
        messages = []
        if prompt.system_message:
            messages.append({"role": "system", "content": prompt.system_message})
        messages.append({"role": "user", "content": prompt.user_message})

        # Call API
        try:
            response = self.client.chat_completion(
                model_id=model,
                messages=messages,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_tokens,
            )
            response_text = response.content if hasattr(response, "content") else str(response)
            error = None
        except Exception as e:
            response_text = ""
            error = str(e)

        return build_record(
            prompt=prompt,
            response=response_text,
            conflict=conflict,
            model=model,
            config=self.config,
            error=error,
            threshold_overrides=threshold_overrides,
        )

    def append_record(self, model_id: str, record: dict[str, Any]) -> None:
        """Append a single record to the model's JSONL file (thread-safe)."""
        results_path = self._get_results_path(model_id)
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
