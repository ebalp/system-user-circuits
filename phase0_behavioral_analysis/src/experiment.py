"""
Experiment runner for Phase 0 Behavioral Analysis.

Orchestrates experiment execution with resume capability, progress tracking,
and result storage in JSONL format. Includes hash-based experiment tracking
for deduplication and reproducibility.
"""

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


from tqdm import tqdm

from .api_client import HFClient, ChatResponse
from .classifiers import ClassificationResult, get_classifier, compute_label
from .config import ExperimentConfig


logger = logging.getLogger(__name__)


@dataclass
class ExperimentKey:
    """
    Unique identifier for a single experiment.
    
    An experiment is the smallest unit of work that can be run independently.
    Each experiment runs `instances_per_cell` times to gather statistical samples.
    
    This dataclass captures all parameters that define a unique experiment,
    including model, constraint configuration, templates, task, condition,
    direction, and generation parameters. Any change to these fields will
    result in a different experiment hash.
    
    Note on instances_per_cell:
    - With temperature=0 (deterministic): Set to 1, as outputs are identical
    - With temperature>0 (sampling): Set to 10-15 for statistical confidence
    
    Attributes:
        model: HuggingFace model ID
        constraint_type: Name of the constraint type (e.g., "language", "format")
        option_a: Name of the first option in the pair
        option_b: Name of the second option in the pair
        system_template_name: Name of the system template strength level
        system_template_content: Full content of the system template
        user_template_name: Name of the user template style
        user_template_content: Full content of the user template
        task_id: Identifier for the task
        task_prompt: Full text of the task prompt
        condition: Experimental condition (A, B, C, D)
        direction: Counterbalancing direction ('a_to_b', 'b_to_a', or 'none')
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
        instances_per_cell: Number of instances to run
        counterbalancing_enabled: Whether counterbalancing is enabled
        instruction_template: Constraint instruction template with {value} placeholder
        negative_template: Constraint negative instruction text
        option_a_value: Display value for option A
        option_a_expected: Expected classifier value for option A
        option_b_value: Display value for option B
        option_b_expected: Expected classifier value for option B
    
    **Validates: Requirements 8.1, 8.4, 8.5**
    """
    model: str
    constraint_type: str
    option_a: str
    option_b: str
    system_template_name: str | None  # None for B (no system constraint)
    system_template_content: str
    user_template_name: str
    user_template_content: str
    task_id: str
    task_prompt: str
    condition: str
    direction: str  # 'a_to_b', 'b_to_a', or 'none'
    temperature: float
    max_tokens: int
    instances_per_cell: int
    counterbalancing_enabled: bool
    
    # Constraint details for reproducibility
    instruction_template: str
    negative_template: str
    option_a_value: str
    option_a_expected: str
    option_b_value: str
    option_b_expected: str


def compute_experiment_hash(key: ExperimentKey) -> str:
    """
    Compute a deterministic hash for a single experiment.
    
    The hash uniquely identifies this specific experiment configuration.
    If any field changes, the hash changes, allowing re-run. The hash
    is computed from a canonical JSON representation with sorted keys
    to ensure determinism.
    
    Args:
        key: The ExperimentKey containing all experiment parameters.
        
    Returns:
        A 16-character hexadecimal hash string (truncated SHA-256).
        
    Example:
        >>> key = ExperimentKey(model="test-model", ...)
        >>> hash1 = compute_experiment_hash(key)
        >>> hash2 = compute_experiment_hash(key)
        >>> assert hash1 == hash2  # Same key always produces same hash
    
    **Validates: Requirements 8.2, 8.4**
    """
    # Create canonical representation with all fields that define uniqueness
    canonical = {
        'model': key.model,
        'constraint_type': key.constraint_type,
        'option_a': key.option_a,
        'option_b': key.option_b,
        'system_template': {
            'name': key.system_template_name,
            'content': key.system_template_content
        },
        'user_template': {
            'name': key.user_template_name,
            'content': key.user_template_content
        },
        'task': {
            'id': key.task_id,
            'prompt': key.task_prompt
        },
        'condition': key.condition,
        'direction': key.direction,
        'generation': {
            'temperature': key.temperature,
            'max_tokens': key.max_tokens,
            'instances_per_cell': key.instances_per_cell
        },
        'counterbalancing': key.counterbalancing_enabled,
        'constraint': {
            'instruction_template': key.instruction_template,
            'negative_template': key.negative_template,
            'option_a_value': key.option_a_value,
            'option_a_expected': key.option_a_expected,
            'option_b_value': key.option_b_value,
            'option_b_expected': key.option_b_expected
        }
    }
    
    # Create canonical JSON with sorted keys for determinism
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    
    # Compute SHA-256 hash and truncate to 16 characters
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


@dataclass
class ExperimentResult:
    """Result of running a single prompt through a model.
    
    Attributes:
        prompt_id: Unique identifier of the prompt
        model: HuggingFace model ID
        direction: Counterbalancing direction ('a_to_b', 'b_to_a', or 'none')
        response: Raw model response text
        timestamp: ISO timestamp of when the response was received
        classification: Classification result from the appropriate classifier
        label: Compliance label ('followed_system', 'followed_user', etc.)
        confidence: Confidence score for the classification
        error: Error message if the API call failed, None otherwise
    """
    prompt_id: str
    model: str
    direction: str
    response: str
    timestamp: str
    classification: ClassificationResult
    label: str
    confidence: float
    error: str | None


class ExperimentRunner:
    """
    Orchestrates experiment execution with resume capability.
    
    Runs prompts through models, classifies responses, and stores results
    in JSONL format with full metadata.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        client: HFClient,
        output_dir: str = 'data/results',
        per_model_concurrency: int = 3,
    ):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration
            client: HuggingFace API client
            output_dir: Directory for storing results
            per_model_concurrency: Max concurrent API requests per model endpoint
        """
        self.config = config
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_model_concurrency = per_model_concurrency
    
    def _get_results_path(self, model_id: str) -> Path:
        """Get the path to the results file for a model."""
        # Sanitize model ID for filename
        safe_name = model_id.replace('/', '_').replace('\\', '_')
        return self.output_dir / f"{safe_name}_results.jsonl"
    
    def load_results(self, model_id: str) -> list[ExperimentResult]:
        """
        Load all results for a model from the JSONL file.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            List of experiment results
        """
        results_path = self._get_results_path(model_id)
        results = []
        
        if not results_path.exists():
            return results
        
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Reconstruct ClassificationResult
                        classification = ClassificationResult(
                            detected=data['classification']['detected'],
                            confidence=data['classification']['confidence'],
                            details=data['classification'].get('details')
                        )
                        result = ExperimentResult(
                            prompt_id=data['prompt_id'],
                            model=data['model'],
                            direction=data['direction'],
                            response=data['response'],
                            timestamp=data['timestamp'],
                            classification=classification,
                            label=data['label'],
                            confidence=data['confidence'],
                            error=data.get('error')
                        )
                        results.append(result)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed result: {e}")
        
        return results
    
    # =========================================================================
    # Hash-based Experiment Tracking Methods
    # =========================================================================
    
    def _get_constraint_type(self, name: str):
        """Get a constraint type by name from the config.
        
        Args:
            name: The name of the constraint type to retrieve.
            
        Returns:
            The ConstraintType with the given name.
            
        Raises:
            ValueError: If the constraint type is not found.
        """
        for ct in self.config.constraint_types:
            if ct.name == name:
                return ct
        raise ValueError(f"Constraint type '{name}' not found in config")
    
    def _get_strengths_for_condition(self, condition: str) -> list[str]:
        """Get the strength levels to use for a given condition.
        
        Args:
            condition: The experimental condition (A, B, C, D).
            
        Returns:
            List of strength level names to use.
        """
        if condition == 'C':
            return self.config.condition_c_strengths
        if condition == 'B':
            return [None]
        return ['weak']
    
    def _get_directions_for_condition(self, condition: str) -> list[str]:
        """Get the directions to use for a given condition.

        Args:
            condition: The experimental condition (A, B, C, D).

        Returns:
            List of direction strings ('a_to_b', 'b_to_a', 'option_a', or 'option_b').
        """
        if condition in ('C', 'D'):
            if self.config.counterbalancing and self.config.counterbalancing.enabled:
                return ['a_to_b', 'b_to_a']
            return ['a_to_b']
        # Conditions A and B: test both options independently
        return ['option_a', 'option_b']
    
    def generate_experiment_keys(self) -> list[ExperimentKey]:
        """
        Generate all experiment keys from configuration.
        
        Expands the configuration into individual experiments:
        - For each model
        - For each experiment pair
        - For each system template (based on condition)
        - For each user style (with_instruction only for A/B/D, all configured for C)
        - For each task
        - For each condition
        - For each direction (option_a/option_b for A/B, a_to_b/b_to_a for C/D)
        
        Returns:
            List of ExperimentKey objects representing all experiments to run.
            
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        keys = []
        
        # Use new config structure if available, otherwise skip
        if not self.config.constraint_types or not self.config.experiment_pairs:
            logger.warning("No constraint_types or experiment_pairs in config, cannot generate experiment keys")
            return keys
        
        # Get user styles to test (default to ['standard'] if not specified)
        user_styles = self.config.user_styles_to_test or [self.config.default_user_style]
        
        for model in self.config.models:
            for pair in self.config.experiment_pairs:
                ct = self._get_constraint_type(pair.constraint_type)
                opt_a = ct.get_option(pair.option_a)
                opt_b = ct.get_option(pair.option_b)
                
                for task in self.config.tasks:
                    for condition in self.config.conditions:
                        if condition in ('A', 'B', 'D'):
                            styles = [self.config.default_user_style]
                        else:
                            styles = user_styles
                        for user_style in styles:
                            strengths = self._get_strengths_for_condition(condition)
                            directions = self._get_directions_for_condition(condition)
                            
                            for strength in strengths:
                                for direction in directions:
                                    # Get template content
                                    system_template_content = self.config.system_templates.get(
                                        strength, ""
                                    )
                                    user_template_content = self.config.user_templates.get(
                                        user_style, ""
                                    )
                                    
                                    key = ExperimentKey(
                                        model=model,
                                        constraint_type=pair.constraint_type,
                                        option_a=pair.option_a,
                                        option_b=pair.option_b,
                                        system_template_name=strength,
                                        system_template_content=system_template_content,
                                        user_template_name=user_style,
                                        user_template_content=user_template_content,
                                        task_id=task.id,
                                        task_prompt=task.prompt,
                                        condition=condition,
                                        direction=direction,
                                        temperature=self.config.generation.temperature if self.config.generation else 0.0,
                                        max_tokens=self.config.generation.max_tokens if self.config.generation else 512,
                                        instances_per_cell=self.config.generation.instances_per_cell if self.config.generation else 1,
                                        counterbalancing_enabled=self.config.counterbalancing.enabled if self.config.counterbalancing else False,
                                        instruction_template=ct.instruction_template,
                                        negative_template=ct.negative_template,
                                        option_a_value=opt_a.value,
                                        option_a_expected=opt_a.expected_value,
                                        option_b_value=opt_b.value,
                                        option_b_expected=opt_b.expected_value
                                    )
                                    keys.append(key)
        
        return keys
    
    def is_completed(self, key: ExperimentKey) -> bool:
        """
        Check if this specific experiment has already been completed.
        
        Checks the model's JSONL results file for records matching this
        experiment's prompt_id pattern and expected instance count.
        
        Args:
            key: The ExperimentKey to check.
            
        Returns:
            True if the experiment has been completed, False otherwise.
        """
        results_path = self._get_results_path(key.model)
        if not results_path.exists():
            return False
        
        # Build the prompt_id prefix for this experiment
        prefix = self._build_prompt_id_prefix(key)
        
        # Count matching records
        count = 0
        try:
            with open(results_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get('prompt_id', '').startswith(prefix):
                            count += 1
                    except json.JSONDecodeError:
                        continue
        except IOError:
            return False
        
        return count >= key.instances_per_cell
    
    def _build_prompt_id_prefix(self, key: ExperimentKey) -> str:
        """Build the prompt_id prefix that identifies all instances of an experiment."""
        parts = [
            key.condition,
            key.constraint_type,
            key.option_a,
            key.option_b,
        ]
        if key.direction != 'none':
            parts.append(key.direction)
        parts.extend([
            key.system_template_name,
            key.user_template_name,
            key.task_id,
        ])
        return '_'.join(parts)
    
    @staticmethod
    def _interleave_by_model(keys: list[ExperimentKey]) -> list[ExperimentKey]:
        """Round-robin interleave keys across models.

        Without this, all keys for model-0 appear first in the list.
        The thread pool would fill all workers with model-0 tasks, leaving
        12-of-15 blocked on model-0's semaphore with no other model running.

        After interleaving, the first N submitted tasks span all models, so
        each model's semaphore fills independently and all run in parallel.
        """
        from collections import defaultdict
        by_model: dict[str, list[ExperimentKey]] = defaultdict(list)
        for k in keys:
            by_model[k.model].append(k)
        interleaved: list[ExperimentKey] = []
        queues = list(by_model.values())
        while any(queues):
            for q in queues:
                if q:
                    interleaved.append(q.pop(0))
        return interleaved

    def _load_completed_hash_counts(self, model_id: str) -> dict[str, int]:
        """Read the model's JSONL once and return experiment_hash -> record count.

        Used at startup to build an in-memory completion index so that
        is_completed checks are O(1) instead of O(file_size) per key.
        """
        results_path = self._get_results_path(model_id)
        counts: dict[str, int] = {}
        if not results_path.exists():
            return counts
        try:
            with open(results_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get('experiment_hash', '')
                        if h:
                            counts[h] = counts.get(h, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        return counts

    def run_all_with_dedup(self, force: bool = False) -> dict[str, list[ExperimentResult]]:
        """
        Run all experiments, skipping completed ones.

        Uses experiment_hash-based deduplication (O(1) per key) via an
        in-memory index built from a single JSONL scan at startup.

        Requests to the same model are rate-limited to ``per_model_concurrency``
        concurrent calls via a per-model semaphore. Different models run fully
        in parallel since they are independent HF endpoints.

        Args:
            force: If True, re-run all experiments even if completed.

        Returns:
            Dict mapping model_id -> list of ExperimentResult.
        """
        keys = self.generate_experiment_keys()
        results: dict[str, list[ExperimentResult]] = {}

        # Build in-memory completion index: one JSONL scan per model
        all_models = list({k.model for k in keys})
        completed_counts: dict[str, dict[str, int]] = {
            model: self._load_completed_hash_counts(model)
            for model in all_models
        }

        # Filter pending experiments upfront
        pending: list[ExperimentKey] = []
        skipped = 0
        for key in keys:
            h = compute_experiment_hash(key)
            done = completed_counts.get(key.model, {}).get(h, 0)
            if not force and done >= key.instances_per_cell:
                skipped += 1
            else:
                pending.append(key)

        logger.info(f"Pending: {len(pending)}, Already done: {skipped}")
        if not pending:
            logger.info("All experiments already completed.")
            return results

        # Interleave pending keys round-robin by model so that all models
        # get dispatched to workers immediately rather than one model at a time.
        pending = self._interleave_by_model(pending)

        # Per-model semaphores (rate limiting) and locks (safe file writes)
        model_semaphores = {m: threading.Semaphore(self.per_model_concurrency) for m in all_models}
        model_locks = {m: threading.Lock() for m in all_models}

        def run_key(key: ExperimentKey) -> tuple[ExperimentKey, list[ExperimentResult]]:
            with model_semaphores[key.model]:
                exp_results = self._run_experiment_flat(key, model_locks[key.model])
            return key, exp_results

        max_workers = len(all_models) * self.per_model_concurrency
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(run_key, key): key for key in pending}
            for future in tqdm(as_completed(future_to_key), total=len(pending), desc="Running experiments"):
                key = future_to_key[future]
                try:
                    _, exp_results = future.result()
                    results.setdefault(key.model, []).extend(exp_results)
                    completed_count += 1
                except Exception as e:
                    logger.error(
                        f"Experiment {key.condition}/{key.constraint_type} "
                        f"({key.model}) failed: {e}"
                    )

        logger.info(f"Completed: {completed_count}, Skipped: {skipped}")
        return results
    
    def _run_experiment_flat(
        self,
        key: ExperimentKey,
        write_lock: threading.Lock,
    ) -> list[ExperimentResult]:
        """
        Run a single experiment and append results to the model's JSONL file.

        Each result record includes full experiment metadata (condition, strength,
        constraint_type, direction, user_style, etc.) so the JSONL file is
        self-contained and easy to filter.

        Results with errors (after retries) or empty responses are NOT saved,
        so they can be retried on the next run via deduplication.

        Args:
            key: The ExperimentKey defining the experiment to run.
            write_lock: Per-model lock that guards concurrent JSONL appends.

        Returns:
            List of ExperimentResult for all instances.
        """
        results = []
        results_path = self._get_results_path(key.model)

        for i in range(key.instances_per_cell):
            result = self._run_single_instance(key, i)
            results.append(result)

            # Skip saving records with errors or empty responses so they
            # can be retried on the next run
            if result.error:
                logger.warning(f"Skipping save for {result.prompt_id}: error={result.error[:80]}")
                continue
            if not result.response or not result.response.strip():
                logger.warning(f"Skipping save for {result.prompt_id}: empty response")
                continue

            # Build enriched record with experiment metadata
            record = self._build_enriched_record(key, result)

            with write_lock:
                with open(results_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

        return results
    
    def _build_enriched_record(self, key: ExperimentKey, result: ExperimentResult) -> dict:
        """Build a flat JSON record with full experiment metadata.

        This produces self-contained records that can be filtered/analyzed
        without needing to cross-reference a separate config file.
        """
        return {
            'prompt_id': result.prompt_id,
            'experiment_hash': compute_experiment_hash(key),
            'model': result.model,
            'condition': key.condition,
            'constraint_type': key.constraint_type,
            'option_a': key.option_a,
            'option_b': key.option_b,
            'direction': result.direction,
            'strength': key.system_template_name,
            'user_style': key.user_template_name,
            'task_id': key.task_id,
            'system_prompt': self._render_system_message(key),
            'user_prompt': self._render_user_message(key),
            'response': result.response,
            'timestamp': result.timestamp,
            'classification': asdict(result.classification),
            'label': result.label,
            'confidence': result.confidence,
            'error': result.error,
        }
    
    def _run_single_instance(self, key: ExperimentKey, instance_idx: int) -> ExperimentResult:
        """
        Run a single instance of an experiment.
        
        Args:
            key: The ExperimentKey defining the experiment.
            instance_idx: The index of this instance (0-based).
            
        Returns:
            ExperimentResult for this instance.
        """
        # Build prompt ID — includes all dimensions for uniqueness and easy parsing
        direction_part = f"_{key.direction}" if key.direction != 'none' else ""
        strength_part = f"_{key.system_template_name}" if key.system_template_name else ""
        prompt_id = (
            f"{key.condition}_{key.constraint_type}_{key.option_a}_{key.option_b}"
            f"{direction_part}{strength_part}_{key.user_template_name}"
            f"_{key.task_id}_{instance_idx:03d}"
        )
        
        # Render system message
        system_message = self._render_system_message(key)
        
        # Render user message
        user_message = self._render_user_message(key)
        
        # Make API call
        chat_response = self.client.chat_completion(
            model_id=key.model,
            system_message=system_message,
            user_message=user_message,
            temperature=key.temperature,
            max_tokens=key.max_tokens
        )
        
        # Handle API errors
        if chat_response.error:
            return ExperimentResult(
                prompt_id=prompt_id,
                model=key.model,
                direction=key.direction,
                response="",
                timestamp=chat_response.timestamp,
                classification=ClassificationResult(
                    detected="error",
                    confidence=0.0,
                    details={"error": chat_response.error}
                ),
                label="error",
                confidence=0.0,
                error=chat_response.error
            )
        
        # Classify response
        classifier = get_classifier(key.constraint_type)
        classification = classifier.classify(chat_response.content)
        
        # Determine system and user constraints based on condition and direction.
        # NOTE: We use option *names* (e.g., "english", "json") because that's
        # what the classifiers return — NOT the expected_value codes ("en", "es").
        # The LanguageClassifier maps langdetect codes to names via LANG_MAP,
        # and FormatClassifier returns "json"/"plain"/"yaml" directly.
        if key.condition == 'A':
            # System-only: system has constraint, user has task only
            # Select based on direction: option_a or option_b
            system_constraint = key.option_a if key.direction == 'option_a' else key.option_b
            user_constraint = None
        elif key.condition == 'B':
            # User-only: system is generic, user has constraint
            # Select based on direction: option_a or option_b
            system_constraint = None
            user_constraint = key.option_a if key.direction == 'option_a' else key.option_b
        elif key.direction == 'a_to_b':
            system_constraint = key.option_a
            user_constraint = key.option_b
        elif key.direction == 'b_to_a':
            system_constraint = key.option_b
            user_constraint = key.option_a
        else:
            system_constraint = key.option_a
            user_constraint = None
        
        # Compute label
        label, confidence = compute_label(
            classification,
            system_constraint,
            user_constraint
        )
        
        return ExperimentResult(
            prompt_id=prompt_id,
            model=key.model,
            direction=key.direction,
            response=chat_response.content,
            timestamp=chat_response.timestamp,
            classification=classification,
            label=label,
            confidence=confidence,
            error=None
        )
    
    def _render_system_message(self, key: ExperimentKey) -> str:
        """Render the system message for an experiment.

        Condition-aware:
        - A: Bare constraint only ('{instruction}. {negative}.') — no strength template
        - B: No system prompt (empty)
        - C: System has constraint using the configured strength template
        - D: No system prompt (empty)
        """
        # Conditions B and D: no system prompt
        if key.condition in ('B', 'D'):
            return ""

        # Resolve the option value from direction
        if key.direction in ('a_to_b', 'option_a'):
            instruction_value = key.option_a_value
        elif key.direction in ('b_to_a', 'option_b'):
            instruction_value = key.option_b_value
        else:
            instruction_value = key.option_a_value

        instruction = key.instruction_template.format(value=instruction_value)
        negative = key.negative_template

        # Condition A: bare instruction, no strength wrapper
        if key.condition == 'A':
            return f"{instruction}. {negative}."

        # Condition C: use the strength template
        if key.system_template_content:
            return key.system_template_content.format(
                instruction=instruction,
                negative=negative
            )
        return instruction
    
    def _render_user_message(self, key: ExperimentKey) -> str:
        """Render the user message for an experiment.
        
        Condition-aware:
        - A: Task only (no constraint in user message)
        - B: User has constraint + task
        - C: User has opposite constraint + task
        - D: User has first constraint, then second constraint + task (recency)
        """
        # Condition A: user gets task only, no constraint
        if key.condition == 'A':
            return key.task_prompt
        
        # Condition D: recency conflict — two constraints in sequence
        if key.condition == 'D':
            if key.direction == 'a_to_b':
                first_value = key.option_a_value
                second_value = key.option_b_value
            elif key.direction == 'b_to_a':
                first_value = key.option_b_value
                second_value = key.option_a_value
            else:
                first_value = key.option_a_value
                second_value = key.option_b_value
            first_instr = key.instruction_template.format(value=first_value)
            second_instr = key.instruction_template.format(value=second_value)
            negative = key.negative_template
            # Capitalize for user message
            first_instr = first_instr[0].upper() + first_instr[1:] if first_instr else first_instr
            return f"{first_instr}. {negative}. Please {second_instr}. {key.task_prompt}"
        
        # Conditions B and C: user has a constraint + task
        if key.condition == 'B':
            # User-only: user has the constraint, direction-based selection
            if key.direction == 'option_a':
                instruction_value = key.option_a_value
            elif key.direction == 'option_b':
                instruction_value = key.option_b_value
            else:
                # Fallback for unexpected direction values
                instruction_value = key.option_a_value
        elif key.direction == 'a_to_b':
            # C with a_to_b: system=option_a, user=option_b
            instruction_value = key.option_b_value
        elif key.direction == 'b_to_a':
            # C with b_to_a: system=option_b, user=option_a
            instruction_value = key.option_a_value
        else:
            instruction_value = key.option_b_value
        
        instruction = key.instruction_template.format(value=instruction_value)
        
        if key.user_template_content:
            return key.user_template_content.format(
                instruction=instruction,
                task=key.task_prompt
            )
        return f"{instruction}. {key.task_prompt}"
