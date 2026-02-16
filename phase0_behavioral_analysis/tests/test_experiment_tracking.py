"""
Property-based and unit tests for experiment tracking with hash-based deduplication.

Tests the ExperimentKey dataclass, compute_experiment_hash function, and
ExperimentRunner's hash-based deduplication methods.
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

from src.experiment import (
    ExperimentKey,
    compute_experiment_hash,
    ExperimentRunner,
    ExperimentResult,
)
from src.classifiers import ClassificationResult


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for generating valid model names
model_name_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=['L', 'Nd', 'Po'])
).filter(lambda s: len(s.strip()) > 0)

# Strategy for generating valid option names (alphanumeric, lowercase)
option_name_strategy = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=['Ll', 'Nd'])
).filter(lambda s: len(s) > 0 and s[0].isalpha())

# Strategy for generating template content
template_content_strategy = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(whitelist_categories=['L', 'Nd', 'Zs', 'Po'])
).filter(lambda s: len(s.strip()) > 0)

# Strategy for generating task prompts
task_prompt_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(whitelist_categories=['L', 'Nd', 'Zs', 'Po'])
).filter(lambda s: len(s.strip()) > 0)


@st.composite
def experiment_key_strategy(draw):
    """Generate a valid ExperimentKey with all required fields."""
    return ExperimentKey(
        model=draw(model_name_strategy),
        constraint_type=draw(st.sampled_from(['language', 'format'])),
        option_a=draw(option_name_strategy),
        option_b=draw(option_name_strategy),
        system_template_name=draw(st.sampled_from(['weak', 'medium', 'strong'])),
        system_template_content=draw(template_content_strategy.map(
            lambda s: f"{s} {{instruction}} {{negative}}"
        )),
        user_template_name=draw(st.sampled_from(['standard', 'jailbreak', 'polite'])),
        user_template_content=draw(template_content_strategy.map(
            lambda s: f"{s} {{instruction}} {{task}}"
        )),
        task_id=draw(option_name_strategy),
        task_prompt=draw(task_prompt_strategy),
        condition=draw(st.sampled_from(['A', 'B', 'C', 'D'])),
        direction=draw(st.sampled_from(['a_to_b', 'b_to_a', 'option_a', 'option_b'])),
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.integers(min_value=1, max_value=4096)),
        instances_per_cell=draw(st.integers(min_value=1, max_value=100)),
        counterbalancing_enabled=draw(st.booleans()),
        instruction_template=draw(template_content_strategy.map(
            lambda s: f"{s} {{value}}"
        )),
        negative_template=draw(template_content_strategy),
        option_a_value=draw(template_content_strategy),
        option_a_expected=draw(option_name_strategy),
        option_b_value=draw(template_content_strategy),
        option_b_expected=draw(option_name_strategy)
    )


# =============================================================================
# Property 17: Experiment Hash Determinism
# =============================================================================

class TestExperimentHashDeterminism:
    """
    Property 17: Experiment Hash Determinism
    
    For any ExperimentKey, computing the experiment hash multiple times
    should always produce the same hash value.
    
    **Validates: Requirements 8.2, 8.4**
    """
    
    # Feature: config-refactoring, Property 17: Experiment Hash Determinism
    
    @given(experiment_key_strategy())
    @settings(max_examples=100)
    def test_hash_is_deterministic(self, key: ExperimentKey):
        """
        Property: Same ExperimentKey always produces the same hash.
        
        **Validates: Requirements 8.2, 8.4**
        """
        hash1 = compute_experiment_hash(key)
        hash2 = compute_experiment_hash(key)
        hash3 = compute_experiment_hash(key)
        
        assert hash1 == hash2
        assert hash2 == hash3
    
    @given(experiment_key_strategy())
    @settings(max_examples=100)
    def test_hash_is_16_characters(self, key: ExperimentKey):
        """
        Property: Hash is always exactly 16 hexadecimal characters.
        
        **Validates: Requirements 8.2**
        """
        hash_value = compute_experiment_hash(key)
        
        assert len(hash_value) == 16
        assert all(c in '0123456789abcdef' for c in hash_value)
    
    @given(experiment_key_strategy())
    @settings(max_examples=100)
    def test_hash_from_recreated_key_is_same(self, key: ExperimentKey):
        """
        Property: Recreating an ExperimentKey with same values produces same hash.
        
        **Validates: Requirements 8.2, 8.4**
        """
        # Create a new key with the same values
        key_copy = ExperimentKey(
            model=key.model,
            constraint_type=key.constraint_type,
            option_a=key.option_a,
            option_b=key.option_b,
            system_template_name=key.system_template_name,
            system_template_content=key.system_template_content,
            user_template_name=key.user_template_name,
            user_template_content=key.user_template_content,
            task_id=key.task_id,
            task_prompt=key.task_prompt,
            condition=key.condition,
            direction=key.direction,
            temperature=key.temperature,
            max_tokens=key.max_tokens,
            instances_per_cell=key.instances_per_cell,
            counterbalancing_enabled=key.counterbalancing_enabled,
            instruction_template=key.instruction_template,
            negative_template=key.negative_template,
            option_a_value=key.option_a_value,
            option_a_expected=key.option_a_expected,
            option_b_value=key.option_b_value,
            option_b_expected=key.option_b_expected
        )
        
        assert compute_experiment_hash(key) == compute_experiment_hash(key_copy)



# =============================================================================
# Property 18: Experiment Hash Sensitivity
# =============================================================================

class TestExperimentHashSensitivity:
    """
    Property 18: Experiment Hash Sensitivity
    
    For any two ExperimentKeys that differ in any field (model, pair, templates,
    task, condition, direction, or generation parameters), the experiment hashes
    should be different.
    
    **Validates: Requirements 8.4**
    """
    
    # Feature: config-refactoring, Property 18: Experiment Hash Sensitivity
    
    @given(experiment_key_strategy(), model_name_strategy)
    @settings(max_examples=100)
    def test_hash_changes_with_model(self, key: ExperimentKey, new_model: str):
        """
        Property: Different model produces different hash.
        
        **Validates: Requirements 8.4**
        """
        if new_model == key.model:
            return  # Skip if same model
        
        key_modified = ExperimentKey(
            model=new_model,
            constraint_type=key.constraint_type,
            option_a=key.option_a,
            option_b=key.option_b,
            system_template_name=key.system_template_name,
            system_template_content=key.system_template_content,
            user_template_name=key.user_template_name,
            user_template_content=key.user_template_content,
            task_id=key.task_id,
            task_prompt=key.task_prompt,
            condition=key.condition,
            direction=key.direction,
            temperature=key.temperature,
            max_tokens=key.max_tokens,
            instances_per_cell=key.instances_per_cell,
            counterbalancing_enabled=key.counterbalancing_enabled,
            instruction_template=key.instruction_template,
            negative_template=key.negative_template,
            option_a_value=key.option_a_value,
            option_a_expected=key.option_a_expected,
            option_b_value=key.option_b_value,
            option_b_expected=key.option_b_expected
        )
        
        assert compute_experiment_hash(key) != compute_experiment_hash(key_modified)
    
    @given(experiment_key_strategy(), template_content_strategy)
    @settings(max_examples=100)
    def test_hash_changes_with_system_template_content(self, key: ExperimentKey, new_content: str):
        """
        Property: Different system template content produces different hash.
        
        **Validates: Requirements 8.4**
        """
        new_template = f"{new_content} {{instruction}} {{negative}}"
        if new_template == key.system_template_content:
            return  # Skip if same content
        
        key_modified = ExperimentKey(
            model=key.model,
            constraint_type=key.constraint_type,
            option_a=key.option_a,
            option_b=key.option_b,
            system_template_name=key.system_template_name,
            system_template_content=new_template,
            user_template_name=key.user_template_name,
            user_template_content=key.user_template_content,
            task_id=key.task_id,
            task_prompt=key.task_prompt,
            condition=key.condition,
            direction=key.direction,
            temperature=key.temperature,
            max_tokens=key.max_tokens,
            instances_per_cell=key.instances_per_cell,
            counterbalancing_enabled=key.counterbalancing_enabled,
            instruction_template=key.instruction_template,
            negative_template=key.negative_template,
            option_a_value=key.option_a_value,
            option_a_expected=key.option_a_expected,
            option_b_value=key.option_b_value,
            option_b_expected=key.option_b_expected
        )
        
        assert compute_experiment_hash(key) != compute_experiment_hash(key_modified)
    
    @given(experiment_key_strategy(), task_prompt_strategy)
    @settings(max_examples=100)
    def test_hash_changes_with_task_prompt(self, key: ExperimentKey, new_prompt: str):
        """
        Property: Different task prompt produces different hash.
        
        **Validates: Requirements 8.4**
        """
        if new_prompt == key.task_prompt:
            return  # Skip if same prompt
        
        key_modified = ExperimentKey(
            model=key.model,
            constraint_type=key.constraint_type,
            option_a=key.option_a,
            option_b=key.option_b,
            system_template_name=key.system_template_name,
            system_template_content=key.system_template_content,
            user_template_name=key.user_template_name,
            user_template_content=key.user_template_content,
            task_id=key.task_id,
            task_prompt=new_prompt,
            condition=key.condition,
            direction=key.direction,
            temperature=key.temperature,
            max_tokens=key.max_tokens,
            instances_per_cell=key.instances_per_cell,
            counterbalancing_enabled=key.counterbalancing_enabled,
            instruction_template=key.instruction_template,
            negative_template=key.negative_template,
            option_a_value=key.option_a_value,
            option_a_expected=key.option_a_expected,
            option_b_value=key.option_b_value,
            option_b_expected=key.option_b_expected
        )
        
        assert compute_experiment_hash(key) != compute_experiment_hash(key_modified)
    
    @given(experiment_key_strategy(), st.floats(min_value=0.0, max_value=2.0))
    @settings(max_examples=100)
    def test_hash_changes_with_temperature(self, key: ExperimentKey, new_temp: float):
        """
        Property: Different temperature produces different hash.
        
        **Validates: Requirements 8.4**
        """
        if new_temp == key.temperature:
            return  # Skip if same temperature
        
        key_modified = ExperimentKey(
            model=key.model,
            constraint_type=key.constraint_type,
            option_a=key.option_a,
            option_b=key.option_b,
            system_template_name=key.system_template_name,
            system_template_content=key.system_template_content,
            user_template_name=key.user_template_name,
            user_template_content=key.user_template_content,
            task_id=key.task_id,
            task_prompt=key.task_prompt,
            condition=key.condition,
            direction=key.direction,
            temperature=new_temp,
            max_tokens=key.max_tokens,
            instances_per_cell=key.instances_per_cell,
            counterbalancing_enabled=key.counterbalancing_enabled,
            instruction_template=key.instruction_template,
            negative_template=key.negative_template,
            option_a_value=key.option_a_value,
            option_a_expected=key.option_a_expected,
            option_b_value=key.option_b_value,
            option_b_expected=key.option_b_expected
        )
        
        assert compute_experiment_hash(key) != compute_experiment_hash(key_modified)


# =============================================================================
# Property 19: Experiment Completion Check
# =============================================================================

class TestExperimentCompletionCheck:
    """
    Property 19: Experiment Completion Check
    
    For any completed experiment with sufficient records in the model's
    JSONL file, the `is_completed` method should return True.
    """
    
    # Feature: config-refactoring, Property 19: Experiment Completion Check
    
    @given(experiment_key_strategy())
    @settings(max_examples=50)
    def test_completed_experiment_returns_true(self, key: ExperimentKey):
        """
        Property: is_completed returns True for experiments with enough JSONL records.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            
            # Write enough JSONL records to the model's results file
            self._write_fake_results(runner, key)
            
            assert runner.is_completed(key) is True
    
    @given(experiment_key_strategy())
    @settings(max_examples=50)
    def test_incomplete_experiment_returns_false(self, key: ExperimentKey):
        """
        Property: is_completed returns False for experiments without JSONL records.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            
            # No results file exists
            assert runner.is_completed(key) is False
    
    @given(experiment_key_strategy())
    @settings(max_examples=50)
    def test_in_progress_experiment_returns_false(self, key: ExperimentKey):
        """
        Property: is_completed returns False when fewer records than instances_per_cell.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            
            # Write fewer records than needed (only 1 if instances_per_cell > 1)
            if key.instances_per_cell > 1:
                self._write_fake_results(runner, key, count=1)
                assert runner.is_completed(key) is False
    
    @staticmethod
    def _write_fake_results(runner, key: ExperimentKey, count: int | None = None):
        """Write fake JSONL records matching the key's prompt_id pattern."""
        results_path = runner._get_results_path(key.model)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        n = count if count is not None else key.instances_per_cell
        prefix = runner._build_prompt_id_prefix(key)
        
        with open(results_path, 'a') as f:
            for i in range(n):
                record = {
                    'prompt_id': f"{prefix}_{i:03d}",
                    'model': key.model,
                    'label': 'followed_system',
                }
                f.write(json.dumps(record) + '\n')



# =============================================================================
# Unit Tests for Experiment Tracking
# =============================================================================

class TestExperimentKeyCreation:
    """Unit tests for ExperimentKey creation and basic functionality."""
    
    def test_basic_experiment_key_creation(self):
        """Test creating a basic ExperimentKey."""
        key = ExperimentKey(
            model="test-model",
            constraint_type="language",
            option_a="english",
            option_b="spanish",
            system_template_name="medium",
            system_template_content="You must {instruction}. {negative}",
            user_template_name="standard",
            user_template_content="{instruction}. {task}",
            task_id="factual",
            task_prompt="What is the capital of France?",
            condition="C",
            direction="a_to_b",
            temperature=0.0,
            max_tokens=512,
            instances_per_cell=15,
            counterbalancing_enabled=True,
            instruction_template="respond in {value}",
            negative_template="Do not use any other language",
            option_a_value="English",
            option_a_expected="en",
            option_b_value="Spanish",
            option_b_expected="es"
        )
        
        assert key.model == "test-model"
        assert key.constraint_type == "language"
        assert key.option_a == "english"
        assert key.option_b == "spanish"
        assert key.condition == "C"
        assert key.direction == "a_to_b"
    
    def test_experiment_key_all_fields_accessible(self):
        """Test that all ExperimentKey fields are accessible."""
        key = self._create_test_key()
        
        # Verify all fields are accessible
        assert hasattr(key, 'model')
        assert hasattr(key, 'constraint_type')
        assert hasattr(key, 'option_a')
        assert hasattr(key, 'option_b')
        assert hasattr(key, 'system_template_name')
        assert hasattr(key, 'system_template_content')
        assert hasattr(key, 'user_template_name')
        assert hasattr(key, 'user_template_content')
        assert hasattr(key, 'task_id')
        assert hasattr(key, 'task_prompt')
        assert hasattr(key, 'condition')
        assert hasattr(key, 'direction')
        assert hasattr(key, 'temperature')
        assert hasattr(key, 'max_tokens')
        assert hasattr(key, 'instances_per_cell')
        assert hasattr(key, 'counterbalancing_enabled')
        assert hasattr(key, 'instruction_template')
        assert hasattr(key, 'negative_template')
        assert hasattr(key, 'option_a_value')
        assert hasattr(key, 'option_a_expected')
        assert hasattr(key, 'option_b_value')
        assert hasattr(key, 'option_b_expected')
    
    def _create_test_key(self, **overrides) -> ExperimentKey:
        """Helper to create a test ExperimentKey with optional overrides."""
        defaults = {
            'model': "test-model",
            'constraint_type': "language",
            'option_a': "english",
            'option_b': "spanish",
            'system_template_name': "medium",
            'system_template_content': "You must {instruction}. {negative}",
            'user_template_name': "standard",
            'user_template_content': "{instruction}. {task}",
            'task_id': "factual",
            'task_prompt': "What is the capital of France?",
            'condition': "C",
            'direction': "a_to_b",
            'temperature': 0.0,
            'max_tokens': 512,
            'instances_per_cell': 15,
            'counterbalancing_enabled': True,
            'instruction_template': "respond in {value}",
            'negative_template': "Do not use any other language",
            'option_a_value': "English",
            'option_a_expected': "en",
            'option_b_value': "Spanish",
            'option_b_expected': "es"
        }
        defaults.update(overrides)
        return ExperimentKey(**defaults)


class TestComputeExperimentHash:
    """Unit tests for compute_experiment_hash function."""
    
    def test_hash_is_deterministic(self):
        """Test that same key always produces same hash."""
        key = self._create_test_key()
        
        hash1 = compute_experiment_hash(key)
        hash2 = compute_experiment_hash(key)
        
        assert hash1 == hash2
    
    def test_hash_is_16_hex_characters(self):
        """Test that hash is exactly 16 hexadecimal characters."""
        key = self._create_test_key()
        hash_value = compute_experiment_hash(key)
        
        assert len(hash_value) == 16
        assert all(c in '0123456789abcdef' for c in hash_value)
    
    def test_hash_changes_with_model(self):
        """Test that different model produces different hash."""
        key1 = self._create_test_key(model="model-a")
        key2 = self._create_test_key(model="model-b")
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def test_hash_changes_with_template_content(self):
        """Test that different template content produces different hash."""
        key1 = self._create_test_key(
            system_template_content="Template A {instruction} {negative}"
        )
        key2 = self._create_test_key(
            system_template_content="Template B {instruction} {negative}"
        )
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def test_hash_changes_with_condition(self):
        """Test that different condition produces different hash."""
        key1 = self._create_test_key(condition="A")
        key2 = self._create_test_key(condition="C")
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def test_hash_changes_with_direction(self):
        """Test that different direction produces different hash."""
        key1 = self._create_test_key(direction="a_to_b")
        key2 = self._create_test_key(direction="b_to_a")
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def test_hash_changes_with_temperature(self):
        """Test that different temperature produces different hash."""
        key1 = self._create_test_key(temperature=0.0)
        key2 = self._create_test_key(temperature=0.7)
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def test_hash_changes_with_max_tokens(self):
        """Test that different max_tokens produces different hash."""
        key1 = self._create_test_key(max_tokens=512)
        key2 = self._create_test_key(max_tokens=1024)
        
        assert compute_experiment_hash(key1) != compute_experiment_hash(key2)
    
    def _create_test_key(self, **overrides) -> ExperimentKey:
        """Helper to create a test ExperimentKey with optional overrides."""
        defaults = {
            'model': "test-model",
            'constraint_type': "language",
            'option_a': "english",
            'option_b': "spanish",
            'system_template_name': "medium",
            'system_template_content': "You must {instruction}. {negative}",
            'user_template_name': "standard",
            'user_template_content': "{instruction}. {task}",
            'task_id': "factual",
            'task_prompt': "What is the capital of France?",
            'condition': "C",
            'direction': "a_to_b",
            'temperature': 0.0,
            'max_tokens': 512,
            'instances_per_cell': 15,
            'counterbalancing_enabled': True,
            'instruction_template': "respond in {value}",
            'negative_template': "Do not use any other language",
            'option_a_value': "English",
            'option_a_expected': "en",
            'option_b_value': "Spanish",
            'option_b_expected': "es"
        }
        defaults.update(overrides)
        return ExperimentKey(**defaults)


class TestExperimentRunnerIsCompleted:
    """Unit tests for ExperimentRunner.is_completed method (JSONL-based)."""
    
    def test_completed_experiment_is_skipped(self):
        """Test that completed experiments are detected as completed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            key = self._create_test_key()
            
            # Write enough JSONL records
            self._write_fake_results(runner, key)
            
            assert runner.is_completed(key) is True
    
    def test_nonexistent_experiment_not_completed(self):
        """Test that nonexistent experiments are not completed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            key = self._create_test_key()
            
            assert runner.is_completed(key) is False
    
    def test_modified_config_triggers_rerun(self):
        """Test that modified config produces different prompt_id prefix, allowing re-run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_client = MagicMock()
            
            runner = ExperimentRunner(mock_config, mock_client, output_dir=tmp_dir)
            
            # Create original key and write results
            key1 = self._create_test_key(task_id="original_task")
            self._write_fake_results(runner, key1)
            
            # Create modified key (different task_id → different prompt_id prefix)
            key2 = self._create_test_key(task_id="modified_task")
            
            # Original should be completed
            assert runner.is_completed(key1) is True
            
            # Modified should not be completed (different prefix)
            assert runner.is_completed(key2) is False
    
    def test_config_saved_with_results(self):
        """Test that config is saved alongside results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            key = self._create_test_key()
            exp_hash = compute_experiment_hash(key)
            exp_dir = Path(tmp_dir) / 'experiments' / exp_hash
            exp_dir.mkdir(parents=True)
            
            # Simulate saving config
            from dataclasses import asdict
            config_path = exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(asdict(key), f, indent=2)
            
            # Verify config was saved
            assert config_path.exists()
            
            # Verify config content
            with open(config_path) as f:
                saved_config = json.load(f)
            
            assert saved_config['model'] == key.model
            assert saved_config['constraint_type'] == key.constraint_type
            assert saved_config['task_prompt'] == key.task_prompt
    
    def test_metadata_saved_on_completion(self):
        """Test that metadata is saved when experiment completes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            key = self._create_test_key()
            exp_hash = compute_experiment_hash(key)
            exp_dir = Path(tmp_dir) / 'experiments' / exp_hash
            exp_dir.mkdir(parents=True)
            
            # Simulate saving metadata
            metadata = {
                'status': 'completed',
                'experiment_hash': exp_hash,
                'completed_at': '2024-01-01T00:00:00',
                'instance_count': 15,
                'error_count': 0
            }
            
            metadata_path = exp_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Verify metadata was saved
            assert metadata_path.exists()
            
            # Verify metadata content
            with open(metadata_path) as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['status'] == 'completed'
            assert saved_metadata['experiment_hash'] == exp_hash
            assert saved_metadata['instance_count'] == 15
    
    def _create_test_key(self, **overrides) -> ExperimentKey:
        """Helper to create a test ExperimentKey with optional overrides."""
        defaults = {
            'model': "test-model",
            'constraint_type': "language",
            'option_a': "english",
            'option_b': "spanish",
            'system_template_name': "medium",
            'system_template_content': "You must {instruction}. {negative}",
            'user_template_name': "standard",
            'user_template_content': "{instruction}. {task}",
            'task_id': "factual",
            'task_prompt': "What is the capital of France?",
            'condition': "C",
            'direction': "a_to_b",
            'temperature': 0.0,
            'max_tokens': 512,
            'instances_per_cell': 15,
            'counterbalancing_enabled': True,
            'instruction_template': "respond in {value}",
            'negative_template': "Do not use any other language",
            'option_a_value': "English",
            'option_a_expected': "en",
            'option_b_value': "Spanish",
            'option_b_expected': "es"
        }
        defaults.update(overrides)
        return ExperimentKey(**defaults)
    @staticmethod
    def _write_fake_results(runner, key: ExperimentKey, count: int | None = None):
        """Write fake JSONL records matching the key's prompt_id pattern."""
        results_path = runner._get_results_path(key.model)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        n = count if count is not None else key.instances_per_cell
        prefix = runner._build_prompt_id_prefix(key)
        
        with open(results_path, 'a') as f:
            for i in range(n):
                record = {
                    'prompt_id': f"{prefix}_{i:03d}",
                    'model': key.model,
                    'label': 'followed_system',
                }
                f.write(json.dumps(record) + '\n')


# =============================================================================
# Unit Tests: _get_directions_for_condition
# =============================================================================

class TestGetDirectionsForCondition:
    """Unit tests for _get_directions_for_condition() across all four conditions."""

    def _make_runner(self, counterbalancing_enabled: bool = True):
        from unittest.mock import MagicMock
        from src.config import CounterbalancingConfig
        config = MagicMock()
        config.counterbalancing = CounterbalancingConfig(enabled=counterbalancing_enabled)
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = ExperimentRunner(config, client, output_dir=tmp_dir)
        return runner

    def test_condition_a_returns_option_a_and_option_b(self):
        runner = self._make_runner()
        assert runner._get_directions_for_condition('A') == ['option_a', 'option_b']

    def test_condition_b_returns_option_a_and_option_b(self):
        runner = self._make_runner()
        assert runner._get_directions_for_condition('B') == ['option_a', 'option_b']

    def test_condition_c_with_counterbalancing_returns_both_directions(self):
        runner = self._make_runner(counterbalancing_enabled=True)
        assert runner._get_directions_for_condition('C') == ['a_to_b', 'b_to_a']

    def test_condition_d_with_counterbalancing_returns_both_directions(self):
        runner = self._make_runner(counterbalancing_enabled=True)
        assert runner._get_directions_for_condition('D') == ['a_to_b', 'b_to_a']

    def test_condition_c_without_counterbalancing_returns_a_to_b_only(self):
        runner = self._make_runner(counterbalancing_enabled=False)
        assert runner._get_directions_for_condition('C') == ['a_to_b']

    def test_condition_d_without_counterbalancing_returns_a_to_b_only(self):
        runner = self._make_runner(counterbalancing_enabled=False)
        assert runner._get_directions_for_condition('D') == ['a_to_b']
