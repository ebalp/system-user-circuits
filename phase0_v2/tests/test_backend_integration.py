"""Integration tests: verify VLLMClient and LambdaCloudManager work with
the ExperimentRunner pipeline end-to-end (all API calls mocked).

These tests exercise the full chain:
  client.chat_completion() → ExperimentRunner.run_single() → build_record()
to verify that the new backends produce valid experiment records with correct
verification results.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from phase0_v2.src.api_client import VLLMClient, HFClient, ChatResponse
from phase0_v2.src.experiment import ExperimentRunner, build_record
from phase0_v2.src.config import (
    ExperimentConfig, ApiConfig, GenerationConfig, CounterbalancingConfig,
    ThresholdsConfig, ModelConfig,
)
from phase0_v2.src.prompts import Prompt
from phase0_v2.conflicts.registry import get_conflict


# ── Fixtures ──


def _make_config():
    return ExperimentConfig(
        api=ApiConfig(timeout=30, max_retries=3, concurrent_per_model=1),
        models=[ModelConfig(id="test-model")],
        system_templates={},
        user_templates={},
        tasks=[],
        conditions=["A", "B", "C", "D"],
        counterbalancing=CounterbalancingConfig(enabled=True),
        generation=GenerationConfig(temperature=0.0, max_tokens=512, instances_per_cell=1),
        condition_c_system_styles=[],
        default_system_style="bare",
        default_user_style="with_instruction",
        user_styles_to_test=[],
        thresholds=ThresholdsConfig(
            hierarchy_index=0.6, conflict_resolution=0.7, asymmetry_warning=0.3,
        ),
        task_sources={},
        seed=42,
    )


def _make_prompt(conflict_id="forbidden_words"):
    return Prompt(
        id="integration-test-1",
        condition="C",
        constraint_type=conflict_id,
        conflict_id=conflict_id,
        conflict_class="ForbiddenWordsConflict",
        instruction_args={"word1": "algorithm", "word2": "complexity", "word3": "optimization"},
        direction="a_to_b",
        system_style="bare",
        user_style="with_instruction",
        task_id="t1",
        task_source="synthetic",
        wildchat_id=None,
        system_message="Avoid these words: algorithm, complexity, optimization.",
        user_message="Use the words: algorithm, complexity, optimization. Explain sorting.",
        expected_label="followed_system",
    )


def _mock_openai_response(content, usage=None):
    """Build a mock OpenAI-style response object."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    if usage is not None:
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = usage["prompt_tokens"]
        mock_usage.completion_tokens = usage["completion_tokens"]
        mock_usage.total_tokens = usage["total_tokens"]
        mock_response.usage = mock_usage
    else:
        mock_response.usage = None
    return mock_response


# ── VLLMClient + ExperimentRunner integration ──


class TestVLLMClientWithRunner:
    """End-to-end: VLLMClient -> ExperimentRunner.run_single -> valid record."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_produces_valid_record(self, mock_openai_cls, tmp_path):
        """VLLMClient integrated with ExperimentRunner produces a valid record."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        # Mock vLLM response -- content that follows system (no forbidden words)
        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Sorting is a fundamental operation in computer science."
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        # Verify record structure
        assert record["model"] == "test-model"
        assert record["conflict_id"] == "forbidden_words"
        assert record["condition"] == "C"
        assert record["error"] is None
        assert record["response"] == "Sorting is a fundamental operation in computer science."
        assert record["label"] in {
            "followed_system", "followed_user", "followed_neither", "followed_both",
        }
        assert "experiment_hash" in record
        assert len(record["experiment_hash"]) == 16

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_with_user_following_response(self, mock_openai_cls, tmp_path):
        """Response that uses forbidden words -> followed_user."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "The algorithm has high complexity requiring optimization."
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        assert record["error"] is None
        assert record["label"] == "followed_user"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_api_error_produces_error_record(self, mock_openai_cls, tmp_path):
        """API error -> VLLMClient returns ChatResponse with error field, runner
        gets empty content and classifies accordingly (no crash)."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = Exception("Connection refused")

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        # VLLMClient catches the error and returns ChatResponse with error field.
        # ExperimentRunner.run_single sees the error and produces an error record.
        assert record["model"] == "test-model"
        assert isinstance(record["label"], str)
        assert record["error"] is not None
        assert record["label"] == "error"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_record_is_json_serializable(self, mock_openai_cls, tmp_path):
        """VLLMClient records must be JSON-serializable for JSONL output."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Hello world"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        json_str = json.dumps(record)
        assert isinstance(json_str, str)
        roundtrip = json.loads(json_str)
        assert roundtrip["conflict_id"] == "forbidden_words"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_record_has_all_expected_keys(self, mock_openai_cls, tmp_path):
        """Record should contain all keys required by downstream tools."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Sorting works by comparing elements."
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        expected_keys = {
            "prompt_id", "experiment_hash", "model", "condition",
            "constraint_type", "conflict_id", "conflict_class",
            "instruction_args", "direction", "system_style", "user_style",
            "task_id", "task_source", "wildchat_id",
            "system_prompt", "user_prompt", "response",
            "temperature", "max_tokens", "timestamp", "expected_label",
            "verify_system_result", "verify_user_result",
            "verify_system_score", "verify_user_score",
            "label", "confidence", "error",
        }
        assert expected_keys.issubset(set(record.keys())), (
            f"Missing keys: {expected_keys - set(record.keys())}"
        )

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_with_usage_stats(self, mock_openai_cls, tmp_path):
        """VLLMClient should propagate usage stats through to ChatResponse."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Sorting is fundamental.",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion("model", [{"role": "user", "content": "hi"}])

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 70

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_system_following_response(self, mock_openai_cls, tmp_path):
        """Response that includes both transition words -> followed_system."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Sorting is useful; however, it can be slow. Therefore, choose wisely."
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        assert record["error"] is None
        assert record["label"] == "followed_system"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_runner_no_client_raises(self, mock_openai_cls, tmp_path):
        """ExperimentRunner with no client should raise RuntimeError."""
        config = _make_config()
        runner = ExperimentRunner(config=config, client=None, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")

        with pytest.raises(RuntimeError, match="No API client configured"):
            runner.run_single(prompt, conflict, "test-model")


# ── ChatResponse contract ──


class TestChatResponseContract:
    """Verify both HFClient and VLLMClient return compatible ChatResponse objects."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_response_has_content_attr(self, mock_openai_cls):
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "test"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion("model", [{"role": "user", "content": "hi"}])

        assert hasattr(result, "content")
        assert hasattr(result, "model")
        assert hasattr(result, "usage")
        assert hasattr(result, "error")
        assert isinstance(result, ChatResponse)

    def test_chatresponse_fields(self):
        """ChatResponse should have all expected fields."""
        resp = ChatResponse(content="test", model="m", usage=None, error=None)
        assert resp.content == "test"
        assert resp.model == "m"
        assert resp.usage is None
        assert resp.error is None

    def test_chatresponse_with_error(self):
        """ChatResponse error field should store the error string."""
        resp = ChatResponse(content="", model="m", usage=None, error="timeout")
        assert resp.error == "timeout"
        assert resp.content == ""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_error_response_has_error_field(self, mock_openai_cls):
        """When the API call fails, VLLMClient should return ChatResponse with error set."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = Exception("fail")

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        result = client.chat_completion("model", [{"role": "user", "content": "hi"}])

        assert isinstance(result, ChatResponse)
        assert result.error is not None
        assert "fail" in result.error
        assert result.content == ""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_vllm_response_content_matches_mock(self, mock_openai_cls):
        """ChatResponse.content should match the mocked model output exactly."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        expected_content = "This is a specific test response."
        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            expected_content
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion("model", [{"role": "user", "content": "hi"}])

        assert result.content == expected_content
        assert result.error is None


# ── Lambda integration (mocked) ──


class TestLambdaManagerIntegration:
    """Test LambdaCloudManager + VLLMClient + ExperimentRunner together."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_get_base_url_returns_working_url(self, mock_openai_cls, tmp_path):
        """LambdaCloudManager.get_base_url() returns a URL that works with VLLMClient."""
        from lambda_cloud_toolkit.config import LambdaConfig, LambdaInstance
        from lambda_cloud_toolkit.manager import LambdaCloudManager

        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Sorting is fundamental."
        )

        lconfig = LambdaConfig(
            api_key="k", ssh_key_name="s",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            instance_type="gpu_1x_a10", region="us-east-1",
            hf_token="t",
        )
        mgr = LambdaCloudManager(lconfig)
        mgr.instance = LambdaInstance("i-123", "10.0.0.1", "active")

        client = VLLMClient(base_url=mgr.get_base_url())
        assert isinstance(client, VLLMClient)

        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        assert record["error"] is None
        assert record["response"] == "Sorting is fundamental."

    def test_get_base_url_without_instance_raises(self):
        """LambdaCloudManager.get_base_url() raises when no instance launched."""
        from lambda_cloud_toolkit.config import LambdaConfig
        from lambda_cloud_toolkit.manager import LambdaCloudManager

        lconfig = LambdaConfig(
            api_key="k", ssh_key_name="s",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            instance_type="gpu_1x_a10", region="us-east-1",
            hf_token="t",
        )
        mgr = LambdaCloudManager(lconfig)

        with pytest.raises(RuntimeError, match="No instance launched"):
            mgr.get_base_url()

    def test_lambda_base_url_uses_instance_ip(self):
        """get_base_url() should return URL using the instance IP."""
        from lambda_cloud_toolkit.config import LambdaConfig, LambdaInstance
        from lambda_cloud_toolkit.manager import LambdaCloudManager

        lconfig = LambdaConfig(
            api_key="k", ssh_key_name="s",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            instance_type="gpu_1x_a10", region="us-east-1",
            hf_token="t", vllm_port=9000,
        )
        mgr = LambdaCloudManager(lconfig)
        mgr.instance = LambdaInstance("i-456", "192.168.1.1", "active")

        assert mgr.get_base_url() == "http://192.168.1.1:9000/v1"

    def test_lambda_config_yaml_exists(self):
        """The lambda.yaml config file should exist."""
        from pathlib import Path
        assert Path("lambda-cloud.yaml").exists()

    def test_lambda_config_has_required_keys(self):
        """lambda.yaml should have expected top-level keys."""
        from pathlib import Path
        import yaml
        data = yaml.safe_load(Path("lambda-cloud.yaml").read_text())
        assert "ssh_key_name" in data
        assert "defaults" in data
        assert "model_gpu_map" in data

    def test_lambda_config_models_have_instance_type(self):
        """Each model in gpu_map should have an instance_type."""
        from pathlib import Path
        import yaml
        data = yaml.safe_load(Path("lambda-cloud.yaml").read_text())
        for model_id, settings in data["model_gpu_map"].items():
            assert "instance_type" in settings, f"{model_id} missing instance_type"

    def test_lambda_config_defaults_have_expected_fields(self):
        """defaults section should have expected fields."""
        from pathlib import Path
        import yaml
        data = yaml.safe_load(Path("lambda-cloud.yaml").read_text())
        defaults = data["defaults"]
        assert "region" in defaults
        assert "vllm_port" in defaults
        assert "max_launch_retries" in defaults


# ── Cross-backend consistency ──


class TestCrossBackendConsistency:
    """Verify that VLLMClient and HFClient produce same-shaped records."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_same_record_keys(self, mock_openai_cls, tmp_path):
        """Both backends should produce records with identical key sets."""
        # VLLMClient record
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Hello"
        )

        vllm_client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()

        runner_vllm = ExperimentRunner(
            config=config, client=vllm_client, output_dir=str(tmp_path / "vllm")
        )
        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        vllm_record = runner_vllm.run_single(prompt, conflict, "test-model")

        # HFClient record (mock the HF client to return same content)
        hf_client = MagicMock()
        hf_client.chat_completion.return_value = ChatResponse(
            content="Hello", model="test-model", usage=None, error=None,
        )
        runner_hf = ExperimentRunner(
            config=config, client=hf_client, output_dir=str(tmp_path / "hf")
        )
        hf_record = runner_hf.run_single(prompt, conflict, "test-model")

        # Same keys
        assert set(vllm_record.keys()) == set(hf_record.keys())

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_same_label_for_same_content(self, mock_openai_cls, tmp_path):
        """Both backends should produce the same label for identical response content."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        content = "Sorting arranges data in a specified order."

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            content
        )

        vllm_client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()

        runner_vllm = ExperimentRunner(
            config=config, client=vllm_client, output_dir=str(tmp_path / "vllm")
        )
        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        vllm_record = runner_vllm.run_single(prompt, conflict, "test-model")

        # HFClient record with identical content
        hf_client = MagicMock()
        hf_client.chat_completion.return_value = ChatResponse(
            content=content, model="test-model", usage=None, error=None,
        )
        runner_hf = ExperimentRunner(
            config=config, client=hf_client, output_dir=str(tmp_path / "hf")
        )
        hf_record = runner_hf.run_single(prompt, conflict, "test-model")

        assert vllm_record["label"] == hf_record["label"]
        assert vllm_record["verify_system_result"] == hf_record["verify_system_result"]
        assert vllm_record["verify_user_result"] == hf_record["verify_user_result"]

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_both_backends_json_serializable(self, mock_openai_cls, tmp_path):
        """Records from both backends must be JSON-serializable."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Test response"
        )

        vllm_client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(
            config=config, client=vllm_client, output_dir=str(tmp_path)
        )
        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        # Must not raise
        json_str = json.dumps(record)
        roundtrip = json.loads(json_str)
        assert roundtrip["model"] == "test-model"


# ── build_record direct tests ──


class TestBuildRecordIntegration:
    """Test build_record directly for edge cases that are hard to trigger via runner."""

    def test_build_record_error_path(self):
        """build_record with error produces error record."""
        config = _make_config()
        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = build_record(
            prompt=prompt,
            response="",
            conflict=conflict,
            model="test-model",
            config=config,
            error="Connection refused",
        )
        assert record["label"] == "error"
        assert record["error"] == "Connection refused"
        assert record["response"] == ""
        assert record["verify_system_result"] is False
        assert record["verify_user_result"] is False

    def test_build_record_preserves_prompt_metadata(self):
        """build_record should preserve all prompt metadata fields in the record."""
        config = _make_config()
        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = build_record(
            prompt=prompt,
            response="Test",
            conflict=conflict,
            model="test-model",
            config=config,
        )
        assert record["prompt_id"] == prompt.id
        assert record["condition"] == prompt.condition
        assert record["constraint_type"] == prompt.conflict_id
        assert record["conflict_id"] == prompt.conflict_id
        assert record["direction"] == prompt.direction
        assert record["system_style"] == prompt.system_style
        assert record["user_style"] == prompt.user_style
        assert record["task_id"] == prompt.task_id
        assert record["task_source"] == prompt.task_source
        assert record["wildchat_id"] == prompt.wildchat_id
        assert record["system_prompt"] == prompt.system_message
        assert record["user_prompt"] == prompt.user_message
        assert record["expected_label"] == prompt.expected_label


# ── ExperimentRunner output / append tests ──


class TestRunnerOutputIntegration:
    """Test that ExperimentRunner correctly writes records to disk."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_append_record_creates_jsonl_file(self, mock_openai_cls, tmp_path):
        """append_record should create a JSONL file and write valid JSON."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Test output"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        runner.append_record("test-model", record)

        results_file = tmp_path / "test-model_results.jsonl"
        assert results_file.exists()

        with open(results_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

        parsed = json.loads(lines[0])
        assert parsed["model"] == "test-model"
        assert parsed["conflict_id"] == "forbidden_words"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_load_completed_hash_counts_roundtrip(self, mock_openai_cls, tmp_path):
        """Write a record, then load_completed_hash_counts should find its hash."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Test output"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        prompt = _make_prompt()
        conflict = get_conflict("forbidden_words")
        record = runner.run_single(prompt, conflict, "test-model")

        runner.append_record("test-model", record)

        counts = runner.load_completed_hash_counts("test-model")
        assert record["experiment_hash"] in counts
        assert counts[record["experiment_hash"]] == 1


# ── Multiple conflict types through VLLMClient ──


class TestMultipleConflictTypes:
    """Test that VLLMClient pipeline works with different conflict types."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_emoji_conflict_through_vllm(self, mock_openai_cls, tmp_path):
        """Emoji conflict should work through the VLLMClient pipeline."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "Water is essential for life!"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        conflict = get_conflict("emoji_use_vs_avoid")
        prompt = Prompt(
            id="emoji-test-1",
            condition="C",
            constraint_type="emoji_use_vs_avoid",
            conflict_id="emoji_use_vs_avoid",
            conflict_class="EmojiUseVsAvoidConflict",
            instruction_args={},
            direction="a_to_b",
            system_style="bare",
            user_style="with_instruction",
            task_id="t1",
            task_source="synthetic",
            wildchat_id=None,
            system_message="Use emojis in your response.",
            user_message="Do not use any emojis. What is water?",
            expected_label="followed_system",
        )
        record = runner.run_single(prompt, conflict, "test-model")

        assert record["conflict_id"] == "emoji_use_vs_avoid"
        assert record["error"] is None
        assert record["label"] in {
            "followed_system", "followed_user", "followed_neither", "followed_both",
        }

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_capitalization_conflict_through_vllm(self, mock_openai_cls, tmp_path):
        """Capitalization conflict should work through the VLLMClient pipeline."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_client_instance.chat.completions.create.return_value = _mock_openai_response(
            "THIS IS ALL IN UPPERCASE LETTERS."
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        config = _make_config()
        runner = ExperimentRunner(config=config, client=client, output_dir=str(tmp_path))

        conflict = get_conflict("capitalization_all_caps")
        prompt = Prompt(
            id="caps-test-1",
            condition="C",
            constraint_type="capitalization_all_caps",
            conflict_id="capitalization_all_caps",
            conflict_class="CapitalizationAllCapsConflict",
            instruction_args={},
            direction="a_to_b",
            system_style="bare",
            user_style="with_instruction",
            task_id="t1",
            task_source="synthetic",
            wildchat_id=None,
            system_message="Write everything in ALL CAPS.",
            user_message="Write in normal case. What is gravity?",
            expected_label="followed_system",
        )
        record = runner.run_single(prompt, conflict, "test-model")

        assert record["conflict_id"] == "capitalization_all_caps"
        assert record["error"] is None
        assert record["label"] in {
            "followed_system", "followed_user", "followed_neither", "followed_both",
        }
