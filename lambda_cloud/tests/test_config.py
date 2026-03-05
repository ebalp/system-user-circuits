"""Tests for lambda_cloud.config — LambdaConfig, LambdaInstance, load_lambda_config."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lambda_cloud.config import LambdaConfig, LambdaInstance, load_lambda_config


class TestLambdaConfig:
    def test_defaults(self):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="gpu_1x_a10", region="us-east-1",
            hf_token="t",
        )
        assert config.vllm_port == 8000
        assert config.vllm_extra_args == ""
        assert config.max_launch_retries == 5
        assert config.launch_retry_delay == 60
        assert config.readiness_timeout == 900
        assert config.ssh_key_file == "~/.ssh/anusha-cre-lambda-key.pem"
        assert config.vllm_venv_path == "/home/ubuntu/vllm-venv"

    def test_custom_values(self):
        config = LambdaConfig(
            api_key="test-api-key",
            ssh_key_name="test-key",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            instance_type="gpu_1x_a10",
            region="us-east-1",
            hf_token="hf-test-token",
            vllm_extra_args="--max-model-len 4096",
        )
        assert config.api_key == "test-api-key"
        assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.vllm_extra_args == "--max-model-len 4096"


class TestLambdaInstance:
    def test_fields(self):
        inst = LambdaInstance(instance_id="i-123", ip="10.0.0.1", status="active")
        assert inst.instance_id == "i-123"
        assert inst.ip == "10.0.0.1"
        assert inst.status == "active"


class TestLoadLambdaConfig:
    def test_loads_known_model(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "test-key"
defaults:
  region: us-west-2
  vllm_port: 9000
  readiness_timeout: 600
model_gpu_map:
  test-model:
    instance_type: gpu_1x_a10
    vllm_args: "--max-model-len 2048"
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "key123", "HF_TOKEN": "hf123"}):
            config = load_lambda_config(config_yaml, "test-model")
        assert config.instance_type == "gpu_1x_a10"
        assert config.vllm_extra_args == "--max-model-len 2048"
        assert config.region == "us-west-2"
        assert config.vllm_port == 9000
        assert config.ssh_key_name == "test-key"

    def test_falls_back_to_default(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults:
  region: us-east-1
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
    vllm_args: "--max-model-len 4096"
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "key", "HF_TOKEN": "hf"}):
            config = load_lambda_config(config_yaml, "unknown-model")
        assert config.instance_type == "gpu_1x_a100"
        assert config.model_id == "unknown-model"

    def test_missing_api_key_raises(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("ssh_key_name: k\nmodel_gpu_map: {}\ndefaults: {}")
        env = {"HF_TOKEN": "hf"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("LAMBDA_API_KEY", None)
            with pytest.raises(ValueError, match="LAMBDA_API_KEY"):
                load_lambda_config(config_yaml, "m")

    def test_missing_hf_token_raises(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("ssh_key_name: k\nmodel_gpu_map: {}\ndefaults: {}")
        env = {"LAMBDA_API_KEY": "key"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(ValueError, match="HF_TOKEN"):
                load_lambda_config(config_yaml, "m")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_lambda_config("/nonexistent/lambda.yaml", "m")

    def test_reads_env_vars(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults: {}
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "my-api-key", "HF_TOKEN": "my-hf-token"}):
            config = load_lambda_config(config_yaml, "m")
        assert config.api_key == "my-api-key"
        assert config.hf_token == "my-hf-token"

    def test_real_lambda_yaml_parses(self):
        """The actual lambda.yaml in the repo should parse without error."""
        yaml_path = Path("lambda_cloud/config/lambda.yaml")
        if yaml_path.exists():
            with patch.dict(os.environ, {"LAMBDA_API_KEY": "test", "HF_TOKEN": "test"}):
                config = load_lambda_config(yaml_path, "meta-llama/Llama-3.1-8B-Instruct")
            assert config.instance_type == "gpu_1x_a10"
            assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"

    def test_defaults_applied_when_missing_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults: {}
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "m")
        assert config.region == "us-east-1"
        assert config.vllm_port == 8000
        assert config.max_launch_retries == 5
        assert config.launch_retry_delay == 60
        assert config.readiness_timeout == 900

    def test_no_model_or_default_uses_hardcoded_fallback(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults: {}
model_gpu_map: {}
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "some-model")
        assert config.instance_type == "gpu_1x_a100"
        assert config.vllm_extra_args == ""

    def test_accepts_string_path(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults: {}
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(str(config_yaml), "m")
        assert config.instance_type == "gpu_1x_a100"

    def test_model_id_preserved(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults: {}
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "my/custom-model")
        assert config.model_id == "my/custom-model"

    def test_readiness_timeout_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults:
  readiness_timeout: 1200
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "m")
        assert config.readiness_timeout == 1200

    def test_ssh_key_file_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
ssh_key_file: "~/.ssh/custom-key.pem"
defaults: {}
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "m")
        assert config.ssh_key_file == "~/.ssh/custom-key.pem"

    def test_vllm_venv_path_from_yaml(self, tmp_path):
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults:
  vllm_venv_path: /opt/vllm-env
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            config = load_lambda_config(config_yaml, "m")
        assert config.vllm_venv_path == "/opt/vllm-env"
