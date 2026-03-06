"""Lambda Cloud configuration types and YAML loader."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class LambdaConfig:
    """Configuration for a Lambda Cloud vLLM deployment."""
    api_key: str
    ssh_key_name: str
    model_id: str
    instance_type: str
    region: str
    hf_token: str
    ssh_key_file: str = "~/.ssh/id_rsa"
    vllm_venv_path: str = "/home/ubuntu/vllm-venv"
    vllm_port: int = 8000
    vllm_extra_args: str = ""
    max_launch_retries: int = 5
    launch_retry_delay: int = 60
    readiness_timeout: int = 900
    concurrent_per_model: int = 10
    instance_preferences: list[str] | None = None
    poll_interval: int = 10


@dataclass
class LambdaInstance:
    """A running Lambda Cloud instance."""
    instance_id: str
    ip: str
    status: str


def load_lambda_config(path: str | Path, model_id: str) -> LambdaConfig:
    """Load Lambda config from YAML and merge model-specific GPU settings.

    Reads LAMBDA_API_KEY and HF_TOKEN from environment variables.

    Args:
        path: Path to lambda.yaml config file.
        model_id: The model to deploy (used for GPU mapping lookup).

    Returns:
        LambdaConfig ready for LambdaCloudManager.

    Raises:
        ValueError: If required env vars are missing.
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lambda config not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        raise ValueError(
            "LAMBDA_API_KEY env var not set. "
            "Generate at: Lambda Cloud Dashboard -> Settings -> API Keys"
        )
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN env var not set.")

    defaults = data.get("defaults", {})
    gpu_map = data.get("model_gpu_map", {})

    # Look up model-specific settings, fall back to _default
    model_settings = gpu_map.get(model_id, gpu_map.get("_default", {}))

    # Build instance preferences: model-specific list, then global, then just the primary type
    primary_type = model_settings.get(
        "instance_type", defaults.get("instance_type", "gpu_1x_a100")
    )
    instance_prefs = model_settings.get(
        "instance_preferences",
        data.get("instance_preferences"),
    )
    # Ensure primary type is first in the list
    if instance_prefs:
        if primary_type not in instance_prefs:
            instance_prefs = [primary_type] + list(instance_prefs)
    else:
        instance_prefs = [primary_type]

    return LambdaConfig(
        api_key=api_key,
        ssh_key_name=data.get("ssh_key_name", ""),
        model_id=model_id,
        instance_type=primary_type,
        region=defaults.get("region", "us-east-1"),
        hf_token=hf_token,
        ssh_key_file=data.get("ssh_key_file", "~/.ssh/id_rsa"),
        vllm_venv_path=defaults.get("vllm_venv_path", "/home/ubuntu/vllm-venv"),
        vllm_port=defaults.get("vllm_port", 8000),
        vllm_extra_args=model_settings.get("vllm_args", ""),
        max_launch_retries=defaults.get("max_launch_retries", 5),
        launch_retry_delay=defaults.get("launch_retry_delay", 60),
        readiness_timeout=defaults.get("readiness_timeout", 900),
        concurrent_per_model=defaults.get("concurrent_per_model", 10),
        instance_preferences=instance_prefs,
        poll_interval=defaults.get("poll_interval", 10),
    )
