"""Lambda Cloud instance lifecycle management for vLLM deployment.

Automates: launch GPU instance -> install vLLM via cloud-init ->
wait for model to load -> run experiments -> terminate.

Safety nets ensure instances are always terminated, even on crashes:
1. Context manager __exit__
2. atexit handler
3. SIGINT/SIGTERM signal handlers
"""

import atexit
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LambdaConfig:
    """Configuration for a Lambda Cloud vLLM deployment."""
    api_key: str
    ssh_key_name: str
    model_id: str
    instance_type: str
    region: str
    hf_token: str
    vllm_port: int = 8000
    vllm_extra_args: str = ""
    max_launch_retries: int = 5
    launch_retry_delay: int = 60
    readiness_timeout: int = 900


@dataclass
class LambdaInstance:
    """A running Lambda Cloud instance."""
    instance_id: str
    ip: str
    status: str


class LambdaCloudManager:
    """Context manager for Lambda Cloud GPU instance lifecycle.

    Usage:
        config = load_lambda_config("phase0_v2/config/lambda.yaml", model_id)
        with LambdaCloudManager(config) as manager:
            client = manager.get_client()
            # ... use client for experiments ...
        # instance is terminated automatically
    """

    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, config: LambdaConfig):
        self.config = config
        self.instance: LambdaInstance | None = None
        self._terminated = False
        self._original_sigint = None
        self._original_sigterm = None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _build_cloud_init(self) -> str:
        """Generate cloud-init script that installs and starts vLLM."""
        return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/vllm-setup.log 2>&1
pip install --upgrade pip
pip install vllm
export HF_TOKEN="{self.config.hf_token}"
nohup vllm serve {self.config.model_id} \\
    --host 0.0.0.0 --port {self.config.vllm_port} \\
    {self.config.vllm_extra_args} \\
    > /var/log/vllm-server.log 2>&1 &
"""

    def launch(self) -> LambdaInstance:
        """Launch a Lambda Cloud instance with cloud-init for vLLM.

        Retries on availability errors (no capacity). Polls until
        the instance has an IP address.

        Returns:
            LambdaInstance with id and ip populated.

        Raises:
            RuntimeError: If all launch retries are exhausted or instance
                never gets an IP.
        """
        cloud_init = self._build_cloud_init()

        for attempt in range(1, self.config.max_launch_retries + 1):
            logger.info(
                "Launching %s in %s (attempt %d/%d)",
                self.config.instance_type, self.config.region,
                attempt, self.config.max_launch_retries,
            )
            try:
                resp = httpx.post(
                    f"{self.BASE_URL}/instance-operations/launch",
                    headers=self._headers(),
                    json={
                        "region_name": self.config.region,
                        "instance_type_name": self.config.instance_type,
                        "ssh_key_names": [self.config.ssh_key_name],
                        "user_data": cloud_init,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                instance_ids = data.get("data", {}).get("instance_ids", [])
                if not instance_ids:
                    raise RuntimeError(f"No instance IDs in launch response: {data}")
                instance_id = instance_ids[0]
                logger.info("Instance launched: %s", instance_id)
                break
            except (httpx.HTTPStatusError, RuntimeError) as e:
                error_msg = str(e)
                if attempt < self.config.max_launch_retries:
                    logger.warning(
                        "Launch failed (%s), retrying in %ds...",
                        error_msg, self.config.launch_retry_delay,
                    )
                    time.sleep(self.config.launch_retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to launch after {self.config.max_launch_retries} attempts: {error_msg}"
                    )

        # Poll for IP address
        ip = self._poll_for_ip(instance_id)
        self.instance = LambdaInstance(
            instance_id=instance_id, ip=ip, status="active"
        )

        # Install safety nets
        self._install_safety_nets()

        return self.instance

    def _poll_for_ip(self, instance_id: str, timeout: int = 300, interval: int = 10) -> str:
        """Poll GET /instances/{id} until the instance has an IP."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(
                    f"{self.BASE_URL}/instances/{instance_id}",
                    headers=self._headers(),
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                ip = data.get("ip")
                status = data.get("status", "unknown")
                if ip:
                    logger.info("Instance %s ready at %s (status: %s)",
                                instance_id, ip, status)
                    return ip
                logger.info("Instance %s status: %s, waiting for IP...",
                            instance_id, status)
            except Exception as e:
                logger.warning("Error polling instance %s: %s", instance_id, e)
            time.sleep(interval)
        raise RuntimeError(
            f"Instance {instance_id} did not get an IP within {timeout}s"
        )

    def wait_for_ready(self) -> None:
        """Wait for vLLM server to be ready (model loaded).

        Delegates to VLLMClient.wait_until_ready().

        Raises:
            RuntimeError: If no instance is running or vLLM doesn't become ready.
        """
        if self.instance is None:
            raise RuntimeError("No instance launched — call launch() first")

        from .api_client import VLLMClient

        client = VLLMClient(
            base_url=f"http://{self.instance.ip}:{self.config.vllm_port}/v1"
        )
        ready = client.wait_until_ready(
            timeout_seconds=self.config.readiness_timeout
        )
        if not ready:
            raise RuntimeError(
                f"vLLM server on {self.instance.ip} not ready after "
                f"{self.config.readiness_timeout}s"
            )

    def get_client(self):
        """Return a VLLMClient connected to the running instance.

        Returns:
            VLLMClient pointed at http://{ip}:{port}/v1

        Raises:
            RuntimeError: If no instance is running.
        """
        if self.instance is None:
            raise RuntimeError("No instance launched — call launch() first")

        from .api_client import VLLMClient

        return VLLMClient(
            base_url=f"http://{self.instance.ip}:{self.config.vllm_port}/v1"
        )

    def terminate(self) -> None:
        """Terminate the Lambda Cloud instance. Idempotent.

        Safe to call multiple times — only sends the API request once.
        """
        if self._terminated or self.instance is None:
            return

        logger.info("Terminating instance %s", self.instance.instance_id)
        try:
            resp = httpx.post(
                f"{self.BASE_URL}/instance-operations/terminate",
                headers=self._headers(),
                json={"instance_ids": [self.instance.instance_id]},
                timeout=30,
            )
            resp.raise_for_status()
            self._terminated = True
            logger.info("Instance %s terminated", self.instance.instance_id)
        except Exception as e:
            logger.error(
                "Failed to terminate instance %s: %s",
                self.instance.instance_id, e,
            )

    def list_available(self) -> list[dict]:
        """List available GPU instance types and pricing.

        Returns:
            List of dicts with instance type info from Lambda API.
        """
        resp = httpx.get(
            f"{self.BASE_URL}/instance-types",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        result = []
        for type_name, info in data.items():
            instance_type = info.get("instance_type", {})
            regions = info.get("regions_with_capacity_available", [])
            result.append({
                "name": type_name,
                "description": instance_type.get("description", ""),
                "price_cents_per_hour": instance_type.get("price_cents_per_hour"),
                "gpu_count": instance_type.get("specs", {}).get("gpus"),
                "available_regions": [r.get("name") for r in regions],
            })
        return result

    def _install_safety_nets(self) -> None:
        """Install atexit and signal handlers for guaranteed termination."""
        atexit.register(self._atexit_cleanup)

        def _signal_handler(signum, frame):
            logger.warning("Received signal %s, terminating instance...", signum)
            self.terminate()
            # Re-raise with original handler
            if signum == signal.SIGINT and self._original_sigint:
                self._original_sigint(signum, frame)
            elif signum == signal.SIGTERM and self._original_sigterm:
                self._original_sigterm(signum, frame)
            else:
                raise SystemExit(1)

        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    def _atexit_cleanup(self) -> None:
        """atexit handler — terminate if not already done."""
        if not self._terminated:
            logger.warning("atexit: terminating instance (was not cleanly shut down)")
            self.terminate()

    def __enter__(self):
        self.launch()
        self.wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        return False  # don't suppress exceptions


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

    return LambdaConfig(
        api_key=api_key,
        ssh_key_name=data.get("ssh_key_name", ""),
        model_id=model_id,
        instance_type=model_settings.get(
            "instance_type", defaults.get("instance_type", "gpu_1x_a100")
        ),
        region=defaults.get("region", "us-east-1"),
        hf_token=hf_token,
        vllm_port=defaults.get("vllm_port", 8000),
        vllm_extra_args=model_settings.get("vllm_args", ""),
        max_launch_retries=defaults.get("max_launch_retries", 5),
        launch_retry_delay=defaults.get("launch_retry_delay", 60),
        readiness_timeout=defaults.get("readiness_timeout", 900),
    )
