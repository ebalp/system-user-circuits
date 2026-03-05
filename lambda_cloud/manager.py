"""Lambda Cloud instance lifecycle management.

Automates: launch GPU instance -> poll for IP -> terminate.

Safety nets ensure instances are always terminated, even on crashes:
1. Context manager __exit__
2. atexit handler
3. SIGINT/SIGTERM signal handlers
"""

import atexit
import logging
import signal
import time

import httpx

from lambda_cloud.config import LambdaConfig, LambdaInstance

logger = logging.getLogger(__name__)


class LambdaCloudManager:
    """Context manager for Lambda Cloud GPU instance lifecycle.

    Usage:
        config = load_lambda_config("lambda_cloud/config/lambda.yaml", model_id)
        with LambdaCloudManager(config) as manager:
            # manager.instance has .ip and .instance_id
            # set up vLLM via SSH, run experiments, etc.
        # instance is terminated automatically
    """

    BASE_URL = "https://cloud.lambda.ai/api/v1"

    def __init__(self, config: LambdaConfig):
        self.config = config
        self.instance: LambdaInstance | None = None
        self._terminated = False
        self._original_sigint = None
        self._original_sigterm = None

    def _auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(username=self.config.api_key, password="")

    def get_base_url(self) -> str:
        """Return the vLLM base URL for the running instance.

        Returns:
            http://{ip}:{port}/v1

        Raises:
            RuntimeError: If no instance is running.
        """
        if self.instance is None:
            raise RuntimeError("No instance launched — call launch() first")
        return f"http://{self.instance.ip}:{self.config.vllm_port}/v1"

    def launch(self) -> LambdaInstance:
        """Launch a Lambda Cloud instance.

        Retries on availability errors (no capacity). Polls until
        the instance has an IP address.

        Returns:
            LambdaInstance with id and ip populated.

        Raises:
            RuntimeError: If all launch retries are exhausted or instance
                never gets an IP.
        """
        for attempt in range(1, self.config.max_launch_retries + 1):
            logger.info(
                "Launching %s in %s (attempt %d/%d)",
                self.config.instance_type, self.config.region,
                attempt, self.config.max_launch_retries,
            )
            try:
                resp = httpx.post(
                    f"{self.BASE_URL}/instance-operations/launch",
                    auth=self._auth(),
                    json={
                        "region_name": self.config.region,
                        "instance_type_name": self.config.instance_type,
                        "ssh_key_names": [self.config.ssh_key_name],
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
                    auth=self._auth(),
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
                auth=self._auth(),
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
            auth=self._auth(),
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        return False  # don't suppress exceptions
