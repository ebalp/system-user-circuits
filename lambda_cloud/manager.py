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
        """Launch a Lambda Cloud instance, polling for availability.

        Polls the instance-types API for capacity across all preferred
        instance types and regions (like snatch.py). When capacity is
        found, launches immediately. Falls back to retry on launch
        race conditions (capacity gone between check and launch).

        Returns:
            LambdaInstance with id and ip populated.

        Raises:
            RuntimeError: If all launch retries are exhausted.
            KeyboardInterrupt: If the user cancels polling.
        """
        preferences = self.config.instance_preferences or [self.config.instance_type]
        poll_interval = self.config.poll_interval
        max_retries = self.config.max_launch_retries
        launch_failures = 0

        logger.info(
            "Snatching instance for %s (preferences: %s, poll every %ds)",
            self.config.model_id, ", ".join(preferences), poll_interval,
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                result = self._find_available(preferences)
                if result is None:
                    if attempt % 6 == 1:  # Log every ~minute
                        logger.info("No GPU available yet (attempt %d)...", attempt)
                    time.sleep(poll_interval)
                    continue

                instance_type, region = result
                logger.info("Found %s in %s, launching...", instance_type, region)

                resp = httpx.post(
                    f"{self.BASE_URL}/instance-operations/launch",
                    auth=self._auth(),
                    json={
                        "region_name": region,
                        "instance_type_name": instance_type,
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
                logger.info("Instance launched: %s (type=%s, region=%s)",
                            instance_id, instance_type, region)

                ip = self._poll_for_ip(instance_id)
                self.instance = LambdaInstance(
                    instance_id=instance_id, ip=ip, status="active"
                )
                self._install_safety_nets()
                return self.instance

            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                if e.response.status_code == 401:
                    raise RuntimeError("Invalid LAMBDA_API_KEY") from e
                if "insufficient-capacity" in error_body:
                    launch_failures += 1
                    logger.warning(
                        "Capacity gone before launch (race %d/%d), continuing to poll...",
                        launch_failures, max_retries,
                    )
                    if launch_failures >= max_retries:
                        raise RuntimeError(
                            f"Failed to launch after {max_retries} capacity races"
                        ) from e
                    time.sleep(poll_interval)
                    continue
                raise
            except KeyboardInterrupt:
                logger.info("Polling cancelled by user")
                raise

    def _find_available(self, preferences: list[str]) -> tuple[str, str] | None:
        """Check Lambda API for GPU availability across preferred types.

        Returns (instance_type, region) for the first match, or None.
        """
        resp = httpx.get(
            f"{self.BASE_URL}/instance-types",
            auth=self._auth(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        for target in preferences:
            info = data.get(target)
            if not info:
                continue
            regions = info.get("regions_with_capacity_available", [])
            if regions:
                region = regions[0].get("name")
                return target, region

        return None

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
