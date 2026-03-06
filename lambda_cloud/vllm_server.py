"""vLLM server lifecycle management via SSH.

Installs vLLM in an isolated virtualenv (avoids system TensorFlow conflicts),
starts the server in the background, and uses SSH-proxied health checks
(avoids Lambda's port 8000 firewall).
"""

import logging
import time

import httpx

from lambda_cloud.ssh import SSHConnection

logger = logging.getLogger(__name__)


def install_vllm(ssh: SSHConnection, venv_path: str = "/home/ubuntu/vllm-venv") -> None:
    """Create a virtualenv and install vLLM on the remote instance.

    Skips if vLLM is already installed in the venv.

    Args:
        ssh: Active SSH connection to the instance.
        venv_path: Path for the Python virtualenv on the remote.
    """
    # Check if vLLM is already installed
    check = ssh.run(f"test -x {venv_path}/bin/vllm && echo installed || echo missing",
                    timeout=10, check=False)
    if "installed" in check.stdout:
        logger.info("vLLM already installed in %s on %s, skipping", venv_path, ssh.ip)
        return

    logger.info("Installing vLLM in %s on %s", venv_path, ssh.ip)
    ssh.run(f"python3 -m venv {venv_path}", timeout=60)
    ssh.run(f"{venv_path}/bin/pip install --upgrade pip", timeout=120)
    ssh.run(f"{venv_path}/bin/pip install vllm", timeout=600)
    logger.info("vLLM installed successfully on %s", ssh.ip)


def start_vllm(
    ssh: SSHConnection,
    model_id: str,
    hf_token: str,
    port: int = 8000,
    extra_args: str = "",
    venv_path: str = "/home/ubuntu/vllm-venv",
) -> None:
    """Start vLLM server in the background on the remote instance.

    Args:
        ssh: Active SSH connection.
        model_id: HuggingFace model ID to serve.
        hf_token: HuggingFace API token.
        port: Port for vLLM to listen on.
        extra_args: Extra CLI args for vllm serve (e.g. --max-model-len 4096).
        venv_path: Path to virtualenv with vLLM installed.
    """
    logger.info("Starting vLLM on %s (model=%s, port=%d)", ssh.ip, model_id, port)
    cmd = (
        f"HF_TOKEN={hf_token} {venv_path}/bin/vllm serve {model_id} "
        f"--host 0.0.0.0 --port {port} {extra_args}"
    )
    # Redirect logs so we can debug remotely
    ssh.run_background(f"bash -c '{cmd} > /home/ubuntu/vllm-server.log 2>&1'")
    logger.info("vLLM started in background on %s", ssh.ip)


def wait_for_vllm_ready(
    ssh: SSHConnection,
    port: int = 8000,
    timeout: int = 900,
    interval: int = 15,
) -> bool:
    """Wait for vLLM to be ready by checking /v1/models via SSH-proxied curl.

    This avoids the Lambda firewall issue — we curl localhost from inside
    the instance rather than connecting from outside.

    Args:
        ssh: Active SSH connection.
        port: vLLM port on the remote instance.
        timeout: Max seconds to wait.
        interval: Seconds between health checks.

    Returns:
        True if vLLM became ready, False if timed out.
    """
    logger.info("Waiting for vLLM on %s:%d (timeout=%ds)", ssh.ip, port, timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = ssh.run(
                f"curl -sf http://localhost:{port}/v1/models",
                timeout=15, check=False,
            )
            if result.returncode == 0 and "data" in result.stdout:
                logger.info("vLLM ready on %s:%d", ssh.ip, port)
                return True
        except Exception as e:
            logger.debug("vLLM health check failed: %s", e)
        time.sleep(interval)
    logger.warning("vLLM not ready on %s:%d after %ds", ssh.ip, port, timeout)
    return False


def vllm_status(ssh: SSHConnection, port: int = 8000) -> dict | None:
    """Check if vLLM is running and what model is served.

    Returns:
        Dict with 'pid', 'model', and 'cmdline' if running, None otherwise.
    """
    # Check for running process (character class trick avoids pgrep matching itself)
    result = ssh.run("pgrep -fa '[v]llm serve' || true", timeout=10, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None

    lines = result.stdout.strip().splitlines()
    # Parse first matching line: "PID /path/to/vllm serve model-id ..."
    parts = lines[0].split()
    pid = parts[0]
    cmdline = " ".join(parts[1:])

    # Try to get the served model from /v1/models endpoint
    model = None
    health = ssh.run(
        f"curl -sf http://localhost:{port}/v1/models", timeout=10, check=False,
    )
    if health.returncode == 0 and health.stdout.strip():
        try:
            import json
            data = json.loads(health.stdout)
            models = data.get("data", [])
            if models:
                model = models[0].get("id")
        except (json.JSONDecodeError, IndexError):
            pass

    return {"pid": pid, "model": model, "cmdline": cmdline}


def wait_for_vllm_through_tunnel(
    base_url: str, expected_model: str, timeout: int = 60
) -> None:
    """Wait until vLLM is reachable through the local SSH tunnel.

    Polls ``/v1/models`` via *httpx* and validates the served model.

    Args:
        base_url: Local tunnel URL, e.g. ``http://localhost:8000/v1``.
        expected_model: Model ID that vLLM should be serving.
        timeout: Max seconds to wait.

    Raises:
        RuntimeError: On model mismatch or if vLLM is unreachable after *timeout*.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                served_model = models[0]["id"] if models else None
                logger.info(
                    "vLLM reachable through tunnel at %s (model: %s)",
                    base_url, served_model,
                )
                if served_model and served_model != expected_model:
                    raise RuntimeError(
                        f"Model mismatch: expected '{expected_model}' but vLLM is serving "
                        f"'{served_model}'. Wrong tunnel or stale vLLM process?"
                    )
                return
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"vLLM not reachable through tunnel at {base_url} after {timeout}s"
    )


def ensure_vllm_running(
    ssh: SSHConnection,
    model_id: str,
    hf_token: str,
    port: int = 8000,
    extra_args: str = "",
    venv_path: str = "/home/ubuntu/vllm-venv",
    readiness_timeout: int = 900,
) -> None:
    """Ensure vLLM is running on the remote instance.

    Checks ``vllm_status()`` first and skips setup if already running.
    Otherwise runs install → start → wait.

    Raises:
        RuntimeError: If vLLM doesn't become ready within *readiness_timeout*.
    """
    status = vllm_status(ssh, port=port)
    if status:
        logger.info(
            "vLLM already running on %s (model=%s, pid=%s)",
            ssh.ip, status["model"], status["pid"],
        )
        return

    logger.info("vLLM not running on %s, starting...", ssh.ip)
    install_vllm(ssh, venv_path=venv_path)
    start_vllm(
        ssh, model_id=model_id, hf_token=hf_token,
        port=port, extra_args=extra_args, venv_path=venv_path,
    )
    if not wait_for_vllm_ready(ssh, port=port, timeout=readiness_timeout):
        raise RuntimeError(
            f"vLLM not ready on {ssh.ip} after {readiness_timeout}s"
        )


def stop_vllm(ssh: SSHConnection) -> None:
    """Stop the vLLM server process on the remote instance."""
    logger.info("Stopping vLLM on %s", ssh.ip)
    ssh.run("pkill -f 'vllm serve' || true", timeout=15, check=False)
