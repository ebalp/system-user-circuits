"""vLLM server lifecycle management via SSH.

Installs vLLM in an isolated virtualenv (avoids system TensorFlow conflicts),
starts the server in the background, and uses SSH-proxied health checks
(avoids Lambda's port 8000 firewall).
"""

import logging
import time

from lambda_cloud.ssh import SSHConnection

logger = logging.getLogger(__name__)


def install_vllm(ssh: SSHConnection, venv_path: str = "/home/ubuntu/vllm-venv") -> None:
    """Create a virtualenv and install vLLM on the remote instance.

    Args:
        ssh: Active SSH connection to the instance.
        venv_path: Path for the Python virtualenv on the remote.
    """
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
    ssh.run_background(f"bash -c '{cmd} > /var/log/vllm-server.log 2>&1'")
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


def stop_vllm(ssh: SSHConnection) -> None:
    """Stop the vLLM server process on the remote instance."""
    logger.info("Stopping vLLM on %s", ssh.ip)
    ssh.run("pkill -f 'vllm serve' || true", timeout=15, check=False)
