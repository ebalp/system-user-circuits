"""Live tests against a running vLLM server on a Lambda Cloud instance.

These tests require:
  1. A Lambda instance with vLLM running (port 8000)
  2. SSH access configured as Host "lambda" in ~/.ssh/config

Run with:
    uv run pytest lambda_cloud/tests/test_live.py -v -m live

Skipped automatically if SSH to the instance fails.
"""

import os
import subprocess

import pytest

from lambda_cloud.ssh import SSHConnection

# ── Fixtures ──

LAMBDA_SSH_HOST = "lambda"  # from ~/.ssh/config
VLLM_REMOTE_PORT = 8000
TUNNEL_LOCAL_PORT = 18000  # use non-standard port to avoid collisions
_VLLM_MODEL_ID_ENV = os.environ.get("VLLM_MODEL_ID", "")


def _read_lambda_ip() -> str:
    """Read the HostName for 'lambda' from ~/.ssh/config."""
    try:
        result = subprocess.run(
            ["ssh", "-G", LAMBDA_SSH_HOST],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("hostname "):
                return line.split()[1]
    except Exception:
        pass
    return ""


def _vllm_reachable_via_ssh(ip: str) -> bool:
    """Quick check: can we SSH in and curl vLLM?"""
    try:
        result = subprocess.run(
            ["ssh", LAMBDA_SSH_HOST, f"curl -sf http://localhost:{VLLM_REMOTE_PORT}/v1/models"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "data" in result.stdout
    except Exception:
        return False


def _detect_served_model() -> str:
    """Ask the remote vLLM what model it's serving via SSH."""
    try:
        result = subprocess.run(
            ["ssh", LAMBDA_SSH_HOST, f"curl -sf http://localhost:{VLLM_REMOTE_PORT}/v1/models"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            models = data.get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return ""


_lambda_ip = _read_lambda_ip()
_vllm_ok = _vllm_reachable_via_ssh(_lambda_ip) if _lambda_ip else False
MODEL_ID = _VLLM_MODEL_ID_ENV or (_detect_served_model() if _vllm_ok else "") or "meta-llama/Llama-3.1-8B-Instruct"

live = pytest.mark.live
skip_no_vllm = pytest.mark.skipif(
    not _vllm_ok,
    reason="No live vLLM server reachable via 'ssh lambda'",
)


@pytest.fixture(scope="module")
def tunnel():
    """Open an SSH tunnel to the Lambda instance's vLLM port for the test module."""
    ssh = SSHConnection(ip=_lambda_ip)
    proc, local_port = ssh.open_tunnel(local_port=TUNNEL_LOCAL_PORT, remote_port=VLLM_REMOTE_PORT)
    # Give the tunnel a moment to establish
    import time
    time.sleep(1)
    yield f"http://localhost:{local_port}/v1"
    proc.terminate()
    proc.wait()


# ── SSH tunnel itself ──


@live
@skip_no_vllm
class TestSSHTunnel:
    """Verify the SSH tunnel fixture works correctly."""

    def test_tunnel_serves_models_endpoint(self, tunnel):
        """The tunnel should proxy /v1/models from the remote vLLM."""
        import httpx
        resp = httpx.get(f"{tunnel}/models", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == MODEL_ID


# ── vLLM connectivity (no phase0_v2 dependency) ──


@live
@skip_no_vllm
class TestVLLMConnectivity:
    """Basic vLLM chat completion via openai SDK through SSH tunnel."""

    def test_chat_completion_returns_content(self, tunnel):
        from openai import OpenAI
        client = OpenAI(base_url=tunnel, api_key="unused")
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            temperature=0.0,
            max_tokens=64,
        )
        assert resp.choices[0].message.content
        assert len(resp.choices[0].message.content) > 0

    def test_system_message_works(self, tunnel):
        from openai import OpenAI
        client = OpenAI(base_url=tunnel, api_key="unused")
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
                {"role": "user", "content": "What is water?"},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        assert resp.choices[0].message.content
        assert len(resp.choices[0].message.content) > 0

    def test_returns_usage(self, tunnel):
        from openai import OpenAI
        client = OpenAI(base_url=tunnel, api_key="unused")
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0,
            max_tokens=64,
        )
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
