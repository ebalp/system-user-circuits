"""Tests for lambda_cloud.vllm_server — vLLM install/start/healthcheck/stop."""

import subprocess
from unittest.mock import patch, MagicMock, call

import pytest

from lambda_cloud.ssh import SSHConnection
from lambda_cloud.vllm_server import install_vllm, start_vllm, wait_for_vllm_ready, stop_vllm


@pytest.fixture
def ssh():
    """Create a mock SSHConnection."""
    mock = MagicMock(spec=SSHConnection)
    mock.ip = "10.0.0.1"
    return mock


class TestInstallVllm:
    def test_creates_venv_and_installs(self, ssh):
        install_vllm(ssh, venv_path="/home/ubuntu/venv")

        calls = ssh.run.call_args_list
        assert len(calls) == 3
        # Creates venv
        assert "/home/ubuntu/venv" in calls[0][0][0]
        assert "venv" in calls[0][0][0]
        # Upgrades pip
        assert "pip install --upgrade pip" in calls[1][0][0]
        # Installs vllm
        assert "pip install vllm" in calls[2][0][0]

    def test_uses_default_venv_path(self, ssh):
        install_vllm(ssh)
        assert "/home/ubuntu/vllm-venv" in ssh.run.call_args_list[0][0][0]


class TestStartVllm:
    def test_starts_in_background(self, ssh):
        start_vllm(
            ssh, model_id="meta-llama/Llama-3.1-8B-Instruct",
            hf_token="hf-tok", port=8000, extra_args="--max-model-len 4096",
        )
        ssh.run_background.assert_called_once()
        cmd = ssh.run_background.call_args[0][0]
        assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
        assert "--port 8000" in cmd
        assert "--max-model-len 4096" in cmd
        assert "HF_TOKEN=hf-tok" in cmd

    def test_uses_venv_binary(self, ssh):
        start_vllm(
            ssh, model_id="m", hf_token="t",
            venv_path="/opt/vllm-env",
        )
        cmd = ssh.run_background.call_args[0][0]
        assert "/opt/vllm-env/bin/vllm" in cmd


class TestWaitForVllmReady:
    @patch("lambda_cloud.vllm_server.time.sleep")
    def test_returns_true_when_ready(self, mock_sleep, ssh):
        result = MagicMock()
        result.returncode = 0
        result.stdout = '{"data": [{"id": "model"}]}'
        ssh.run.return_value = result

        assert wait_for_vllm_ready(ssh, port=8000, timeout=30) is True

    @patch("lambda_cloud.vllm_server.time.sleep")
    @patch("lambda_cloud.vllm_server.time.time")
    def test_returns_false_on_timeout(self, mock_time, mock_sleep, ssh):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        ssh.run.return_value = result

        # Provide enough time.time() calls (logging also calls it internally)
        call_count = [0]
        def _time():
            call_count[0] += 1
            if call_count[0] <= 2:
                return 0
            return 100
        mock_time.side_effect = _time

        assert wait_for_vllm_ready(ssh, port=8000, timeout=10) is False

    @patch("lambda_cloud.vllm_server.time.sleep")
    def test_health_check_uses_curl(self, mock_sleep, ssh):
        result = MagicMock()
        result.returncode = 0
        result.stdout = '{"data": [{"id": "model"}]}'
        ssh.run.return_value = result

        wait_for_vllm_ready(ssh, port=9000, timeout=30)
        cmd = ssh.run.call_args[0][0]
        assert "curl" in cmd
        assert "localhost:9000" in cmd
        assert "/v1/models" in cmd

    @patch("lambda_cloud.vllm_server.time.sleep")
    def test_retries_on_ssh_exception(self, mock_sleep, ssh):
        result_ok = MagicMock()
        result_ok.returncode = 0
        result_ok.stdout = '{"data": []}'
        ssh.run.side_effect = [Exception("conn refused"), result_ok]

        assert wait_for_vllm_ready(ssh, port=8000, timeout=300) is True


class TestStopVllm:
    def test_kills_vllm_process(self, ssh):
        stop_vllm(ssh)
        ssh.run.assert_called_once()
        cmd = ssh.run.call_args[0][0]
        assert "pkill" in cmd
        assert "vllm serve" in cmd
