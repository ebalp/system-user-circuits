"""Tests for lambda_cloud.ssh — SSHConnection."""

import subprocess
from unittest.mock import patch, MagicMock, call

import pytest

from lambda_cloud.ssh import SSHConnection, _SSH_OPTS


@pytest.fixture
def ssh():
    return SSHConnection(ip="10.0.0.1", key_file="~/.ssh/test.pem")


class TestSSHConnection:
    def test_target(self, ssh):
        assert ssh._target() == "ubuntu@10.0.0.1"

    def test_ssh_base(self, ssh):
        base = ssh._ssh_base()
        assert base[0] == "ssh"
        assert "-i" in base
        assert "~/.ssh/test.pem" in base
        for opt in _SSH_OPTS:
            assert opt in base


class TestRun:
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_run_calls_subprocess(self, mock_run, ssh):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="hello", stderr=""
        )
        result = ssh.run("echo hello")
        assert result.stdout == "hello"

        args = mock_run.call_args[0][0]
        assert args[-1] == "echo hello"
        assert "ubuntu@10.0.0.1" in args

    @patch("lambda_cloud.ssh.subprocess.run")
    def test_run_check_true_raises(self, mock_run, ssh):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        with pytest.raises(subprocess.CalledProcessError):
            ssh.run("false", check=True)

    @patch("lambda_cloud.ssh.subprocess.run")
    def test_run_check_false_no_raise(self, mock_run, ssh):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="err"
        )
        result = ssh.run("false", check=False)
        assert result.returncode == 1

    @patch("lambda_cloud.ssh.subprocess.run")
    def test_run_passes_timeout(self, mock_run, ssh):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        ssh.run("cmd", timeout=42)
        assert mock_run.call_args[1]["timeout"] == 42


class TestRunBackground:
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_run_background_uses_nohup(self, mock_run, ssh):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        ssh.run_background("my-command")
        args = mock_run.call_args[0][0]
        cmd_str = args[-1]
        assert "nohup" in cmd_str
        assert "disown" in cmd_str
        assert "my-command" in cmd_str


class TestOpenTunnel:
    @patch("lambda_cloud.ssh.subprocess.Popen")
    def test_open_tunnel_creates_popen(self, mock_popen, ssh):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        result = ssh.open_tunnel(local_port=9000, remote_port=8000)
        assert result is mock_proc

        args = mock_popen.call_args[0][0]
        assert "-N" in args
        assert "-L" in args
        assert "9000:localhost:8000" in args
        assert "ubuntu@10.0.0.1" in args


class TestWaitForSSH:
    @patch("lambda_cloud.ssh.time.sleep")
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_returns_true_when_reachable(self, mock_run, mock_sleep, ssh):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        assert ssh.wait_for_ssh(timeout=30, interval=1) is True

    @patch("lambda_cloud.ssh.time.sleep")
    @patch("lambda_cloud.ssh.time.time")
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_returns_false_on_timeout(self, mock_run, mock_time, mock_sleep, ssh):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr="refused"
        )
        # Provide enough time.time() calls (logging also calls it internally)
        mock_time.side_effect = lambda: mock_time._clock
        mock_time._clock = 0
        call_count = [0]
        _orig_side_effect = mock_time.side_effect

        def _time():
            call_count[0] += 1
            # First 2 calls: initial time, then while-check (within timeout)
            # After that: past deadline
            if call_count[0] <= 2:
                return 0
            return 100

        mock_time.side_effect = _time
        assert ssh.wait_for_ssh(timeout=10, interval=1) is False

    @patch("lambda_cloud.ssh.time.sleep")
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_retries_on_timeout_expired(self, mock_run, mock_sleep, ssh):
        mock_run.side_effect = [
            subprocess.TimeoutExpired("ssh", 15),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""),
        ]
        assert ssh.wait_for_ssh(timeout=300, interval=1) is True


class TestUploadFile:
    @patch("lambda_cloud.ssh.subprocess.run")
    def test_upload_uses_scp(self, mock_run, ssh):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        ssh.upload_file("/local/file.txt", "/remote/file.txt")

        args = mock_run.call_args[0][0]
        assert args[0] == "scp"
        assert "/local/file.txt" in args
        assert "ubuntu@10.0.0.1:/remote/file.txt" in args

    @patch("lambda_cloud.ssh.subprocess.run")
    def test_upload_raises_on_failure(self, mock_run, ssh):
        mock_run.side_effect = subprocess.CalledProcessError(1, "scp")
        with pytest.raises(subprocess.CalledProcessError):
            ssh.upload_file("/local/file.txt", "/remote/file.txt")
