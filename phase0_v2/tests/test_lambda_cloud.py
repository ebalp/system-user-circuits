"""Tests for LambdaCloudManager and load_lambda_config.

All tests are mocked — no real API calls, no GPU instances launched.
"""

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from phase0_v2.src.lambda_cloud import (
    LambdaConfig,
    LambdaInstance,
    LambdaCloudManager,
    load_lambda_config,
)


# ── Fixtures ──


@pytest.fixture
def lambda_config():
    """Create a minimal LambdaConfig for tests."""
    return LambdaConfig(
        api_key="test-api-key",
        ssh_key_name="test-key",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        instance_type="gpu_1x_a10",
        region="us-east-1",
        hf_token="hf-test-token",
        vllm_port=8000,
        vllm_extra_args="--max-model-len 4096",
        max_launch_retries=2,
        launch_retry_delay=1,
        readiness_timeout=10,
    )


@pytest.fixture
def manager(lambda_config):
    """Create a LambdaCloudManager (not launched)."""
    return LambdaCloudManager(lambda_config)


def _mock_launch_response(instance_id="i-abc123"):
    """Create a mock successful launch response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {"instance_ids": [instance_id]}
    }
    resp.raise_for_status = MagicMock()
    return resp


def _mock_instance_response(instance_id="i-abc123", ip="10.0.0.1", status="active"):
    """Create a mock GET /instances/{id} response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {"id": instance_id, "ip": ip, "status": status}
    }
    resp.raise_for_status = MagicMock()
    return resp


def _mock_terminate_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"terminated_instances": [{"id": "i-abc123"}]}}
    resp.raise_for_status = MagicMock()
    return resp


# ── LambdaConfig tests ──


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

    def test_custom_values(self, lambda_config):
        assert lambda_config.api_key == "test-api-key"
        assert lambda_config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert lambda_config.vllm_extra_args == "--max-model-len 4096"


# ── Cloud-init generation ──


class TestBuildCloudInit:
    def test_contains_model_id(self, manager):
        script = manager._build_cloud_init()
        assert "meta-llama/Llama-3.1-8B-Instruct" in script

    def test_contains_hf_token(self, manager):
        script = manager._build_cloud_init()
        assert "hf-test-token" in script

    def test_contains_vllm_serve(self, manager):
        script = manager._build_cloud_init()
        assert "vllm serve" in script

    def test_contains_port(self, manager):
        script = manager._build_cloud_init()
        assert "--port 8000" in script

    def test_contains_extra_args(self, manager):
        script = manager._build_cloud_init()
        assert "--max-model-len 4096" in script

    def test_contains_pip_install_vllm(self, manager):
        script = manager._build_cloud_init()
        assert "pip install vllm" in script

    def test_starts_with_shebang(self, manager):
        script = manager._build_cloud_init()
        assert script.startswith("#!/bin/bash")

    def test_uses_nohup(self, manager):
        script = manager._build_cloud_init()
        assert "nohup" in script

    def test_logs_to_file(self, manager):
        script = manager._build_cloud_init()
        assert "/var/log/vllm-setup.log" in script


# ── Launch ──


class TestLaunch:
    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_successful_launch(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response("i-123")
        mock_get.return_value = _mock_instance_response("i-123", "10.0.0.5")

        instance = manager.launch()
        assert instance.instance_id == "i-123"
        assert instance.ip == "10.0.0.5"
        assert manager.instance is not None

    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_sends_correct_payload(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["region_name"] == "us-east-1"
        assert call_json["instance_type_name"] == "gpu_1x_a10"
        assert call_json["ssh_key_names"] == ["test-key"]
        assert "user_data" in call_json

    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_retries_on_failure(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, mock_sleep, manager):
        """Should retry on HTTP errors up to max_launch_retries."""
        import httpx as real_httpx
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        mock_post.side_effect = [fail_resp, _mock_launch_response()]
        mock_get.return_value = _mock_instance_response()

        instance = manager.launch()
        assert instance.instance_id == "i-abc123"
        assert mock_post.call_count == 2

    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_exhausts_retries(self, mock_post, mock_sleep, manager):
        import httpx as real_httpx
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = fail_resp

        with pytest.raises(RuntimeError, match="Failed to launch"):
            manager.launch()
        assert mock_post.call_count == manager.config.max_launch_retries

    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_installs_atexit(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        mock_atexit.assert_called_once()

    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_installs_signal_handlers(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        """launch() should install SIGINT and SIGTERM handlers."""
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        # getsignal called for SIGINT and SIGTERM
        assert mock_getsignal.call_count == 2
        # signal.signal called for SIGINT and SIGTERM
        assert mock_signal.call_count == 2

    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_sends_auth_header(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        """launch() should send Bearer token in headers."""
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers == {"Authorization": "Bearer test-api-key"}

    @patch("phase0_v2.src.lambda_cloud.signal.signal")
    @patch("phase0_v2.src.lambda_cloud.signal.getsignal")
    @patch("phase0_v2.src.lambda_cloud.atexit.register")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_launch_empty_instance_ids_raises(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        """launch() should raise RuntimeError if API returns empty instance_ids."""
        resp = MagicMock()
        resp.json.return_value = {"data": {"instance_ids": []}}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="Failed to launch"):
            manager.launch()


# ── Poll for IP ──


class TestPollForIp:
    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_returns_ip_immediately(self, mock_get, mock_sleep, manager):
        mock_get.return_value = _mock_instance_response(ip="10.0.0.1")
        ip = manager._poll_for_ip("i-123", timeout=30, interval=1)
        assert ip == "10.0.0.1"

    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.time.time")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_polls_until_ip_appears(self, mock_get, mock_time, mock_sleep, manager):
        """Should poll multiple times if IP is not immediately available."""
        no_ip_resp = MagicMock()
        no_ip_resp.json.return_value = {"data": {"ip": None, "status": "booting"}}
        no_ip_resp.raise_for_status = MagicMock()
        ip_resp = _mock_instance_response(ip="10.0.0.2")
        mock_get.side_effect = [no_ip_resp, ip_resp]
        # time.time() calls: deadline calc, first check, second check
        mock_time.side_effect = [0, 0, 5]

        ip = manager._poll_for_ip("i-123", timeout=300, interval=1)
        assert ip == "10.0.0.2"
        assert mock_get.call_count == 2

    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.time.time")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_timeout_raises(self, mock_get, mock_time, mock_sleep, manager):
        """Should raise RuntimeError if IP never appears within timeout."""
        no_ip_resp = MagicMock()
        no_ip_resp.json.return_value = {"data": {"ip": None, "status": "booting"}}
        no_ip_resp.raise_for_status = MagicMock()
        mock_get.return_value = no_ip_resp
        # deadline = 0 + 30 = 30; first check: 0 < 30 enters; second check: 100 exits
        mock_time.side_effect = [0, 0, 100]

        with pytest.raises(RuntimeError, match="did not get an IP"):
            manager._poll_for_ip("i-123", timeout=30, interval=1)

    @patch("phase0_v2.src.lambda_cloud.time.sleep")
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_handles_poll_errors_gracefully(self, mock_get, mock_sleep, manager):
        """Should continue polling even if individual requests fail."""
        ip_resp = _mock_instance_response(ip="10.0.0.3")
        mock_get.side_effect = [Exception("network error"), ip_resp]

        ip = manager._poll_for_ip("i-123", timeout=300, interval=1)
        assert ip == "10.0.0.3"


# ── Terminate ──


class TestTerminate:
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_terminate_sends_correct_payload(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["instance_ids"] == ["i-abc"]

    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_terminate_idempotent(self, mock_post, manager):
        """Calling terminate twice should only send one API request."""
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        manager.terminate()
        assert mock_post.call_count == 1

    def test_terminate_no_instance_noop(self, manager):
        """terminate() with no instance should not raise."""
        manager.terminate()  # Should not raise

    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_terminate_api_error_logged_not_raised(self, mock_post, manager):
        """API errors during terminate should be logged, not raised."""
        mock_post.side_effect = Exception("network error")
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()  # Should not raise
        assert not manager._terminated  # Failed to terminate

    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_terminate_sets_terminated_flag(self, mock_post, manager):
        """Successful terminate should set _terminated = True."""
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        assert not manager._terminated
        manager.terminate()
        assert manager._terminated

    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_terminate_sends_auth_header(self, mock_post, manager):
        """terminate() should send Bearer token in headers."""
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers == {"Authorization": "Bearer test-api-key"}

    def test_terminate_already_terminated_noop(self, manager):
        """terminate() when _terminated=True should not send API request."""
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        manager._terminated = True
        # No mock needed — if it tries to call httpx.post, it will fail
        manager.terminate()


# ── Context manager ──


class TestContextManager:
    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "wait_for_ready")
    @patch.object(LambdaCloudManager, "launch")
    def test_enter_calls_launch_and_wait(self, mock_launch, mock_wait, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with mgr:
            mock_launch.assert_called_once()
            mock_wait.assert_called_once()
        mock_terminate.assert_called_once()

    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "wait_for_ready")
    @patch.object(LambdaCloudManager, "launch")
    def test_exit_terminates_on_exception(self, mock_launch, mock_wait, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with pytest.raises(ValueError):
            with mgr:
                raise ValueError("test error")
        mock_terminate.assert_called_once()

    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "wait_for_ready")
    @patch.object(LambdaCloudManager, "launch")
    def test_exit_does_not_suppress_exceptions(self, mock_launch, mock_wait, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with pytest.raises(RuntimeError, match="boom"):
            with mgr:
                raise RuntimeError("boom")

    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "wait_for_ready")
    @patch.object(LambdaCloudManager, "launch")
    def test_enter_returns_self(self, mock_launch, mock_wait, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with mgr as ctx:
            assert ctx is mgr


# ── get_client ──


class TestGetClient:
    def test_get_client_no_instance_raises(self, manager):
        with pytest.raises(RuntimeError, match="No instance launched"):
            manager.get_client()

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_get_client_returns_vllm_client(self, mock_openai_cls, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        client = manager.get_client()
        from phase0_v2.src.api_client import VLLMClient
        assert isinstance(client, VLLMClient)
        assert client.base_url == "http://10.0.0.1:8000/v1"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_get_client_custom_port(self, mock_openai_cls, manager):
        manager.config.vllm_port = 9000
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        client = manager.get_client()
        assert client.base_url == "http://10.0.0.1:9000/v1"


# ── wait_for_ready ──


class TestWaitForReady:
    def test_no_instance_raises(self, manager):
        with pytest.raises(RuntimeError, match="No instance launched"):
            manager.wait_for_ready()

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_calls_wait_until_ready(self, mock_openai_cls, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        with patch("httpx.get") as mock_get, \
             patch("phase0_v2.src.api_client.time.sleep"):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "test-model"}]}
            mock_get.return_value = mock_resp

            manager.wait_for_ready()

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_raises_on_timeout(self, mock_openai_cls, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        with patch("httpx.get", side_effect=Exception("down")), \
             patch("phase0_v2.src.api_client.time.sleep"), \
             patch("phase0_v2.src.api_client.time.time") as mock_time:
            # deadline = 0 + 10 = 10; first check: 0 < 10 enters; next check: 100 exits
            mock_time.side_effect = [0, 0, 100]
            with pytest.raises(RuntimeError, match="not ready"):
                manager.wait_for_ready()


# ── list_available ──


class TestListAvailable:
    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_returns_parsed_types(self, mock_get, manager):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {
                "gpu_1x_a10": {
                    "instance_type": {
                        "description": "1x A10",
                        "price_cents_per_hour": 75,
                        "specs": {"gpus": 1},
                    },
                    "regions_with_capacity_available": [
                        {"name": "us-east-1"}
                    ],
                }
            }
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = manager.list_available()
        assert len(result) == 1
        assert result[0]["name"] == "gpu_1x_a10"
        assert result[0]["gpu_count"] == 1
        assert result[0]["available_regions"] == ["us-east-1"]
        assert result[0]["price_cents_per_hour"] == 75
        assert result[0]["description"] == "1x A10"

    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_empty_data(self, mock_get, manager):
        resp = MagicMock()
        resp.json.return_value = {"data": {}}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = manager.list_available()
        assert result == []

    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_multiple_instance_types(self, mock_get, manager):
        resp = MagicMock()
        resp.json.return_value = {
            "data": {
                "gpu_1x_a10": {
                    "instance_type": {
                        "description": "1x A10",
                        "price_cents_per_hour": 75,
                        "specs": {"gpus": 1},
                    },
                    "regions_with_capacity_available": [],
                },
                "gpu_8x_a100": {
                    "instance_type": {
                        "description": "8x A100",
                        "price_cents_per_hour": 880,
                        "specs": {"gpus": 8},
                    },
                    "regions_with_capacity_available": [
                        {"name": "us-east-1"}, {"name": "us-west-2"}
                    ],
                },
            }
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = manager.list_available()
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"gpu_1x_a10", "gpu_8x_a100"}

    @patch("phase0_v2.src.lambda_cloud.httpx.get")
    def test_sends_auth_header(self, mock_get, manager):
        resp = MagicMock()
        resp.json.return_value = {"data": {}}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        manager.list_available()
        call_headers = mock_get.call_args[1]["headers"]
        assert call_headers == {"Authorization": "Bearer test-api-key"}


# ── Safety nets ──


class TestSafetyNets:
    @patch("phase0_v2.src.lambda_cloud.httpx.post")
    def test_atexit_cleanup_calls_terminate(self, mock_post, manager):
        """_atexit_cleanup should call terminate if not already terminated."""
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager._atexit_cleanup()
        assert manager._terminated

    def test_atexit_cleanup_noop_when_terminated(self, manager):
        """_atexit_cleanup should not call terminate if already terminated."""
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        manager._terminated = True

        # Should not raise or make API calls
        manager._atexit_cleanup()

    def test_headers_contain_bearer_token(self, manager):
        headers = manager._headers()
        assert headers == {"Authorization": "Bearer test-api-key"}


# ── LambdaInstance ──


class TestLambdaInstance:
    def test_fields(self):
        inst = LambdaInstance(instance_id="i-123", ip="10.0.0.1", status="active")
        assert inst.instance_id == "i-123"
        assert inst.ip == "10.0.0.1"
        assert inst.status == "active"


# ── load_lambda_config ──


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
            # Make sure LAMBDA_API_KEY is not set
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
        yaml_path = Path("phase0_v2/config/lambda.yaml")
        if yaml_path.exists():
            with patch.dict(os.environ, {"LAMBDA_API_KEY": "test", "HF_TOKEN": "test"}):
                config = load_lambda_config(yaml_path, "meta-llama/Llama-3.1-8B-Instruct")
            assert config.instance_type == "gpu_1x_a10"
            assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"

    def test_concurrent_per_model_in_defaults(self, tmp_path):
        """concurrent_per_model should be readable from defaults even though
        it's not part of LambdaConfig — it's used by the runner."""
        config_yaml = tmp_path / "lambda.yaml"
        config_yaml.write_text("""
ssh_key_name: "k"
defaults:
  concurrent_per_model: 10
model_gpu_map:
  _default:
    instance_type: gpu_1x_a100
""")
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "k", "HF_TOKEN": "t"}):
            # load_lambda_config doesn't extract concurrent_per_model into
            # LambdaConfig, but the YAML should parse fine
            config = load_lambda_config(config_yaml, "m")
            assert config.instance_type == "gpu_1x_a100"

    def test_defaults_applied_when_missing_from_yaml(self, tmp_path):
        """When defaults section is empty, hardcoded fallbacks should apply."""
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
        """When model is not in gpu_map and no _default, use hardcoded fallback."""
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
        """load_lambda_config should accept both str and Path."""
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
        """The model_id passed to load_lambda_config should be preserved in config."""
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
        """readiness_timeout from YAML defaults should override hardcoded default."""
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
