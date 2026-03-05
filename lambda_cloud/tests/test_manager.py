"""Tests for LambdaCloudManager.

All tests are mocked — no real API calls, no GPU instances launched.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest

from lambda_cloud.config import LambdaConfig, LambdaInstance
from lambda_cloud.manager import LambdaCloudManager


# ── Fixtures ──


@pytest.fixture
def lambda_config():
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
    return LambdaCloudManager(lambda_config)


def _mock_launch_response(instance_id="i-abc123"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"instance_ids": [instance_id]}}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_instance_response(instance_id="i-abc123", ip="10.0.0.1", status="active"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"id": instance_id, "ip": ip, "status": status}}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_terminate_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"terminated_instances": [{"id": "i-abc123"}]}}
    resp.raise_for_status = MagicMock()
    return resp


# ── Launch ──


class TestLaunch:
    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_successful_launch(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response("i-123")
        mock_get.return_value = _mock_instance_response("i-123", "10.0.0.5")

        instance = manager.launch()
        assert instance.instance_id == "i-123"
        assert instance.ip == "10.0.0.5"
        assert manager.instance is not None

    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_sends_correct_payload(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["region_name"] == "us-east-1"
        assert call_json["instance_type_name"] == "gpu_1x_a10"
        assert call_json["ssh_key_names"] == ["test-key"]
        assert "user_data" not in call_json  # No cloud-init anymore

    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_retries_on_failure(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, mock_sleep, manager):
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

    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.httpx.post")
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

    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_installs_atexit(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        mock_atexit.assert_called_once()

    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_installs_signal_handlers(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        assert mock_getsignal.call_count == 2
        assert mock_signal.call_count == 2

    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_sends_basic_auth(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        mock_post.return_value = _mock_launch_response()
        mock_get.return_value = _mock_instance_response()

        manager.launch()
        call_auth = mock_post.call_args[1]["auth"]
        assert isinstance(call_auth, httpx.BasicAuth)
        assert call_auth._auth_header == httpx.BasicAuth("test-api-key", "")._auth_header

    @patch("lambda_cloud.manager.signal.signal")
    @patch("lambda_cloud.manager.signal.getsignal")
    @patch("lambda_cloud.manager.atexit.register")
    @patch("lambda_cloud.manager.httpx.get")
    @patch("lambda_cloud.manager.httpx.post")
    def test_launch_empty_instance_ids_raises(self, mock_post, mock_get, mock_atexit, mock_getsignal, mock_signal, manager):
        resp = MagicMock()
        resp.json.return_value = {"data": {"instance_ids": []}}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="Failed to launch"):
            manager.launch()


# ── Poll for IP ──


class TestPollForIp:
    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.httpx.get")
    def test_returns_ip_immediately(self, mock_get, mock_sleep, manager):
        mock_get.return_value = _mock_instance_response(ip="10.0.0.1")
        ip = manager._poll_for_ip("i-123", timeout=30, interval=1)
        assert ip == "10.0.0.1"

    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.time.time")
    @patch("lambda_cloud.manager.httpx.get")
    def test_polls_until_ip_appears(self, mock_get, mock_time, mock_sleep, manager):
        no_ip_resp = MagicMock()
        no_ip_resp.json.return_value = {"data": {"ip": None, "status": "booting"}}
        no_ip_resp.raise_for_status = MagicMock()
        ip_resp = _mock_instance_response(ip="10.0.0.2")
        mock_get.side_effect = [no_ip_resp, ip_resp]
        mock_time.side_effect = [0, 0, 5]

        ip = manager._poll_for_ip("i-123", timeout=300, interval=1)
        assert ip == "10.0.0.2"
        assert mock_get.call_count == 2

    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.time.time")
    @patch("lambda_cloud.manager.httpx.get")
    def test_timeout_raises(self, mock_get, mock_time, mock_sleep, manager):
        no_ip_resp = MagicMock()
        no_ip_resp.json.return_value = {"data": {"ip": None, "status": "booting"}}
        no_ip_resp.raise_for_status = MagicMock()
        mock_get.return_value = no_ip_resp
        mock_time.side_effect = [0, 0, 100]

        with pytest.raises(RuntimeError, match="did not get an IP"):
            manager._poll_for_ip("i-123", timeout=30, interval=1)

    @patch("lambda_cloud.manager.time.sleep")
    @patch("lambda_cloud.manager.httpx.get")
    def test_handles_poll_errors_gracefully(self, mock_get, mock_sleep, manager):
        ip_resp = _mock_instance_response(ip="10.0.0.3")
        mock_get.side_effect = [Exception("network error"), ip_resp]

        ip = manager._poll_for_ip("i-123", timeout=300, interval=1)
        assert ip == "10.0.0.3"


# ── Terminate ──


class TestTerminate:
    @patch("lambda_cloud.manager.httpx.post")
    def test_terminate_sends_correct_payload(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        call_json = mock_post.call_args[1]["json"]
        assert call_json["instance_ids"] == ["i-abc"]

    @patch("lambda_cloud.manager.httpx.post")
    def test_terminate_idempotent(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        manager.terminate()
        assert mock_post.call_count == 1

    def test_terminate_no_instance_noop(self, manager):
        manager.terminate()

    @patch("lambda_cloud.manager.httpx.post")
    def test_terminate_api_error_logged_not_raised(self, mock_post, manager):
        mock_post.side_effect = Exception("network error")
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        assert not manager._terminated

    @patch("lambda_cloud.manager.httpx.post")
    def test_terminate_sets_terminated_flag(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        assert not manager._terminated
        manager.terminate()
        assert manager._terminated

    @patch("lambda_cloud.manager.httpx.post")
    def test_terminate_sends_basic_auth(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager.terminate()
        call_auth = mock_post.call_args[1]["auth"]
        assert isinstance(call_auth, httpx.BasicAuth)
        assert call_auth._auth_header == httpx.BasicAuth("test-api-key", "")._auth_header

    def test_terminate_already_terminated_noop(self, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        manager._terminated = True
        manager.terminate()


# ── Context manager ──


class TestContextManager:
    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "launch")
    def test_enter_calls_launch(self, mock_launch, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with mgr:
            mock_launch.assert_called_once()
        mock_terminate.assert_called_once()

    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "launch")
    def test_exit_terminates_on_exception(self, mock_launch, mock_terminate):
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
    @patch.object(LambdaCloudManager, "launch")
    def test_exit_does_not_suppress_exceptions(self, mock_launch, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with pytest.raises(RuntimeError, match="boom"):
            with mgr:
                raise RuntimeError("boom")

    @patch.object(LambdaCloudManager, "terminate")
    @patch.object(LambdaCloudManager, "launch")
    def test_enter_returns_self(self, mock_launch, mock_terminate):
        config = LambdaConfig(
            api_key="k", ssh_key_name="s", model_id="m",
            instance_type="t", region="r", hf_token="t",
        )
        mgr = LambdaCloudManager(config)
        with mgr as ctx:
            assert ctx is mgr


# ── get_base_url ──


class TestGetBaseUrl:
    def test_no_instance_raises(self, manager):
        with pytest.raises(RuntimeError, match="No instance launched"):
            manager.get_base_url()

    def test_returns_correct_url(self, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        assert manager.get_base_url() == "http://10.0.0.1:8000/v1"

    def test_custom_port(self, manager):
        manager.config.vllm_port = 9000
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        assert manager.get_base_url() == "http://10.0.0.1:9000/v1"


# ── list_available ──


class TestListAvailable:
    @patch("lambda_cloud.manager.httpx.get")
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

    @patch("lambda_cloud.manager.httpx.get")
    def test_empty_data(self, mock_get, manager):
        resp = MagicMock()
        resp.json.return_value = {"data": {}}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        assert manager.list_available() == []

    @patch("lambda_cloud.manager.httpx.get")
    def test_sends_basic_auth(self, mock_get, manager):
        resp = MagicMock()
        resp.json.return_value = {"data": {}}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        manager.list_available()
        call_auth = mock_get.call_args[1]["auth"]
        assert isinstance(call_auth, httpx.BasicAuth)


# ── Safety nets ──


class TestSafetyNets:
    @patch("lambda_cloud.manager.httpx.post")
    def test_atexit_cleanup_calls_terminate(self, mock_post, manager):
        mock_post.return_value = _mock_terminate_response()
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")

        manager._atexit_cleanup()
        assert manager._terminated

    def test_atexit_cleanup_noop_when_terminated(self, manager):
        manager.instance = LambdaInstance("i-abc", "10.0.0.1", "active")
        manager._terminated = True
        manager._atexit_cleanup()

    def test_auth_uses_basic_auth(self, manager):
        auth = manager._auth()
        assert isinstance(auth, httpx.BasicAuth)
        assert auth._auth_header == httpx.BasicAuth("test-api-key", "")._auth_header
