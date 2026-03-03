"""Tests for VLLMClient and refactored retry/strip helpers in api_client.py."""

import time
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from phase0_v2.src.api_client import (
    VLLMClient,
    ChatResponse,
    EmptyResponseError,
    _retry_with_backoff,
    _strip_think_blocks,
)


# ── Module-level helper tests ──


class TestRetryWithBackoff:
    """Test the extracted module-level _retry_with_backoff function."""

    def test_returns_on_first_success(self):
        result = _retry_with_backoff(lambda: 42, max_retries=3)
        assert result == 42

    def test_retries_on_429_then_succeeds(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429 Too Many Requests")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert call_count == 2

    def test_retries_on_empty_response_error(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise EmptyResponseError("empty")
            return "recovered"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "recovered"
        assert call_count == 2

    def test_retries_on_502(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("502 Bad Gateway")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert call_count == 3

    def test_retries_on_503(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("503 Service Unavailable")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"

    def test_raises_non_retryable_immediately(self):
        """Non-retryable errors (e.g. 400, ValueError) should raise immediately."""
        def func():
            raise ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            _retry_with_backoff(func, max_retries=3)

    def test_raises_after_max_retries_exhausted(self):
        def func():
            raise Exception("429 rate limited")
        with patch("phase0_v2.src.api_client.time.sleep"):
            with pytest.raises(Exception, match="429"):
                _retry_with_backoff(func, max_retries=2)

    def test_backoff_factor_doubles_delay(self):
        """Verify sleep delays increase with backoff_factor."""
        sleep_calls = []
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise Exception("429")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            _retry_with_backoff(func, max_retries=5, initial_delay=1.0, backoff_factor=2.0)
        assert sleep_calls == [1.0, 2.0, 4.0]

    def test_retries_on_rate_limit_string(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"

    def test_retries_on_service_unavailable_string(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("service unavailable")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"

    def test_max_retries_one_no_retry(self):
        """With max_retries=1, retryable errors still raise immediately (no retry)."""
        def func():
            raise Exception("429")
        with pytest.raises(Exception, match="429"):
            _retry_with_backoff(func, max_retries=1)

    def test_retries_on_500(self):
        """500 Internal Server Error should be retryable."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("500 Internal Server Error")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"
        assert call_count == 2

    def test_retries_on_too_many_requests(self):
        """'too many requests' string should be retryable."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("too many requests")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep"):
            result = _retry_with_backoff(func, max_retries=3)
        assert result == "ok"

    def test_custom_initial_delay(self):
        """Custom initial_delay should be used for the first sleep."""
        sleep_calls = []
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("429")
            return "ok"
        with patch("phase0_v2.src.api_client.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            _retry_with_backoff(func, max_retries=3, initial_delay=5.0)
        assert sleep_calls == [5.0]

    def test_non_retryable_404_raises_immediately(self):
        """A 404 error should not be retried."""
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            raise Exception("404 Not Found")
        with pytest.raises(Exception, match="404"):
            _retry_with_backoff(func, max_retries=3)
        assert call_count == 1


class TestStripThinkBlocks:
    """Test the extracted module-level _strip_think_blocks function."""

    def test_removes_think_block(self):
        text = "<think>internal reasoning</think>Hello world"
        assert _strip_think_blocks(text) == "Hello world"

    def test_removes_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\nResult here"
        assert _strip_think_blocks(text) == "Result here"

    def test_no_think_block_unchanged(self):
        text = "Hello world"
        assert _strip_think_blocks(text) == "Hello world"

    def test_empty_think_block(self):
        text = "<think></think>Content"
        assert _strip_think_blocks(text) == "Content"

    def test_multiple_think_blocks(self):
        text = "<think>a</think>Hello <think>b</think>world"
        assert _strip_think_blocks(text) == "Hello world"

    def test_only_think_block_returns_empty(self):
        text = "<think>only thinking</think>"
        assert _strip_think_blocks(text) == ""

    def test_whitespace_after_think_block_stripped(self):
        text = "<think>x</think>   \n  Result"
        assert _strip_think_blocks(text) == "Result"

    def test_empty_string_unchanged(self):
        assert _strip_think_blocks("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _strip_think_blocks("   ") == ""

    def test_nested_angle_brackets_in_think(self):
        """Content with < > inside think block should still be stripped."""
        text = "<think>if x < 5 and y > 3</think>Answer"
        assert _strip_think_blocks(text) == "Answer"


# ── VLLMClient tests ──


def _make_mock_response(content="Hello", prompt_tokens=10, completion_tokens=5):
    """Create a mock OpenAI ChatCompletion response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


class TestVLLMClientInit:
    """Test VLLMClient initialization."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_default_api_key(self, mock_openai_cls):
        client = VLLMClient(base_url="http://localhost:8000/v1")
        assert client.api_key == "EMPTY"
        assert client.base_url == "http://localhost:8000/v1"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_custom_params(self, mock_openai_cls):
        client = VLLMClient(
            base_url="http://10.0.0.1:9000/v1",
            api_key="mykey",
            timeout=120,
            max_retries=5,
        )
        assert client.api_key == "mykey"
        assert client.timeout == 120
        assert client.max_retries == 5

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_creates_openai_client(self, mock_openai_cls):
        VLLMClient(base_url="http://localhost:8000/v1")
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            timeout=60,
        )

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_default_timeout_and_retries(self, mock_openai_cls):
        client = VLLMClient(base_url="http://localhost:8000/v1")
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_raises_import_error_when_openai_missing(self):
        """VLLMClient should raise ImportError if openai is not installed."""
        with patch("phase0_v2.src.api_client.OpenAI", None):
            with pytest.raises(ImportError, match="openai package is required"):
                VLLMClient(base_url="http://localhost:8000/v1")


class TestVLLMClientChatCompletion:
    """Test VLLMClient.chat_completion method."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_basic_completion(self, mock_openai_cls):
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response("Hi there")

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, ChatResponse)
        assert result.content == "Hi there"
        assert result.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert result.error is None
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_passes_params_to_openai(self, mock_openai_cls):
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response()

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}]
        client.chat_completion(
            model_id="test-model", messages=msgs, temperature=0.7, max_tokens=256,
        )
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=msgs,
            temperature=0.7,
            max_tokens=256,
        )

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_qwen3_no_think_appended(self, mock_openai_cls):
        """Qwen3 models should get /no_think appended to last user message."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response("result")

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat_completion(model_id="Qwen/Qwen3-8B", messages=msgs)

        called_msgs = mock_client_instance.chat.completions.create.call_args[1]["messages"]
        assert called_msgs[-1]["content"] == "Hello /no_think"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_qwen3_think_blocks_stripped(self, mock_openai_cls):
        """Qwen3 think blocks should be stripped from response."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response(
            "<think>reasoning</think>Final answer"
        )

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion(model_id="Qwen/Qwen3-8B",
                                        messages=[{"role": "user", "content": "Hi"}])
        assert result.content == "Final answer"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_non_qwen_no_think_not_appended(self, mock_openai_cls):
        """Non-Qwen3 models should NOT get /no_think."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response()

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat_completion(model_id="meta-llama/Llama-3.1-8B-Instruct", messages=msgs)

        called_msgs = mock_client_instance.chat.completions.create.call_args[1]["messages"]
        assert called_msgs[-1]["content"] == "Hello"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_disable_thinking_false_skips_no_think(self, mock_openai_cls):
        """disable_thinking=False should not append /no_think even for Qwen3."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response()

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat_completion(model_id="Qwen/Qwen3-8B", messages=msgs, disable_thinking=False)

        called_msgs = mock_client_instance.chat.completions.create.call_args[1]["messages"]
        assert called_msgs[-1]["content"] == "Hello"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_empty_response_returns_error(self, mock_openai_cls):
        """Empty API response should return ChatResponse with error."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response(content="")

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.error is not None
        assert "empty" in result.error.lower() or "None" in result.error

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_none_response_returns_error(self, mock_openai_cls):
        """None content should return ChatResponse with error."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response(content=None)

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.error is not None

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_api_exception_returns_error(self, mock_openai_cls):
        """API exception should return ChatResponse with error, not raise."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = Exception("Connection refused")

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(result, ChatResponse)
        assert result.content == ""
        assert result.error is not None
        assert "Connection refused" in result.error

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_no_usage_in_response(self, mock_openai_cls):
        """Handle responses without usage info (usage=None)."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        resp = _make_mock_response("Hello")
        resp.usage = None
        mock_client_instance.chat.completions.create.return_value = resp

        client = VLLMClient(base_url="http://localhost:8000/v1")
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == "Hello"
        assert result.usage is None

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_original_messages_not_mutated(self, mock_openai_cls):
        """chat_completion should not mutate the caller's messages list."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response()

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "user", "content": "Hello"}]
        original_content = msgs[0]["content"]
        client.chat_completion(model_id="Qwen/Qwen3-8B", messages=msgs)
        assert msgs[0]["content"] == original_content

    @patch("phase0_v2.src.api_client.time.sleep")
    @patch("phase0_v2.src.api_client.OpenAI")
    def test_retries_on_server_error(self, mock_openai_cls, mock_sleep):
        """VLLMClient should retry on 500/502/503 errors."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.side_effect = [
            Exception("502 Bad Gateway"),
            _make_mock_response("recovered"),
        ]

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=3)
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.content == "recovered"
        assert result.error is None

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_qwen3_case_insensitive(self, mock_openai_cls):
        """Qwen3 detection should be case-insensitive."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response("result")

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [{"role": "user", "content": "Hello"}]
        client.chat_completion(model_id="QWEN/QWEN3-8B", messages=msgs)

        called_msgs = mock_client_instance.chat.completions.create.call_args[1]["messages"]
        assert called_msgs[-1]["content"] == "Hello /no_think"

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_qwen3_multiple_user_messages_only_last_modified(self, mock_openai_cls):
        """Only the last user message should get /no_think appended."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response()

        client = VLLMClient(base_url="http://localhost:8000/v1")
        msgs = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        client.chat_completion(model_id="Qwen/Qwen3-8B", messages=msgs)

        called_msgs = mock_client_instance.chat.completions.create.call_args[1]["messages"]
        assert called_msgs[0]["content"] == "First"  # Not modified
        assert called_msgs[2]["content"] == "Second /no_think"  # Last user msg

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_whitespace_only_response_returns_error(self, mock_openai_cls):
        """Whitespace-only response should be treated as empty."""
        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = _make_mock_response(content="   \n  ")

        client = VLLMClient(base_url="http://localhost:8000/v1", max_retries=1)
        result = client.chat_completion(
            model_id="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.error is not None


class TestVLLMClientWaitUntilReady:
    """Test VLLMClient.wait_until_ready method."""

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_ready_immediately(self, mock_openai_cls):
        client = VLLMClient(base_url="http://localhost:8000/v1")
        with patch("httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"id": "test-model"}]}
            mock_get.return_value = mock_resp
            assert client.wait_until_ready(timeout_seconds=5) is True

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_ready_after_polls(self, mock_openai_cls):
        client = VLLMClient(base_url="http://localhost:8000/v1")
        with patch("httpx.get") as mock_get, \
             patch("phase0_v2.src.api_client.time.sleep"):
            fail_resp = MagicMock()
            fail_resp.status_code = 503
            ok_resp = MagicMock()
            ok_resp.status_code = 200
            ok_resp.json.return_value = {"data": [{"id": "model"}]}
            mock_get.side_effect = [Exception("conn refused"), fail_resp, ok_resp]
            assert client.wait_until_ready(timeout_seconds=600, poll_interval=1) is True

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_timeout_returns_false(self, mock_openai_cls):
        client = VLLMClient(base_url="http://localhost:8000/v1")
        with patch("httpx.get", side_effect=Exception("down")), \
             patch("phase0_v2.src.api_client.time.sleep"), \
             patch("phase0_v2.src.api_client.time.time") as mock_time:
            # Simulate: first call returns 0 (start), second returns 0 (check),
            # third returns 100 (exceeds timeout)
            mock_time.side_effect = [0, 0, 100]
            assert client.wait_until_ready(timeout_seconds=5) is False

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_empty_data_keeps_polling(self, mock_openai_cls):
        """Server responds 200 but no models loaded yet -> keep polling."""
        client = VLLMClient(base_url="http://localhost:8000/v1")
        with patch("httpx.get") as mock_get, \
             patch("phase0_v2.src.api_client.time.sleep"), \
             patch("phase0_v2.src.api_client.time.time") as mock_time:
            empty_resp = MagicMock()
            empty_resp.status_code = 200
            empty_resp.json.return_value = {"data": []}
            ok_resp = MagicMock()
            ok_resp.status_code = 200
            ok_resp.json.return_value = {"data": [{"id": "model"}]}
            mock_get.side_effect = [empty_resp, ok_resp]
            mock_time.side_effect = [0, 0, 0, 0]
            assert client.wait_until_ready(timeout_seconds=600, poll_interval=1) is True

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_url_construction(self, mock_openai_cls):
        """Verify /models URL is built correctly from base_url."""
        client = VLLMClient(base_url="http://10.0.0.1:8000/v1")
        with patch("httpx.get") as mock_get, \
             patch("phase0_v2.src.api_client.time.sleep"), \
             patch("phase0_v2.src.api_client.time.time") as mock_time:
            # deadline = 0 + 5 = 5; loop check: 1 < 5 enters; next check: 100 exits
            mock_time.side_effect = [0, 1, 100]
            mock_get.side_effect = Exception("down")
            client.wait_until_ready(timeout_seconds=5)
            mock_get.assert_called_with("http://10.0.0.1:8000/v1/models", timeout=10)

    @patch("phase0_v2.src.api_client.OpenAI")
    def test_url_construction_trailing_slash(self, mock_openai_cls):
        """Trailing slash on base_url should not create double slash."""
        client = VLLMClient(base_url="http://10.0.0.1:8000/v1/")
        with patch("httpx.get") as mock_get, \
             patch("phase0_v2.src.api_client.time.sleep"), \
             patch("phase0_v2.src.api_client.time.time") as mock_time:
            # deadline = 0 + 5 = 5; loop check: 1 < 5 enters; next check: 100 exits
            mock_time.side_effect = [0, 1, 100]
            mock_get.side_effect = Exception("down")
            client.wait_until_ready(timeout_seconds=5)
            mock_get.assert_called_with("http://10.0.0.1:8000/v1/models", timeout=10)


# ── HFClient still works after refactoring ──


class TestHFClientRefactorIntact:
    """Verify HFClient still works after extracting module-level functions."""

    def test_hfclient_requires_token(self):
        """HFClient should still require a token."""
        import os
        env_backup = os.environ.get("HF_TOKEN")
        try:
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(ValueError, match="No HF token"):
                from phase0_v2.src.api_client import HFClient
                HFClient()
        finally:
            if env_backup is not None:
                os.environ["HF_TOKEN"] = env_backup

    def test_hfclient_accepts_api_key(self):
        """HFClient should accept api_key param."""
        from phase0_v2.src.api_client import HFClient
        client = HFClient(api_key="test-token")
        assert client.token == "test-token"

    def test_hfclient_default_params(self):
        """HFClient should have correct defaults."""
        from phase0_v2.src.api_client import HFClient
        client = HFClient(api_key="test-token")
        assert client.timeout == 60
        assert client.max_retries == 3

    def test_hfclient_custom_params(self):
        """HFClient should accept custom timeout and max_retries."""
        from phase0_v2.src.api_client import HFClient
        client = HFClient(api_key="test-token", timeout=120, max_retries=5)
        assert client.timeout == 120
        assert client.max_retries == 5
