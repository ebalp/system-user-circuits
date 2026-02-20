"""
Tests for the HuggingFace API client.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.api_client import HFClient, ChatResponse


def _has_api_token() -> bool:
    """Check if an API token is available for integration tests."""
    return bool(os.environ.get('HF_TOKEN'))


class TestTokenLoading:
    """Tests for token loading from parameter or HF_TOKEN env var."""

    def test_token_from_parameter(self):
        """Token passed as parameter should be used."""
        client = HFClient(api_key="test_token_param")
        assert client.token == "test_token_param"

    def test_token_from_env_var(self):
        """Token should be loaded from HF_TOKEN env var when no parameter given."""
        with patch.dict(os.environ, {'HF_TOKEN': 'test_token_env'}):
            client = HFClient()
            assert client.token == "test_token_env"

    def test_token_priority_param_over_env(self):
        """Parameter should take priority over env var."""
        with patch.dict(os.environ, {'HF_TOKEN': 'env_token'}):
            client = HFClient(api_key="param_token")
            assert client.token == "param_token"

    def test_no_token_raises_error(self):
        """Should raise ValueError if no token found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No HF token found"):
                HFClient()


class TestPromptHash:
    """Tests for prompt hashing."""
    
    def test_hash_consistency(self):
        """Same inputs should produce same hash."""
        hash1 = HFClient._compute_prompt_hash("system", "user")
        hash2 = HFClient._compute_prompt_hash("system", "user")
        assert hash1 == hash2
    
    def test_hash_different_inputs(self):
        """Different inputs should produce different hashes."""
        hash1 = HFClient._compute_prompt_hash("system1", "user")
        hash2 = HFClient._compute_prompt_hash("system2", "user")
        assert hash1 != hash2
    
    def test_hash_length(self):
        """Hash should be 16 characters."""
        hash1 = HFClient._compute_prompt_hash("system", "user")
        assert len(hash1) == 16


class TestRetryLogic:
    """Tests for retry with backoff."""
    
    def test_no_retry_on_success(self):
        """Should not retry if function succeeds."""
        client = HFClient(api_key="test")
        call_count = 0
        
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = client._retry_with_backoff(success_func)
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_rate_limit(self):
        """Should retry on rate limit errors."""
        client = HFClient(api_key="test", max_retries=3)
        call_count = 0
        
        def rate_limit_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success"
        
        with patch('time.sleep'):  # Don't actually sleep in tests
            result = client._retry_with_backoff(rate_limit_then_success)
        
        assert result == "success"
        assert call_count == 3
    
    def test_no_retry_on_other_errors(self):
        """Should not retry on non-rate-limit errors."""
        client = HFClient(api_key="test", max_retries=3)
        call_count = 0
        
        def other_error():
            nonlocal call_count
            call_count += 1
            raise Exception("Some other error")
        
        with pytest.raises(Exception, match="Some other error"):
            client._retry_with_backoff(other_error)
        
        assert call_count == 1
    
    def test_max_retries_exceeded(self):
        """Should raise after max retries."""
        client = HFClient(api_key="test", max_retries=2)
        call_count = 0
        
        def always_rate_limit():
            nonlocal call_count
            call_count += 1
            raise Exception("429 rate limit")
        
        with patch('time.sleep'):
            with pytest.raises(Exception, match="429"):
                client._retry_with_backoff(always_rate_limit)
        
        assert call_count == 2


class TestRetryLogicWithMockedRateLimits:
    """Unit tests for retry logic with mocked rate limit responses."""
    
    def test_exponential_backoff_delays(self):
        """Should use exponential backoff delays between retries."""
        client = HFClient(api_key="test", max_retries=4)
        call_count = 0
        sleep_calls = []
        
        def rate_limit_error():
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")
        
        with patch('time.sleep', side_effect=lambda x: sleep_calls.append(x)):
            with pytest.raises(Exception, match="429"):
                client._retry_with_backoff(
                    rate_limit_error,
                    initial_delay=1.0,
                    backoff_factor=2.0
                )
        
        # Should have 3 sleep calls (retries - 1, since last attempt doesn't sleep)
        assert len(sleep_calls) == 3
        # Verify exponential backoff: 1.0, 2.0, 4.0
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0
    
    def test_retry_on_rate_limit_text_variations(self):
        """Should recognize various rate limit error message formats."""
        rate_limit_messages = [
            "429 Too Many Requests",
            "Rate limit exceeded",
            "too many requests - please slow down",
            "Error 429: rate limit",
            "RATE LIMIT reached",  # Case insensitive
        ]
        
        for error_msg in rate_limit_messages:
            client = HFClient(api_key="test", max_retries=2)
            call_count = 0
            
            def make_error_func(msg):
                def error_func():
                    nonlocal call_count
                    call_count += 1
                    if call_count < 2:
                        raise Exception(msg)
                    return "success"
                return error_func
            
            with patch('time.sleep'):
                result = client._retry_with_backoff(make_error_func(error_msg))
            
            assert result == "success", f"Failed for error message: {error_msg}"
            assert call_count == 2, f"Expected 2 calls for: {error_msg}"
    
    def test_mocked_api_rate_limit_response(self):
        """Test retry behavior with mocked HF API rate limit response."""
        client = HFClient(api_key="test", max_retries=3)
        call_count = 0
        
        # Create mock response for successful call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success after retry"
        mock_response.usage = None
        
        def mock_make_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate HF API rate limit error
                raise Exception("HfHubHTTPError: 429 Client Error: Too Many Requests")
            return mock_response
        
        with patch.object(client, '_make_request', side_effect=mock_make_request):
            with patch('time.sleep'):
                result = client.chat_completion(
                    model_id="test-model",
                    system_message="System",
                    user_message="User"
                )
        
        assert result.content == "Success after retry"
        assert result.error is None
        assert call_count == 3
    
    def test_rate_limit_exhausts_retries_returns_error(self):
        """When rate limits exhaust all retries, should return error in ChatResponse."""
        client = HFClient(api_key="test", max_retries=2)
        
        def always_rate_limit(*args, **kwargs):
            raise Exception("429 Too Many Requests")
        
        with patch.object(client, '_make_request', side_effect=always_rate_limit):
            with patch('time.sleep'):
                result = client.chat_completion(
                    model_id="test-model",
                    system_message="System",
                    user_message="User"
                )
        
        assert result.content == ""
        assert result.error is not None
        assert "429" in result.error
    
    def test_custom_backoff_parameters(self):
        """Should respect custom initial_delay and backoff_factor."""
        client = HFClient(api_key="test", max_retries=3)
        sleep_calls = []
        call_count = 0
        
        def rate_limit_error():
            nonlocal call_count
            call_count += 1
            raise Exception("rate limit")
        
        with patch('time.sleep', side_effect=lambda x: sleep_calls.append(x)):
            with pytest.raises(Exception):
                client._retry_with_backoff(
                    rate_limit_error,
                    initial_delay=0.5,
                    backoff_factor=3.0
                )
        
        # Verify custom backoff: 0.5, 1.5
        assert sleep_calls[0] == 0.5
        assert sleep_calls[1] == 1.5


class TestChatCompletion:
    """Tests for chat completion method."""
    
    def test_chat_completion_success(self):
        """Should return ChatResponse on success."""
        client = HFClient(api_key="test")
        
        # Mock the _make_request method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        with patch.object(client, '_make_request', return_value=mock_response):
            result = client.chat_completion(
                model_id="test-model",
                system_message="You are helpful",
                user_message="Hello"
            )
        
        assert isinstance(result, ChatResponse)
        assert result.content == "Test response"
        assert result.model == "test-model"
        assert result.error is None
        assert result.usage['total_tokens'] == 15
    
    def test_chat_completion_error(self):
        """Should return ChatResponse with error on failure."""
        client = HFClient(api_key="test")
        
        with patch.object(client, '_make_request', side_effect=Exception("API Error")):
            result = client.chat_completion(
                model_id="test-model",
                system_message="You are helpful",
                user_message="Hello"
            )
        
        assert isinstance(result, ChatResponse)
        assert result.content == ""
        assert result.error == "API Error"


class TestIntegration:
    """Integration tests requiring live API access."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not _has_api_token(), reason="No API token available")
    def test_live_api_call_simple_prompt(self):
        """
        Integration test: Make a live API call to one model with a simple prompt.
        
        This test verifies:
        - API connectivity works
        - Chat completion returns a valid response
        - Response structure is correct
        """
        # Use a small, fast model for testing
        model_id = "openai/gpt-oss-20b"
        
        client = HFClient()
        
        response = client.chat_completion(
            model_id=model_id,
            system_message="You are a helpful assistant. Be brief.",
            user_message="What is 2 + 2? Reply with just the number.",
            temperature=0.0,
            max_tokens=50
        )
        
        # Verify response structure
        assert isinstance(response, ChatResponse)
        assert response.model == model_id
        assert response.timestamp is not None
        assert response.prompt_hash is not None
        assert len(response.prompt_hash) == 16
        
        # Verify successful response (no error)
        assert response.error is None, f"API call failed: {response.error}"
        
        # Verify we got actual content
        assert response.content is not None
        assert len(response.content) > 0
        
        # The response should contain "4" somewhere (basic sanity check)
        assert "4" in response.content, f"Expected '4' in response: {response.content}"
