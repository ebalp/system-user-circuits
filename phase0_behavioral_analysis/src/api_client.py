"""
Hugging Face Inference API client for Phase 0 Behavioral Analysis.
"""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class EmptyResponseError(Exception):
    """Raised when the API returns None or empty content. Retryable."""
    pass


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    content: str
    model: str
    timestamp: str
    prompt_hash: str
    usage: dict[str, int] | None
    error: str | None


class HFClient:
    """
    Hugging Face Inference API client with retry logic and logging.
    
    Token priority: api_key param > token_file > HF_API_KEY env var
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        token_file: str = 'hf_token.txt',
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize the HF client.
        
        Args:
            api_key: API key (highest priority)
            token_file: Path to file containing API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for rate limits
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.token = self._load_token(api_key, token_file)
        self._client: InferenceClient | None = None
    
    def _load_token(self, api_key: str | None, token_file: str) -> str:
        """Load API token from various sources."""
        # Priority 1: Direct api_key parameter
        if api_key:
            logger.debug("Using API key from parameter")
            return api_key
        
        # Priority 2: Token file
        token_path = Path(token_file)
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                logger.debug(f"Using API key from {token_file}")
                return token
        
        # Priority 3: Environment variable
        env_token = os.environ.get('HF_API_KEY')
        if env_token:
            logger.debug("Using API key from HF_API_KEY environment variable")
            return env_token
        
        raise ValueError(
            "No API token found. Provide api_key, create hf_token.txt, "
            "or set HF_API_KEY environment variable."
        )
    
    def _get_client(self, model_id: str) -> InferenceClient:
        """Get or create an InferenceClient for the given model."""
        return InferenceClient(model=model_id, token=self.token, timeout=self.timeout)
    
    @staticmethod
    def _strip_think_blocks(content: str) -> str:
        """Remove <think>...</think> blocks from Qwen3 responses."""
        return re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
    
    @staticmethod
    def _compute_prompt_hash(system_message: str, user_message: str) -> str:
        """Compute a hash of the prompt for deduplication."""
        combined = f"{system_message}|||{user_message}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def chat_completion(
        self,
        model_id: str,
        system_message: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        disable_thinking: bool = True
    ) -> ChatResponse:
        """
        Send a chat completion request.
        
        Args:
            model_id: HuggingFace model ID
            system_message: System prompt content
            user_message: User message content
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            disable_thinking: If True, disable Qwen3's reasoning mode (default: True)
            
        Returns:
            ChatResponse with content, metadata, and any errors
        """
        prompt_hash = self._compute_prompt_hash(system_message, user_message)
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        # For Qwen3 models, append /no_think to disable reasoning mode
        final_user_message = user_message
        if disable_thinking and 'qwen3' in model_id.lower():
            final_user_message = f"{user_message} /no_think"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": final_user_message}
        ]
        
        logger.info(
            f"API call: model={model_id}, hash={prompt_hash}, "
            f"system_len={len(system_message)}, user_len={len(user_message)}"
        )
        
        try:
            response = self._retry_with_backoff(
                lambda: self._make_request_and_validate(
                    model_id, messages, temperature, max_tokens,
                    disable_thinking
                )
            )
            
            content, usage = response
            
            logger.info(f"API success: hash={prompt_hash}, response_len={len(content)}")
            
            return ChatResponse(
                content=content,
                model=model_id,
                timestamp=timestamp,
                prompt_hash=prompt_hash,
                usage=usage,
                error=None
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API error: hash={prompt_hash}, error={error_msg}")
            
            return ChatResponse(
                content="",
                model=model_id,
                timestamp=timestamp,
                prompt_hash=prompt_hash,
                usage=None,
                error=error_msg
            )
    
    def _make_request(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Any:
        """Make the actual API request."""
        client = self._get_client(model_id)
        return client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _make_request_and_validate(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        disable_thinking: bool
    ) -> tuple[str, dict[str, int] | None]:
        """Make API request and validate the response content.
        
        Raises EmptyResponseError if the model returns None or empty string,
        so the retry logic can handle it.
        
        Returns:
            Tuple of (content, usage_dict_or_None)
        """
        response = self._make_request(model_id, messages, temperature, max_tokens)
        
        content = response.choices[0].message.content
        
        # Treat None or empty content as a retryable error
        if content is None or (isinstance(content, str) and content.strip() == ""):
            raise EmptyResponseError(
                f"Model returned empty/None response for {model_id}"
            )
        
        # Strip <think>...</think> blocks for Qwen3 when thinking is disabled
        if disable_thinking and 'qwen3' in model_id.lower():
            content = self._strip_think_blocks(content)
        
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        return content, usage
    
    def _retry_with_backoff(
        self,
        func,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0
    ) -> Any:
        """
        Retry a function with exponential backoff for rate limits and empty responses.
        
        Args:
            func: Function to retry
            initial_delay: Initial delay in seconds
            backoff_factor: Multiplier for each retry
            
        Returns:
            Result of the function
            
        Raises:
            Last exception if all retries fail
        """
        delay = initial_delay
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_rate_limit = (
                    '429' in error_str or
                    'rate limit' in error_str or
                    'too many requests' in error_str
                )
                is_empty_response = isinstance(e, EmptyResponseError)
                is_server_error = (
                    '500' in error_str or
                    '502' in error_str or
                    '503' in error_str or
                    'service unavailable' in error_str
                )
                
                is_retryable = is_rate_limit or is_empty_response or is_server_error
                
                if is_retryable and attempt < self.max_retries - 1:
                    reason = "rate limit" if is_rate_limit else (
                        "empty response" if is_empty_response else "server error"
                    )
                    logger.warning(
                        f"{reason.capitalize()} hit, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    raise
        
        raise last_exception
