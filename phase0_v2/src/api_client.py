"""API clients for Phase 0 v2 (HF Inference API + vLLM OpenAI-compatible)."""

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from huggingface_hub import InferenceClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Optional dependency — only needed for VLLMClient

logger = logging.getLogger(__name__)


class EmptyResponseError(Exception):
    """Raised when the API returns None or empty content. Retryable."""
    pass


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    content: str
    model: str
    usage: dict[str, int] | None
    error: str | None


# ── Module-level helpers ──


def _retry_with_backoff(
    func, max_retries: int, initial_delay: float = 1.0, backoff_factor: float = 2.0
):
    """Retry a callable with exponential backoff on retryable errors.

    Retryable: 429, 500, 502, 503, rate limit errors, EmptyResponseError.
    """
    delay = initial_delay
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            is_retryable = (
                "429" in error_str
                or "rate limit" in error_str
                or "too many requests" in error_str
                or isinstance(e, EmptyResponseError)
                or "500" in error_str
                or "502" in error_str
                or "503" in error_str
                or "service unavailable" in error_str
            )
            if is_retryable and attempt < max_retries - 1:
                logger.warning(
                    "Retryable error, retrying in %.1fs (attempt %d/%d): %s",
                    delay, attempt + 1, max_retries, e,
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise
    raise last_exception


def _strip_think_blocks(content: str) -> str:
    """Remove <think>...</think> blocks from model output (e.g. Qwen3)."""
    return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()


# ── HFClient ──


class HFClient:
    """HF Inference API client with retry logic.

    Token priority: api_key param > HF_TOKEN env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.token = api_key or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "No HF token found. Set HF_TOKEN env var or pass api_key."
            )

    def _get_client(self, model_id: str) -> InferenceClient:
        return InferenceClient(model=model_id, token=self.token, timeout=self.timeout)

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        disable_thinking: bool = True,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            model_id: HuggingFace model ID
            messages: List of {"role": ..., "content": ...} dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            disable_thinking: If True, append /no_think for Qwen3 models

        Returns:
            ChatResponse with content and metadata
        """
        # For Qwen3: append /no_think to the last user message
        if disable_thinking and "qwen3" in model_id.lower():
            messages = [m.copy() for m in messages]
            for m in reversed(messages):
                if m["role"] == "user":
                    m["content"] = f"{m['content']} /no_think"
                    break

        try:
            content, usage = _retry_with_backoff(
                lambda: self._make_request_and_validate(
                    model_id, messages, temperature, max_tokens, disable_thinking
                ),
                max_retries=self.max_retries,
            )
            return ChatResponse(
                content=content, model=model_id, usage=usage, error=None
            )
        except Exception as e:
            logger.error("API error: %s", e)
            return ChatResponse(
                content="", model=model_id, usage=None, error=str(e)
            )

    def _make_request_and_validate(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        disable_thinking: bool,
    ) -> tuple[str, dict[str, int] | None]:
        client = self._get_client(model_id)
        response = client.chat_completion(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        if content is None or (isinstance(content, str) and content.strip() == ""):
            raise EmptyResponseError(
                f"Model returned empty/None response for {model_id}"
            )
        if disable_thinking and "qwen3" in model_id.lower():
            content = _strip_think_blocks(content)

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return content, usage


# ── VLLMClient ──


class VLLMClient:
    """Client for vLLM's OpenAI-compatible API endpoint.

    Uses the openai Python SDK to talk to a vLLM server.
    The server exposes an OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for VLLMClient. Install with: uv add openai"
            )
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout
        )

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        disable_thinking: bool = True,
    ) -> ChatResponse:
        """Send a chat completion request to a vLLM server.

        Args:
            model_id: Model ID (must match what vLLM was started with).
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            disable_thinking: If True, append /no_think for Qwen3 models.

        Returns:
            ChatResponse with content and metadata.
        """
        # For Qwen3: append /no_think to the last user message
        if disable_thinking and "qwen3" in model_id.lower():
            messages = [m.copy() for m in messages]
            for m in reversed(messages):
                if m["role"] == "user":
                    m["content"] = f"{m['content']} /no_think"
                    break

        try:
            content, usage = _retry_with_backoff(
                lambda: self._make_request_and_validate(
                    model_id, messages, temperature, max_tokens, disable_thinking
                ),
                max_retries=self.max_retries,
            )
            return ChatResponse(
                content=content, model=model_id, usage=usage, error=None
            )
        except Exception as e:
            logger.error("vLLM API error: %s", e)
            return ChatResponse(
                content="", model=model_id, usage=None, error=str(e)
            )

    def _make_request_and_validate(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        disable_thinking: bool,
    ) -> tuple[str, dict[str, int] | None]:
        response = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None or (isinstance(content, str) and content.strip() == ""):
            raise EmptyResponseError(
                f"vLLM returned empty/None response for {model_id}"
            )
        if disable_thinking and "qwen3" in model_id.lower():
            content = _strip_think_blocks(content)

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return content, usage

    def wait_until_ready(
        self, timeout_seconds: int = 600, poll_interval: int = 15
    ) -> bool:
        """Poll /v1/models until the vLLM server responds.

        Used by LambdaCloudManager to wait for model loading after instance boot.

        Returns:
            True if the server became ready within the timeout, False otherwise.
        """
        import httpx

        models_url = self.base_url.rstrip("/") + "/models"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                resp = httpx.get(models_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("data"):
                        logger.info("vLLM server ready, models: %s",
                                    [m.get("id") for m in data["data"]])
                        return True
            except Exception:
                pass
            logger.info("Waiting for vLLM server at %s ...", models_url)
            time.sleep(poll_interval)
        return False
