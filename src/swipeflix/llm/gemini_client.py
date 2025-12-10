"""Gemini 2.5 Flash client with caching and rate limiting."""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from diskcache import Cache
from loguru import logger

try:
    import google.generativeai as genai

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. LLM features disabled.")

from swipeflix.config import settings
from swipeflix.llm.rate_limiter import get_rate_limiter
from swipeflix.monitoring.llm_metrics import (
    llm_cache_hits,
    llm_errors_total,
    llm_request_duration,
    llm_requests_total,
    llm_tokens_total,
)


class GeminiClient:
    """Client for Gemini 2.5 Flash API with caching and rate limiting."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        enable_caching: Optional[bool] = None,
    ):
        # Use settings as defaults
        self.model_name = model_name or settings.gemini_model or "gemini-1.5-flash"
        self.enable_caching = (
            enable_caching if enable_caching is not None else settings.llm_cache_enabled
        )
        self.rate_limiter = get_rate_limiter()
        self._last_request_time = 0  # Track last request time for automatic spacing

        # API key from env, parameter, or settings
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or settings.gemini_api_key

        # Cache settings
        cache_dir = cache_dir or settings.llm_cache_dir
        cache_ttl = cache_ttl or settings.llm_cache_ttl

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. LLM features will be limited.")
            self._initialized = False
            return

        if not GENAI_AVAILABLE:
            logger.warning(
                "google-generativeai not available. Install with: pip install google-generativeai"
            )
            self._initialized = False
            return

        # Initialize Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self._initialized = True
            logger.info(f"GeminiClient initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model '{self.model_name}': {e}")
            logger.error("Common issues:")
            logger.error("  1. Invalid API key")
            logger.error(
                "  2. Invalid model name (try 'gemini-1.5-flash' or 'gemini-2.0-flash-exp')"
            )
            logger.error("  3. Network connectivity issues")
            self._initialized = False
            return

        # In-memory cache (fast, small)
        self._memory_cache: TTLCache = TTLCache(maxsize=100, ttl=cache_ttl)

        # Disk cache (persistent, larger)
        if self.enable_caching:
            os.makedirs(cache_dir, exist_ok=True)
            self._disk_cache = Cache(cache_dir)
        else:
            self._disk_cache = None

    def is_available(self) -> bool:
        """Check if the client is properly initialized."""
        return self._initialized

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key from prompt and parameters."""
        cache_data = {"prompt": prompt, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Try to get response from cache."""
        # Check memory cache first
        if cache_key in self._memory_cache:
            llm_cache_hits.labels(cache_type="memory").inc()
            return self._memory_cache[cache_key]

        # Check disk cache
        if self._disk_cache and cache_key in self._disk_cache:
            response = self._disk_cache[cache_key]
            # Promote to memory cache
            self._memory_cache[cache_key] = response
            llm_cache_hits.labels(cache_type="disk").inc()
            return response

        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache the response."""
        self._memory_cache[cache_key] = response
        if self._disk_cache:
            self._disk_cache[cache_key] = response

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Average ~4 characters per token for English text
        return len(text) // 4 + 1

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        top_p: float = 0.95,
        stop_sequences: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Generate text using Gemini.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0-2)
            max_output_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences
            use_cache: Whether to use caching

        Returns:
            Dict with 'text', 'tokens_used', 'cached', 'latency_ms'
        """
        if not self._initialized:
            return {
                "text": "[LLM not available - GEMINI_API_KEY not set]",
                "tokens_used": 0,
                "cached": False,
                "latency_ms": 0,
                "error": "Client not initialized",
            }

        # Check cache
        cache_key = self._get_cache_key(
            prompt,
            system=system_instruction,
            temp=temperature,
            max_tokens=max_output_tokens,
        )

        if use_cache and self.enable_caching:
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.debug("Cache hit for prompt")
                return {**cached, "cached": True}

        # Estimate tokens and check rate limit
        estimated_tokens = self._estimate_tokens(prompt)
        if system_instruction:
            estimated_tokens += self._estimate_tokens(system_instruction)

        # Check rate limit with longer timeout and better error handling
        if not self.rate_limiter.can_make_request(estimated_tokens):
            # Check which limit is blocking
            status = self.rate_limiter.get_status()

            if status["requests_today"] >= self.rate_limiter.daily_limit:
                error_msg = f"Daily limit reached ({status['requests_today']}/{self.rate_limiter.daily_limit}). Try again tomorrow."
            elif status["requests_this_minute"] >= self.rate_limiter.rpm_limit:
                error_msg = f"Rate limit: {status['requests_this_minute']}/{self.rate_limiter.rpm_limit} requests this minute. Please wait ~60 seconds."
            elif (
                status["tokens_this_minute"] + estimated_tokens
                > self.rate_limiter.tpm_limit
            ):
                error_msg = f"Token limit: {status['tokens_this_minute']}/{self.rate_limiter.tpm_limit} tokens this minute. Please wait."
            else:
                error_msg = "Rate limit exceeded. Please try again later."

            logger.warning(f"Rate limit check failed: {error_msg}")
            llm_errors_total.labels(error_type="rate_limit").inc()
            return {
                "text": f"[{error_msg}]",
                "tokens_used": 0,
                "cached": False,
                "latency_ms": 0,
                "error": error_msg,
            }

        # Wait if needed (with automatic retry)
        if not self.rate_limiter.wait_if_needed(estimated_tokens, timeout=300.0):
            llm_errors_total.labels(error_type="rate_limit").inc()
            return {
                "text": "[Rate limit exceeded. Please try again later.]",
                "tokens_used": 0,
                "cached": False,
                "latency_ms": 0,
                "error": "Rate limit exceeded - timeout waiting for quota",
            }

        # Add automatic delay between requests to respect rate limits
        # Minimum 12 seconds between requests (5 RPM = 1 request per 12 seconds)
        if hasattr(self, "_last_request_time"):
            time_since_last = time.time() - self._last_request_time
            min_interval = 12.0  # 60 seconds / 5 RPM = 12 seconds minimum
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.debug(
                    f"Rate limiting: waiting {wait_time:.1f}s between requests"
                )
                time.sleep(wait_time)

        # Make API call
        start_time = time.time()
        self._last_request_time = start_time
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences or [],
            )

            # Build full prompt with system instruction
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract text and token usage
            # Handle different response formats
            if hasattr(response, "text") and response.text:
                text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Try to get text from candidates
                text = (
                    response.candidates[0].content.parts[0].text
                    if response.candidates[0].content.parts
                    else ""
                )
            else:
                text = ""
                logger.warning("Empty response from Gemini API")

            # Get token counts from response metadata if available
            tokens_used = estimated_tokens + self._estimate_tokens(text)
            if hasattr(response, "usage_metadata"):
                tokens_used = getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ) + getattr(response.usage_metadata, "candidates_token_count", 0)

            # Record metrics
            self.rate_limiter.record_request(tokens_used)
            llm_requests_total.labels(model=self.model_name, status="success").inc()
            llm_tokens_total.labels(model=self.model_name, type="total").inc(
                tokens_used
            )
            llm_request_duration.labels(model=self.model_name).observe(
                latency_ms / 1000
            )

            result = {
                "text": text,
                "tokens_used": tokens_used,
                "cached": False,
                "latency_ms": latency_ms,
            }

            # Cache the response
            if use_cache and self.enable_caching:
                self._cache_response(cache_key, result)

            logger.debug(
                f"Generated response: {len(text)} chars, "
                f"{tokens_used} tokens, {latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            error_msg = str(e)

            llm_requests_total.labels(model=self.model_name, status="error").inc()
            llm_errors_total.labels(error_type=error_type).inc()

            # Handle specific error types
            if (
                "429" in error_msg
                or "quota" in error_msg.lower()
                or "rate limit" in error_msg.lower()
            ):
                error_msg = "Rate limit exceeded. Please wait and try again."
                logger.warning(f"Rate limit hit: {error_msg}")
            elif (
                "401" in error_msg
                or "unauthorized" in error_msg.lower()
                or "api key" in error_msg.lower()
            ):
                error_msg = "Invalid API key. Please check GEMINI_API_KEY."
                logger.error(f"Authentication error: {error_msg}")
            elif "404" in error_msg or "not found" in error_msg.lower():
                error_msg = f"Model '{self.model_name}' not found. Try 'gemini-1.5-flash' or 'gemini-2.0-flash-exp'."
                logger.error(f"Model error: {error_msg}")
            else:
                logger.error(f"Gemini API error ({error_type}): {error_msg}")

            return {
                "text": f"[Error: {error_msg}]",
                "tokens_used": 0,
                "cached": False,
                "latency_ms": latency_ms,
                "error": error_msg,
                "error_type": error_type,
            }

    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate with automatic retry on failure."""
        last_error = None

        for attempt in range(max_retries):
            result = self.generate(prompt, **kwargs)

            if "error" not in result:
                return result

            last_error = result.get("error")
            logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            if "rate_limit" in last_error.lower():
                # Wait longer for rate limits
                time.sleep(min(60, 10 * (attempt + 1)))
            else:
                time.sleep(2**attempt)

        return {
            "text": f"[Failed after {max_retries} retries: {last_error}]",
            "tokens_used": 0,
            "cached": False,
            "latency_ms": 0,
            "error": last_error,
        }

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts (using local model for efficiency)."""
        # Use sentence-transformers for embeddings (free, fast, local)
        try:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            embeddings = self._embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using dummy embeddings."
            )
            return [[0.0] * 384 for _ in texts]


# Global client instance
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the global Gemini client instance.

    Uses settings from config.py for model name, API key, and cache settings.
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient(
            model_name=settings.gemini_model,
            api_key=settings.gemini_api_key,
            cache_dir=settings.llm_cache_dir,
            cache_ttl=settings.llm_cache_ttl,
            enable_caching=settings.llm_cache_enabled,
        )
    return _gemini_client
