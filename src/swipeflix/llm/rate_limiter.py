"""Rate limiter for Gemini API with free tier limits.

Gemini 2.5 Flash Free Tier Limits:
- 5 Requests Per Minute (RPM)
- 250K Input Tokens Per Minute
- 20 Requests Per Day
"""

import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

try:
    from swipeflix.config import settings

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    settings = None


class RateLimiter:
    """Thread-safe rate limiter for Gemini API free tier."""

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
    ):
        # Use settings if available, otherwise use defaults
        if CONFIG_AVAILABLE and settings:
            self.rpm_limit = (
                requests_per_minute
                if requests_per_minute is not None
                else settings.llm_rpm_limit
            )
            self.tpm_limit = (
                tokens_per_minute
                if tokens_per_minute is not None
                else settings.llm_tpm_limit
            )
            self.daily_limit = (
                requests_per_day
                if requests_per_day is not None
                else settings.llm_daily_limit
            )
        else:
            self.rpm_limit = requests_per_minute or 5
            self.tpm_limit = tokens_per_minute or 250000
            self.daily_limit = requests_per_day or 20

        # Track requests with timestamps
        self._minute_requests: deque = deque()
        self._minute_tokens: deque = deque()
        self._daily_requests: deque = deque()

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.blocked_requests = 0

        logger.info(
            f"RateLimiter initialized: {self.rpm_limit} RPM, "
            f"{self.tpm_limit} TPM, {self.daily_limit} daily"
        )

    def _cleanup_old_entries(self, current_time: datetime) -> None:
        """Remove entries older than their time windows."""
        minute_ago = current_time - timedelta(minutes=1)
        day_ago = current_time - timedelta(days=1)

        # Cleanup minute window
        while self._minute_requests and self._minute_requests[0] < minute_ago:
            self._minute_requests.popleft()

        while self._minute_tokens and self._minute_tokens[0][0] < minute_ago:
            self._minute_tokens.popleft()

        # Cleanup daily window
        while self._daily_requests and self._daily_requests[0] < day_ago:
            self._daily_requests.popleft()

    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if a request can be made within rate limits."""
        with self._lock:
            current_time = datetime.now()
            self._cleanup_old_entries(current_time)

            # Check all limits
            rpm_ok = len(self._minute_requests) < self.rpm_limit
            tpm_ok = (
                sum(t[1] for t in self._minute_tokens) + estimated_tokens
                <= self.tpm_limit
            )
            daily_ok = len(self._daily_requests) < self.daily_limit

            return rpm_ok and tpm_ok and daily_ok

    def wait_if_needed(
        self, estimated_tokens: int = 1000, timeout: float = 300.0
    ) -> bool:
        """Wait until a request can be made or timeout.

        Returns:
            True if request can proceed, False if timed out
        """
        start_time = time.time()
        wait_attempts = 0

        while not self.can_make_request(estimated_tokens):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Rate limit timeout exceeded after {elapsed:.1f}s")
                self.blocked_requests += 1
                return False

            # Calculate optimal sleep time with exponential backoff
            with self._lock:
                current_time = datetime.now()
                self._cleanup_old_entries(current_time)

                # Find when next slot opens
                if len(self._daily_requests) >= self.daily_limit:
                    logger.error(
                        "Daily limit reached! Cannot make more requests today."
                    )
                    return False
                elif len(self._minute_requests) >= self.rpm_limit:
                    # Wait for next minute window
                    if self._minute_requests:
                        oldest = self._minute_requests[0]
                        wait_time = (
                            oldest + timedelta(minutes=1) - current_time
                        ).total_seconds()
                        wait_time = max(
                            12.0, min(wait_time, 65.0)
                        )  # At least 12s, max 65s
                    else:
                        wait_time = 15.0  # Default 15s wait
                elif (
                    sum(t[1] for t in self._minute_tokens) + estimated_tokens
                    > self.tpm_limit
                ):
                    # Token limit - wait for token window to clear
                    wait_time = 12.0  # Wait 12 seconds for token window
                else:
                    # Small delay with exponential backoff
                    wait_time = min(2.0 * (1.5**wait_attempts), 30.0)
                    wait_attempts += 1

            logger.info(
                f"Rate limited. Waiting {wait_time:.1f}s before retry... (attempt {wait_attempts})"
            )
            time.sleep(wait_time)

        return True

    def record_request(self, tokens_used: int) -> None:
        """Record a completed request."""
        with self._lock:
            current_time = datetime.now()
            self._minute_requests.append(current_time)
            self._minute_tokens.append((current_time, tokens_used))
            self._daily_requests.append(current_time)

            self.total_requests += 1
            self.total_tokens += tokens_used

            logger.debug(
                f"Request recorded: {tokens_used} tokens. "
                f"Minute: {len(self._minute_requests)}/{self.rpm_limit}, "
                f"Daily: {len(self._daily_requests)}/{self.daily_limit}"
            )

    def get_status(self) -> dict:
        """Get current rate limiter status."""
        with self._lock:
            current_time = datetime.now()
            self._cleanup_old_entries(current_time)

            return {
                "requests_this_minute": len(self._minute_requests),
                "tokens_this_minute": sum(t[1] for t in self._minute_tokens),
                "requests_today": len(self._daily_requests),
                "rpm_limit": self.rpm_limit,
                "tpm_limit": self.tpm_limit,
                "daily_limit": self.daily_limit,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "blocked_requests": self.blocked_requests,
            }

    def get_remaining_quota(self) -> dict:
        """Get remaining quota for each limit."""
        with self._lock:
            current_time = datetime.now()
            self._cleanup_old_entries(current_time)

            return {
                "rpm_remaining": max(0, self.rpm_limit - len(self._minute_requests)),
                "tpm_remaining": max(
                    0, self.tpm_limit - sum(t[1] for t in self._minute_tokens)
                ),
                "daily_remaining": max(0, self.daily_limit - len(self._daily_requests)),
            }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance.

    Uses settings from config.py for rate limits (RPM, TPM, daily).
    """
    global _rate_limiter
    if _rate_limiter is None:
        if CONFIG_AVAILABLE and settings:
            _rate_limiter = RateLimiter(
                requests_per_minute=settings.llm_rpm_limit,
                tokens_per_minute=settings.llm_tpm_limit,
                requests_per_day=settings.llm_daily_limit,
            )
        else:
            _rate_limiter = RateLimiter()
    return _rate_limiter
