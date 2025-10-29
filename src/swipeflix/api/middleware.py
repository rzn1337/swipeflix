"""Custom middleware for FastAPI application."""

import time
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from swipeflix.monitoring.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track metrics."""
        start_time = time.time()
        method = request.method
        path = request.url.path

        try:
            response = await call_next(request)
            status = response.status_code

            # Record metrics
            http_requests_total.labels(
                method=method, endpoint=path, status=status
            ).inc()

            duration = time.time() - start_time
            http_request_duration_seconds.labels(method=method, endpoint=path).observe(
                duration
            )

            # Log request
            logger.info(
                f"{method} {path} - Status: {status} - Duration: {duration:.3f}s"
            )

            return response

        except Exception as e:
            # Record error metric
            http_requests_total.labels(method=method, endpoint=path, status=500).inc()

            duration = time.time() - start_time
            http_request_duration_seconds.labels(method=method, endpoint=path).observe(
                duration
            )

            logger.error(
                f"{method} {path} - Error: {str(e)} - Duration: {duration:.3f}s"
            )
            raise


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details."""
        logger.debug(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        response = await call_next(request)

        logger.debug(
            f"Outgoing response: {response.status_code} for {request.method} {request.url.path}"
        )

        return response
