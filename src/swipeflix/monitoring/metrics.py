"""Prometheus metrics definitions."""

from prometheus_client import Counter, Gauge, Histogram

# HTTP Request metrics
http_requests_total = Counter(
    "swipeflix_http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "swipeflix_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Inference metrics
inference_duration_seconds = Histogram(
    "swipeflix_inference_duration_seconds",
    "Model inference latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Model version tracking
model_version_info = Gauge(
    "swipeflix_model_version_info",
    "Current model version in use",
    ["version"],
)

# System metrics
model_load_status = Gauge(
    "swipeflix_model_load_status",
    "Model load status (1=loaded, 0=not loaded)",
)

predictions_total = Counter(
    "swipeflix_predictions_total",
    "Total number of predictions made",
    ["model_version"],
)

prediction_errors_total = Counter(
    "swipeflix_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

