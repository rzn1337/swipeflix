"""Prometheus metrics for LLM operations."""

from prometheus_client import Counter, Gauge, Histogram

# LLM Request Metrics
llm_requests_total = Counter(
    "swipeflix_llm_requests_total",
    "Total number of LLM API requests",
    ["model", "status"],
)

llm_request_duration = Histogram(
    "swipeflix_llm_request_duration_seconds",
    "LLM request latency in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Token Metrics
llm_tokens_total = Counter(
    "swipeflix_llm_tokens_total",
    "Total tokens used by LLM",
    ["model", "type"],
)

llm_tokens_per_request = Histogram(
    "swipeflix_llm_tokens_per_request",
    "Tokens per LLM request",
    ["model"],
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

# Cache Metrics
llm_cache_hits = Counter(
    "swipeflix_llm_cache_hits_total",
    "Total LLM cache hits",
    ["cache_type"],
)

llm_cache_misses = Counter(
    "swipeflix_llm_cache_misses_total",
    "Total LLM cache misses",
)

# Error Metrics
llm_errors_total = Counter(
    "swipeflix_llm_errors_total",
    "Total LLM errors",
    ["error_type"],
)

# Rate Limit Metrics
llm_rate_limit_remaining = Gauge(
    "swipeflix_llm_rate_limit_remaining",
    "Remaining rate limit quota",
    ["limit_type"],
)

llm_rate_limit_blocked = Counter(
    "swipeflix_llm_rate_limit_blocked_total",
    "Total requests blocked by rate limiting",
)

# RAG Metrics
rag_retrieval_duration = Histogram(
    "swipeflix_rag_retrieval_duration_seconds",
    "RAG document retrieval latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

rag_documents_retrieved = Histogram(
    "swipeflix_rag_documents_retrieved",
    "Number of documents retrieved per RAG query",
    buckets=(1, 2, 3, 5, 10, 20),
)

rag_relevance_score = Histogram(
    "swipeflix_rag_relevance_score",
    "Relevance scores of retrieved documents",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Guardrail Metrics
guardrail_checks_total = Counter(
    "swipeflix_guardrail_checks_total",
    "Total guardrail checks performed",
    ["guardrail_type", "result"],
)

guardrail_violations_total = Counter(
    "swipeflix_guardrail_violations_total",
    "Total guardrail violations detected",
    ["guardrail_type", "severity"],
)

guardrail_check_duration = Histogram(
    "swipeflix_guardrail_check_duration_seconds",
    "Guardrail check latency",
    ["guardrail_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

# Prompt Experiment Metrics
prompt_experiment_runs = Counter(
    "swipeflix_prompt_experiment_runs_total",
    "Total prompt experiment runs",
    ["strategy", "variant"],
)

prompt_experiment_scores = Histogram(
    "swipeflix_prompt_experiment_scores",
    "Prompt experiment evaluation scores",
    ["strategy", "metric"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# A/B Testing Metrics
ab_test_impressions = Counter(
    "swipeflix_ab_test_impressions_total",
    "A/B test impressions",
    ["experiment", "variant"],
)

ab_test_conversions = Counter(
    "swipeflix_ab_test_conversions_total",
    "A/B test conversions",
    ["experiment", "variant"],
)

# Cost Tracking (estimated)
llm_estimated_cost_dollars = Counter(
    "swipeflix_llm_estimated_cost_dollars",
    "Estimated LLM API cost in dollars",
    ["model"],
)


def update_rate_limit_metrics(remaining: dict) -> None:
    """Update rate limit gauge metrics."""
    llm_rate_limit_remaining.labels(limit_type="rpm").set(
        remaining.get("rpm_remaining", 0)
    )
    llm_rate_limit_remaining.labels(limit_type="tpm").set(
        remaining.get("tpm_remaining", 0)
    )
    llm_rate_limit_remaining.labels(limit_type="daily").set(
        remaining.get("daily_remaining", 0)
    )
