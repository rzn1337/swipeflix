# Security Policy

## Overview

SwipeFlix implements multiple security layers to protect against common LLM
vulnerabilities and ensure responsible AI use. This document describes our security
measures, vulnerability handling process, and compliance guidelines.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :white_check_mark: |
| \< 1.0  | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
1. Email security concerns to the repository maintainers
1. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work on a fix.

## Security Measures

### 1. Prompt Injection Defense

SwipeFlix implements multiple layers of defense against prompt injection attacks:

#### Input Validation (`src/swipeflix/guardrails/validators.py`)

```python
# Detected patterns include:
- "ignore previous instructions"
- "disregard all prior"
- "system:" / "<system>"
- "[INST]" markers
- "pretend you are"
- "act as if"
- Jailbreak attempts ("DAN mode")
```

#### How It Works

1. **Pattern Matching**: All user inputs are scanned against known injection patterns
1. **Severity Classification**: Detections are classified as LOW/MEDIUM/HIGH/CRITICAL
1. **Blocking**: CRITICAL severity blocks the request entirely
1. **Logging**: All attempts are logged for security monitoring

#### Configuration

```python
from swipeflix.guardrails.validators import InputValidator

validator = InputValidator(
    max_input_length=2000,
    enable_pii_filter=True,
    enable_injection_filter=True,
)

result = validator.validate(user_input)
if not result.passed:
    # Handle blocked input
```

### 2. PII Detection and Redaction

Personal Identifiable Information is automatically detected and redacted:

| PII Type    | Pattern               | Replacement        |
| ----------- | --------------------- | ------------------ |
| Email       | `user@example.com`    | `[EMAIL REDACTED]` |
| Phone (US)  | `555-123-4567`        | `[PHONE REDACTED]` |
| SSN         | `123-45-6789`         | `[SSN REDACTED]`   |
| Credit Card | `4111-1111-1111-1111` | `[CARD REDACTED]`  |
| IP Address  | `192.168.1.1`         | `[IP REDACTED]`    |

#### Implementation

```python
from swipeflix.guardrails.filters import PIIFilter

pii_filter = PIIFilter(redact=True)
result = pii_filter.check(text)
sanitized_text = result.filtered_text
```

### 3. Output Moderation

#### Toxicity Filtering

- Uses ML-based detection (Detoxify library) when available
- Falls back to keyword detection
- Configurable threshold (default: 0.7)

```python
from swipeflix.guardrails.filters import ToxicityFilter

filter = ToxicityFilter(threshold=0.7, use_ml=True)
result = filter.check(output_text)
```

#### Hallucination Detection

- Validates LLM outputs against provided context
- Checks for ungrounded claims (years, facts not in context)
- Flags responses that don't reference source material

```python
from swipeflix.guardrails.filters import HallucinationFilter

filter = HallucinationFilter(strict_mode=False)
result = filter.check(response, context=source_documents)
```

### 4. Rate Limiting

Protection against abuse through strict rate limits:

| Limit | Value           | Purpose              |
| ----- | --------------- | -------------------- |
| RPM   | 5 requests/min  | Prevent API abuse    |
| TPM   | 250K tokens/min | Cost control         |
| Daily | 20 requests/day | Free tier compliance |

Rate limiting is implemented in `src/swipeflix/llm/rate_limiter.py`.

### 5. API Security

#### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production Recommendation**: Restrict `allow_origins` to specific domains.

#### Input Validation

All API endpoints use Pydantic models with:

- Type validation
- Length limits
- Pattern validation

Example:

```python
class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
```

## Dependency Security

### Automated Scanning

We use `pip-audit` for dependency vulnerability scanning:

```bash
# Run security scan
pip-audit --no-deps

# CI fails on CRITICAL vulnerabilities
pip-audit --no-deps --format=columns | grep -i CRITICAL && exit 1
```

### Known Vulnerabilities

Check current status:

```bash
make security-scan
```

### Dependency Updates

Dependencies are regularly updated. Review `requirements.txt` and `requirements-llm.txt`
for versions.

## Data Privacy

### What We Store

| Data Type     | Storage               | Retention    |
| ------------- | --------------------- | ------------ |
| User queries  | Memory cache          | Session only |
| LLM responses | Disk cache (optional) | 24 hours     |
| Movie data    | Local/S3              | Permanent    |
| Metrics       | Prometheus            | 15 days      |

### What We DON'T Store

- Personal user information
- Raw API keys (loaded from environment)
- Unredacted PII

### Data Flow

```
User Query → Input Validation → PII Redaction → RAG Pipeline → Output Validation → Response
                    ↓                                              ↓
               Blocked if         →                         Filtered if
               injection detected                            toxic/hallucinated
```

## Responsible AI Guidelines

### Our Commitments

1. **Transparency**: Users know they're interacting with an AI
1. **Accuracy**: Responses are grounded in provided context
1. **Safety**: Content is filtered for harmful material
1. **Privacy**: PII is detected and redacted
1. **Fairness**: No discriminatory recommendations

### Content Policy

The system filters:

- Illegal content references
- Adult/explicit content
- Detailed violence instructions
- Misinformation patterns

### Guardrail Integration

Guardrails are integrated throughout the RAG pipeline:

```
                    ┌─────────────────┐
                    │  User Input     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Input Validator │ ← PII, Injection checks
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │      RAG Pipeline           │
              │  (Retrieval + Generation)   │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │Output Validator │ ← Toxicity, Hallucination checks
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Response     │
                    └─────────────────┘
```

## Monitoring & Alerting

### Security Metrics

Track in Grafana dashboard:

- `swipeflix_guardrail_violations_total` - Violation count by type
- `swipeflix_guardrail_checks_total` - Check pass/fail rates
- `swipeflix_llm_errors_total` - Error patterns

### Alert Rules (Recommended)

```yaml
# Prometheus alerting rules
groups:
  - name: security
    rules:
      - alert: HighGuardrailViolations
        expr: rate(swipeflix_guardrail_violations_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High rate of guardrail violations

      - alert: PromptInjectionAttempts
        expr: increase(swipeflix_guardrail_violations_total{guardrail_type="prompt_injection"}[1h]) > 10
        labels:
          severity: critical
        annotations:
          summary: Multiple prompt injection attempts detected
```

## Compliance

### GDPR Considerations

- PII redaction enabled by default
- No persistent user data storage
- Data minimization principles applied

### SOC 2 Alignment

- Access logging enabled
- Input/output validation
- Monitoring and alerting
- Regular security scanning

## Security Checklist

Before deployment, verify:

- [ ] `GEMINI_API_KEY` stored securely (not in code)
- [ ] CORS origins restricted for production
- [ ] Rate limiting enabled
- [ ] Guardrails enabled (PII, injection, toxicity)
- [ ] pip-audit shows no CRITICAL vulnerabilities
- [ ] Monitoring dashboards configured
- [ ] Alerting rules deployed
- [ ] API access logging enabled

## Updates

This security policy is reviewed and updated with each major release.

______________________________________________________________________

*Last updated: December 2024* *SwipeFlix Security Team*
