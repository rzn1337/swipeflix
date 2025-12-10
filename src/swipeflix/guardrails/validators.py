"""Input and output validators for the RAG pipeline.

Implements:
- Input validation: PII detection, prompt injection filtering
- Output moderation: Toxicity threshold, hallucination detection
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from swipeflix.monitoring.llm_metrics import (
    guardrail_check_duration,
    guardrail_checks_total,
    guardrail_violations_total,
)


class GuardrailSeverity(Enum):
    """Severity levels for guardrail violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class GuardrailViolation:
    """A guardrail violation event."""

    guardrail_type: str
    severity: GuardrailSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    """Result from guardrail validation."""

    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    check_duration_ms: float = 0.0

    def add_violation(
        self,
        guardrail_type: str,
        severity: GuardrailSeverity,
        message: str,
        **details,
    ):
        """Add a violation to the result."""
        self.violations.append(
            GuardrailViolation(
                guardrail_type=guardrail_type,
                severity=severity,
                message=message,
                details=details,
            )
        )
        if severity in (GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL):
            self.passed = False


class InputValidator:
    """Validates user inputs before processing.

    Checks for:
    - PII (personal identifiable information)
    - Prompt injection attempts
    - Content policy violations
    - Input length limits
    """

    def __init__(
        self,
        max_input_length: int = 2000,
        enable_pii_filter: bool = True,
        enable_injection_filter: bool = True,
    ):
        self.max_input_length = max_input_length
        self.enable_pii_filter = enable_pii_filter
        self.enable_injection_filter = enable_injection_filter

        # PII patterns
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        }

        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
            r"disregard\s+(all\s+)?(previous|above|prior)",
            r"forget\s+(everything|all)",
            r"you\s+are\s+now\s+(?:a|an)",
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+(?:if|though)",
            r"system\s*:\s*",
            r"<\s*system\s*>",
            r"\[INST\]",
            r"###\s*(?:instruction|system)",
            r"override\s+(safety|content|guardrail)",
            r"jailbreak",
            r"DAN\s*mode",
        ]

        logger.info("InputValidator initialized")

    def validate(self, text: str) -> GuardrailResult:
        """Validate input text.

        Args:
            text: User input text

        Returns:
            GuardrailResult with validation status and any violations
        """
        start_time = time.time()
        result = GuardrailResult(passed=True, sanitized_text=text)

        # Check length
        if len(text) > self.max_input_length:
            result.add_violation(
                "input_length",
                GuardrailSeverity.WARNING,
                f"Input exceeds maximum length ({len(text)} > {self.max_input_length})",
                actual_length=len(text),
                max_length=self.max_input_length,
            )
            result.sanitized_text = text[: self.max_input_length]

        # Check for PII
        if self.enable_pii_filter:
            pii_result = self._check_pii(text)
            for violation in pii_result.violations:
                result.violations.append(violation)
            if pii_result.sanitized_text:
                result.sanitized_text = pii_result.sanitized_text

        # Check for prompt injection
        if self.enable_injection_filter:
            injection_result = self._check_injection(text)
            for violation in injection_result.violations:
                result.violations.append(violation)
                if violation.severity == GuardrailSeverity.CRITICAL:
                    result.passed = False

        # Update passed status
        for violation in result.violations:
            if violation.severity in (
                GuardrailSeverity.ERROR,
                GuardrailSeverity.CRITICAL,
            ):
                result.passed = False
                break

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        result.check_duration_ms = duration_ms
        guardrail_check_duration.labels(guardrail_type="input").observe(
            duration_ms / 1000
        )
        guardrail_checks_total.labels(
            guardrail_type="input", result="pass" if result.passed else "fail"
        ).inc()

        for violation in result.violations:
            guardrail_violations_total.labels(
                guardrail_type=violation.guardrail_type,
                severity=violation.severity.value,
            ).inc()

        return result

    def _check_pii(self, text: str) -> GuardrailResult:
        """Check for PII in text."""
        result = GuardrailResult(passed=True, sanitized_text=text)
        sanitized = text

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result.add_violation(
                    f"pii_{pii_type}",
                    GuardrailSeverity.WARNING,
                    f"Detected potential {pii_type}: {len(matches)} instance(s)",
                    pii_type=pii_type,
                    count=len(matches),
                )
                # Redact PII
                sanitized = re.sub(
                    pattern,
                    f"[REDACTED_{pii_type.upper()}]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        result.sanitized_text = sanitized
        return result

    def _check_injection(self, text: str) -> GuardrailResult:
        """Check for prompt injection attempts."""
        result = GuardrailResult(passed=True)
        text_lower = text.lower()

        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                result.add_violation(
                    "prompt_injection",
                    GuardrailSeverity.CRITICAL,
                    "Potential prompt injection detected",
                    pattern=pattern,
                )
                logger.warning(f"Prompt injection attempt detected: {pattern}")
                break

        return result


class OutputValidator:
    """Validates LLM outputs before returning to user.

    Checks for:
    - Toxicity/harmful content
    - Hallucinations (ungrounded claims)
    - Policy violations
    - Output format compliance
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.7,
        enable_toxicity_filter: bool = True,
        enable_hallucination_filter: bool = True,
        max_output_length: int = 5000,
    ):
        self.toxicity_threshold = toxicity_threshold
        self.enable_toxicity_filter = enable_toxicity_filter
        self.enable_hallucination_filter = enable_hallucination_filter
        self.max_output_length = max_output_length

        # Try to load toxicity detector
        self._toxicity_model = None
        if enable_toxicity_filter:
            try:
                from detoxify import Detoxify

                self._toxicity_model = Detoxify("original")
                logger.info("Toxicity model loaded")
            except ImportError:
                logger.warning("detoxify not available. Using keyword-based filtering.")

        # Toxic keyword patterns (fallback)
        self.toxic_keywords = [
            "hate",
            "kill",
            "murder",
            "terrorist",
            "racist",
            "sexist",
            "slur",
            "nazi",
            "genocide",
        ]

        logger.info("OutputValidator initialized")

    def validate(
        self,
        text: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> GuardrailResult:
        """Validate output text.

        Args:
            text: LLM output text
            context: Optional context used for generation (for hallucination check)

        Returns:
            GuardrailResult with validation status
        """
        start_time = time.time()
        result = GuardrailResult(passed=True, sanitized_text=text)

        # Check length
        if len(text) > self.max_output_length:
            result.add_violation(
                "output_length",
                GuardrailSeverity.WARNING,
                f"Output exceeds maximum length ({len(text)} > {self.max_output_length})",
            )
            result.sanitized_text = text[: self.max_output_length]

        # Check for toxicity
        if self.enable_toxicity_filter:
            toxicity_result = self._check_toxicity(text)
            for violation in toxicity_result.violations:
                result.violations.append(violation)

        # Check for hallucinations
        if self.enable_hallucination_filter and context:
            hallucination_result = self._check_hallucination(text, context)
            for violation in hallucination_result.violations:
                result.violations.append(violation)

        # Update passed status
        for violation in result.violations:
            if violation.severity in (
                GuardrailSeverity.ERROR,
                GuardrailSeverity.CRITICAL,
            ):
                result.passed = False
                break

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        result.check_duration_ms = duration_ms
        guardrail_check_duration.labels(guardrail_type="output").observe(
            duration_ms / 1000
        )
        guardrail_checks_total.labels(
            guardrail_type="output", result="pass" if result.passed else "fail"
        ).inc()

        for violation in result.violations:
            guardrail_violations_total.labels(
                guardrail_type=violation.guardrail_type,
                severity=violation.severity.value,
            ).inc()

        return result

    def _check_toxicity(self, text: str) -> GuardrailResult:
        """Check for toxic content."""
        result = GuardrailResult(passed=True)

        if self._toxicity_model:
            # Use ML-based detection
            try:
                predictions = self._toxicity_model.predict(text)
                max_score = max(predictions.values())

                if max_score > self.toxicity_threshold:
                    result.add_violation(
                        "toxicity",
                        GuardrailSeverity.ERROR,
                        f"Toxic content detected (score: {max_score:.2f})",
                        scores=predictions,
                        threshold=self.toxicity_threshold,
                    )
            except Exception as e:
                logger.error(f"Toxicity check failed: {e}")
        else:
            # Fallback to keyword detection
            text_lower = text.lower()
            for keyword in self.toxic_keywords:
                if keyword in text_lower:
                    result.add_violation(
                        "toxicity",
                        GuardrailSeverity.WARNING,
                        "Potentially inappropriate content detected",
                        keyword=keyword,
                    )
                    break

        return result

    def _check_hallucination(
        self,
        text: str,
        context: List[Dict[str, Any]],
    ) -> GuardrailResult:
        """Check for hallucinated content not grounded in context.

        This is a simple heuristic check. For production, consider
        using NLI-based models or more sophisticated fact-checking.
        """
        result = GuardrailResult(passed=True)

        # Extract claims (simple: look for quoted movie titles or years)
        year_pattern = r"\b(19|20)\d{2}\b"
        years_mentioned = set(re.findall(year_pattern, text))

        # Get years from context
        context_years = set()
        context_titles = set()
        for doc in context:
            if doc.get("year"):
                context_years.add(str(doc["year"]))
            if doc.get("title"):
                context_titles.add(doc["title"].lower())

        # Check for hallucinated years
        hallucinated_years = (
            years_mentioned - context_years - {"19", "20"}
        )  # Exclude partial matches
        if hallucinated_years and len(hallucinated_years) > 2:
            result.add_violation(
                "hallucination",
                GuardrailSeverity.WARNING,
                "Response may contain ungrounded information",
                hallucinated_years=list(hallucinated_years),
            )

        # Check if at least some context is mentioned
        text_lower = text.lower()
        mentioned_titles = sum(1 for title in context_titles if title in text_lower)

        if context_titles and mentioned_titles == 0:
            result.add_violation(
                "hallucination",
                GuardrailSeverity.INFO,
                "Response does not reference provided context",
                context_titles=list(context_titles)[:5],
            )

        return result
