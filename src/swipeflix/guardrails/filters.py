"""Individual filter implementations for guardrails.

Provides modular filters that can be combined in the pipeline.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class FilterResult:
    """Result from a filter check."""

    passed: bool
    message: str
    details: Dict[str, Any]
    filtered_text: Optional[str] = None


class BaseFilter(ABC):
    """Abstract base class for filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter name."""
        pass

    @abstractmethod
    def check(self, text: str, **kwargs) -> FilterResult:
        """Run the filter check."""
        pass


class PIIFilter(BaseFilter):
    """Filter for detecting and redacting PII.

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Names (optional, using patterns)
    """

    def __init__(self, redact: bool = True):
        self.redact = redact

        # PII detection patterns
        self.patterns = {
            "email": {
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "replacement": "[EMAIL REDACTED]",
            },
            "phone_us": {
                "pattern": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "replacement": "[PHONE REDACTED]",
            },
            "ssn": {
                "pattern": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
                "replacement": "[SSN REDACTED]",
            },
            "credit_card": {
                "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                "replacement": "[CARD REDACTED]",
            },
            "ip_address": {
                "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                "replacement": "[IP REDACTED]",
            },
        }

    @property
    def name(self) -> str:
        return "pii_filter"

    def check(self, text: str, **kwargs) -> FilterResult:
        """Check for PII in text."""
        detections = []
        filtered_text = text

        for pii_type, config in self.patterns.items():
            pattern = config["pattern"]
            matches = re.findall(pattern, text, re.IGNORECASE)

            if matches:
                detections.append(
                    {
                        "type": pii_type,
                        "count": len(matches),
                    }
                )

                if self.redact:
                    filtered_text = re.sub(
                        pattern,
                        config["replacement"],
                        filtered_text,
                        flags=re.IGNORECASE,
                    )

        passed = len(detections) == 0

        return FilterResult(
            passed=passed,
            message=f"Detected {len(detections)} PII types"
            if detections
            else "No PII detected",
            details={"detections": detections},
            filtered_text=filtered_text if self.redact else None,
        )


class PromptInjectionFilter(BaseFilter):
    """Filter for detecting prompt injection attempts.

    Detects patterns commonly used in prompt injection attacks:
    - System prompt overrides
    - Role-playing commands
    - Instruction ignoring commands
    - Jailbreak attempts
    """

    def __init__(self):
        # Injection patterns with severity levels
        self.patterns = [
            # High severity - direct prompt manipulation
            (
                r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
                "high",
            ),
            (r"disregard\s+(all\s+)?(previous|above|prior)", "high"),
            (r"forget\s+(everything|all|your)\s*(instructions?|prompts?)?", "high"),
            (r"system\s*[:\-]\s*", "high"),
            (r"<\s*/?system\s*>", "high"),
            (r"\[/?INST\]", "high"),
            (r"###\s*(?:instruction|system|prompt)", "high"),
            # Medium severity - role manipulation
            (r"you\s+are\s+now\s+(?:a|an)\s+", "medium"),
            (r"pretend\s+(you\s+are|to\s+be)", "medium"),
            (r"act\s+as\s+(?:if|though)\s+you", "medium"),
            (r"roleplay\s+as", "medium"),
            # Low severity - suspicious patterns
            (r"override\s+(safety|content|guardrail)", "low"),
            (r"bypass\s+(filter|safety|restriction)", "low"),
            (r"jailbreak", "low"),
            (r"DAN\s*mode", "low"),
        ]

    @property
    def name(self) -> str:
        return "prompt_injection_filter"

    def check(self, text: str, **kwargs) -> FilterResult:
        """Check for prompt injection attempts."""
        text_lower = text.lower()
        detections = []

        for pattern, severity in self.patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                detections.append(
                    {
                        "pattern": pattern,
                        "severity": severity,
                        "matches": len(matches),
                    }
                )

        # Determine overall severity
        max_severity = "none"
        for d in detections:
            if d["severity"] == "high":
                max_severity = "high"
                break
            elif d["severity"] == "medium" and max_severity != "high":
                max_severity = "medium"
            elif d["severity"] == "low" and max_severity == "none":
                max_severity = "low"

        passed = max_severity in ("none", "low")

        return FilterResult(
            passed=passed,
            message=f"Injection attempt detected (severity: {max_severity})"
            if detections
            else "No injection detected",
            details={
                "detections": detections,
                "max_severity": max_severity,
            },
        )


class ToxicityFilter(BaseFilter):
    """Filter for detecting toxic/harmful content.

    Uses either:
    - ML-based detection (detoxify library)
    - Keyword-based fallback
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_ml: bool = True,
    ):
        self.threshold = threshold
        self.use_ml = use_ml
        self._model = None

        # Try to load ML model
        if use_ml:
            try:
                from detoxify import Detoxify

                self._model = Detoxify("original")
                logger.info("Toxicity ML model loaded")
            except ImportError:
                logger.warning("detoxify not available, using keyword fallback")
                self._model = None

        # Keyword fallback
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
            "violence",
            "abuse",
        ]

    @property
    def name(self) -> str:
        return "toxicity_filter"

    def check(self, text: str, **kwargs) -> FilterResult:
        """Check for toxic content."""
        if self._model:
            return self._check_ml(text)
        return self._check_keywords(text)

    def _check_ml(self, text: str) -> FilterResult:
        """ML-based toxicity check."""
        try:
            predictions = self._model.predict(text)
            max_category = max(predictions, key=predictions.get)
            max_score = predictions[max_category]

            passed = max_score < self.threshold

            return FilterResult(
                passed=passed,
                message=f"Toxicity: {max_category}={max_score:.2f}"
                if not passed
                else "No toxicity detected",
                details={
                    "scores": {k: round(v, 4) for k, v in predictions.items()},
                    "max_category": max_category,
                    "max_score": max_score,
                    "threshold": self.threshold,
                },
            )
        except Exception as e:
            logger.error(f"Toxicity ML check failed: {e}")
            return self._check_keywords(text)

    def _check_keywords(self, text: str) -> FilterResult:
        """Keyword-based toxicity check."""
        text_lower = text.lower()
        found_keywords = [kw for kw in self.toxic_keywords if kw in text_lower]

        passed = len(found_keywords) == 0

        return FilterResult(
            passed=passed,
            message=f"Found toxic keywords: {found_keywords}"
            if found_keywords
            else "No toxic keywords",
            details={
                "found_keywords": found_keywords,
                "method": "keyword",
            },
        )


class HallucinationFilter(BaseFilter):
    """Filter for detecting hallucinated content.

    Checks if generated content is grounded in provided context.
    """

    def __init__(
        self,
        strict_mode: bool = False,
    ):
        self.strict_mode = strict_mode

    @property
    def name(self) -> str:
        return "hallucination_filter"

    def check(
        self,
        text: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> FilterResult:
        """Check for hallucinations against context."""
        if not context:
            return FilterResult(
                passed=True,
                message="No context provided for hallucination check",
                details={"skipped": True},
            )

        # Extract facts from context
        context_facts = self._extract_context_facts(context)

        # Extract claims from response
        response_claims = self._extract_claims(text)

        # Check grounding
        grounded_claims = []
        ungrounded_claims = []

        for claim in response_claims:
            if self._is_grounded(claim, context_facts):
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)

        # Calculate grounding ratio
        total_claims = len(response_claims)
        grounding_ratio = len(grounded_claims) / max(total_claims, 1)

        # Determine pass/fail
        if self.strict_mode:
            passed = len(ungrounded_claims) == 0
        else:
            passed = grounding_ratio >= 0.5

        return FilterResult(
            passed=passed,
            message=f"Grounding ratio: {grounding_ratio:.2f}",
            details={
                "total_claims": total_claims,
                "grounded_claims": len(grounded_claims),
                "ungrounded_claims": ungrounded_claims[:5],  # Limit to first 5
                "grounding_ratio": grounding_ratio,
            },
        )

    def _extract_context_facts(self, context: List[Dict]) -> set:
        """Extract fact keywords from context."""
        facts = set()

        for doc in context:
            # Add title
            if doc.get("title"):
                facts.add(doc["title"].lower())
                # Add title words
                for word in doc["title"].lower().split():
                    if len(word) > 3:
                        facts.add(word)

            # Add director
            if doc.get("director"):
                facts.add(doc["director"].lower())

            # Add year
            if doc.get("year"):
                facts.add(str(doc["year"]))

            # Add genres
            if doc.get("genre"):
                for genre in str(doc["genre"]).split("|"):
                    facts.add(genre.strip().lower())

        return facts

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claim-like segments from text."""
        # Simple: split into sentences
        sentences = re.split(r"[.!?]", text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims

    def _is_grounded(self, claim: str, context_facts: set) -> bool:
        """Check if a claim is grounded in context facts."""
        claim_lower = claim.lower()

        # Check if any significant fact appears in the claim
        matches = sum(1 for fact in context_facts if fact in claim_lower)
        return matches >= 1


class ContentPolicyFilter(BaseFilter):
    """Filter for content policy violations.

    Checks for:
    - Illegal content references
    - Adult content
    - Violent content
    - Misinformation patterns
    """

    def __init__(self):
        # Categories to filter
        self.blocked_patterns = {
            "illegal_content": [
                r"how\s+to\s+(make|build|create)\s+(bomb|weapon|drug)",
                r"where\s+to\s+buy\s+(drug|weapon|illegal)",
            ],
            "adult_content": [
                r"explicit\s+sexual",
                r"pornograph",
            ],
            "violence": [
                r"detailed\s+(instructions?|guide)\s+to\s+(kill|hurt|harm)",
                r"glorif(y|ying)\s+violence",
            ],
        }

    @property
    def name(self) -> str:
        return "content_policy_filter"

    def check(self, text: str, **kwargs) -> FilterResult:
        """Check for content policy violations."""
        text_lower = text.lower()
        violations = []

        for category, patterns in self.blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    violations.append(
                        {
                            "category": category,
                            "pattern": pattern,
                        }
                    )

        passed = len(violations) == 0

        return FilterResult(
            passed=passed,
            message=f"Policy violations: {len(violations)}"
            if violations
            else "No policy violations",
            details={"violations": violations},
        )
