"""RAG API routes for SwipeFlix LLM assistant."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from swipeflix.guardrails.validators import InputValidator, OutputValidator
from swipeflix.monitoring.llm_metrics import ab_test_impressions

# Import RAG components
try:
    from swipeflix.rag.generator import get_rag_generator
    from swipeflix.rag.retriever import get_retriever

    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG components not available: {e}")

# Import LLM components
try:
    from swipeflix.llm.rate_limiter import get_rate_limiter

    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    logger.warning(f"LLM components not available: {e}")


router = APIRouter(prefix="/rag", tags=["RAG Assistant"])

# Global instances
_input_validator: Optional[InputValidator] = None
_output_validator: Optional[OutputValidator] = None


def get_input_validator() -> InputValidator:
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator


def get_output_validator() -> OutputValidator:
    global _output_validator
    if _output_validator is None:
        _output_validator = OutputValidator()
    return _output_validator


# Request/Response schemas
class RAGQueryRequest(BaseModel):
    """Request for RAG Q&A."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of documents to retrieve"
    )
    prompt_strategy: str = Field(
        default="meta_prompt", description="Prompt strategy to use"
    )
    include_sources: bool = Field(default=True, description="Include source citations")


class RAGQueryResponse(BaseModel):
    """Response from RAG Q&A."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    grounded: bool
    confidence: str
    latency_ms: float
    tokens_used: int
    cached: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BlurbRequest(BaseModel):
    """Request for movie blurb generation."""

    movie_id: str = Field(..., description="Movie ID")
    tone: str = Field(
        default="engaging", description="Tone: engaging, casual, professional"
    )
    max_length: int = Field(
        default=100, ge=20, le=280, description="Maximum character length"
    )


class BlurbResponse(BaseModel):
    """Response with generated blurb."""

    movie_id: str
    blurb: str
    tone: str
    character_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReviewSummaryRequest(BaseModel):
    """Request for review summarization."""

    movie_id: str = Field(..., description="Movie ID")
    reviews: List[str] = Field(
        ..., min_items=1, max_items=10, description="Review texts"
    )


class ReviewSummaryResponse(BaseModel):
    """Response with review summary."""

    movie_id: str
    summary: str
    sentiment: str
    review_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StructuredDataRequest(BaseModel):
    """Request for structured data extraction."""

    text: str = Field(
        ..., min_length=10, max_length=5000, description="Free-form text about a movie"
    )


class StructuredDataResponse(BaseModel):
    """Response with extracted structured data."""

    extracted_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RationaleRequest(BaseModel):
    """Request for recommendation rationale."""

    user_id: str = Field(..., description="User ID")
    movie_id: str = Field(..., description="Recommended movie ID")
    user_history: Optional[List[str]] = Field(
        None, description="Previously liked movie IDs"
    )


class RationaleResponse(BaseModel):
    """Response with rationale."""

    user_id: str
    movie_id: str
    rationale: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RateLimitStatus(BaseModel):
    """Rate limit status response."""

    requests_this_minute: int
    tokens_this_minute: int
    requests_today: int
    rpm_remaining: int
    tpm_remaining: int
    daily_remaining: int


# A/B Testing state
_ab_test_config = {
    "prompt_experiment": {
        "enabled": True,
        "variants": ["zero_shot", "few_shot_k3", "meta_prompt"],
        "weights": [0.2, 0.3, 0.5],
    }
}


def get_ab_variant(experiment: str, user_id: str = None) -> str:
    """Get A/B test variant for an experiment."""
    import hashlib
    import random

    config = _ab_test_config.get(experiment)
    if not config or not config["enabled"]:
        return config["variants"][-1] if config else "default"

    # Deterministic assignment based on user_id
    if user_id:
        hash_val = int(hashlib.md5(f"{experiment}:{user_id}".encode()).hexdigest(), 16)
        rand_val = (hash_val % 1000) / 1000
    else:
        rand_val = random.random()

    # Select variant based on weights
    cumulative = 0
    for variant, weight in zip(config["variants"], config["weights"]):
        cumulative += weight
        if rand_val < cumulative:
            ab_test_impressions.labels(experiment=experiment, variant=variant).inc()
            return variant

    return config["variants"][-1]


@router.get("/health", summary="RAG service health check")
async def rag_health():
    """Check RAG service health and component status."""
    status_dict = {
        "status": "healthy",
        "components": {
            "rag_available": RAG_AVAILABLE,
            "llm_available": LLM_AVAILABLE,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    if RAG_AVAILABLE:
        retriever = get_retriever()
        status_dict["components"]["retriever_ready"] = retriever.is_ready()
        status_dict["components"]["document_count"] = (
            len(retriever._ingester.documents) if retriever._ingester else 0
        )

    if LLM_AVAILABLE:
        try:
            # Don't initialize client on health check - just check if it would be available
            # This prevents unnecessary API calls on startup
            from swipeflix.config import settings

            has_api_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
            status_dict["components"]["llm_initialized"] = has_api_key
            # Get rate limit status without initializing client
            limiter = get_rate_limiter()
            status_dict["rate_limit"] = limiter.get_remaining_quota()
        except Exception as e:
            logger.warning(f"Could not get LLM status: {e}")
            status_dict["components"]["llm_initialized"] = False
            status_dict["rate_limit"] = {
                "rpm_remaining": 0,
                "tpm_remaining": 0,
                "daily_remaining": 0,
            }

    return status_dict


@router.get(
    "/rate-limit", response_model=RateLimitStatus, summary="Get rate limit status"
)
async def get_rate_limit_status():
    """Get current Gemini API rate limit status."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM service not available")

    limiter = get_rate_limiter()
    status_info = limiter.get_status()
    remaining = limiter.get_remaining_quota()

    return RateLimitStatus(
        requests_this_minute=status_info["requests_this_minute"],
        tokens_this_minute=status_info["tokens_this_minute"],
        requests_today=status_info["requests_today"],
        rpm_remaining=remaining["rpm_remaining"],
        tpm_remaining=remaining["tpm_remaining"],
        daily_remaining=remaining["daily_remaining"],
    )


@router.post("/query", response_model=RAGQueryResponse, summary="RAG Q&A query")
async def rag_query(request: RAGQueryRequest):
    """Answer a question using RAG retrieval + LLM generation.

    Retrieves relevant movie information and generates a grounded response.
    """
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    # Input validation with guardrails
    input_validator = get_input_validator()
    input_result = input_validator.validate(request.query)

    if not input_result.passed:
        logger.warning(f"Input validation failed: {input_result.violations}")
        raise HTTPException(
            status_code=400,
            detail=f"Input validation failed: {input_result.violations[0].message}",
        )

    # Use sanitized input
    sanitized_query = input_result.sanitized_text or request.query

    try:
        # Get RAG generator
        generator = get_rag_generator()

        # Generate response
        rag_response = generator.generate(
            query=sanitized_query,
            top_k=request.top_k,
        )

        # Output validation
        output_validator = get_output_validator()
        output_result = output_validator.validate(
            rag_response.answer,
            context=rag_response.sources,
        )

        if not output_result.passed:
            logger.warning(f"Output validation failed: {output_result.violations}")
            # Don't fail, but mark as low confidence
            rag_response.confidence = "LOW"

        # Build response
        return RAGQueryResponse(
            query=request.query,
            answer=rag_response.answer,
            sources=rag_response.sources if request.include_sources else [],
            grounded=rag_response.grounded,
            confidence=rag_response.confidence,
            latency_ms=rag_response.latency_ms,
            tokens_used=rag_response.tokens_used,
            cached=rag_response.cached,
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )


@router.post("/blurb", response_model=BlurbResponse, summary="Generate movie blurb")
async def generate_blurb(request: BlurbRequest):
    """Generate a short swipe-card blurb for a movie."""
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    try:
        generator = get_rag_generator()
        blurb = generator.generate_blurb(
            movie_id=request.movie_id,
            tone=request.tone,
            max_length=request.max_length,
        )

        return BlurbResponse(
            movie_id=request.movie_id,
            blurb=blurb,
            tone=request.tone,
            character_count=len(blurb),
        )

    except Exception as e:
        logger.error(f"Blurb generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/summarize-reviews",
    response_model=ReviewSummaryResponse,
    summary="Summarize reviews",
)
async def summarize_reviews(request: ReviewSummaryRequest):
    """Summarize multiple reviews and extract sentiment."""
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    try:
        generator = get_rag_generator()
        result = generator.summarize_reviews(
            movie_id=request.movie_id,
            reviews=request.reviews,
        )

        return ReviewSummaryResponse(
            movie_id=request.movie_id,
            summary=result["summary"],
            sentiment=result["sentiment"],
            review_count=len(request.reviews),
        )

    except Exception as e:
        logger.error(f"Review summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/extract-structured",
    response_model=StructuredDataResponse,
    summary="Extract structured data",
)
async def extract_structured_data(request: StructuredDataRequest):
    """Extract structured movie data from free-form text."""
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    # Input validation
    input_validator = get_input_validator()
    input_result = input_validator.validate(request.text)

    if not input_result.passed:
        raise HTTPException(status_code=400, detail="Input validation failed")

    try:
        generator = get_rag_generator()
        extracted = generator.extract_structured_data(
            text=input_result.sanitized_text or request.text,
        )

        return StructuredDataResponse(extracted_data=extracted)

    except Exception as e:
        logger.error(f"Structured extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rationale",
    response_model=RationaleResponse,
    summary="Generate recommendation rationale",
)
async def generate_rationale(request: RationaleRequest):
    """Generate a brief human-readable rationale for a recommendation."""
    if not RAG_AVAILABLE or not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    try:
        generator = get_rag_generator()
        rationale = generator.generate_rationale(
            user_id=request.user_id,
            movie_id=request.movie_id,
            user_history=request.user_history,
        )

        return RationaleResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            rationale=rationale,
        )

    except Exception as e:
        logger.error(f"Rationale generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", summary="Semantic movie search")
async def semantic_search(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(default=10, ge=1, le=50),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    year_min: Optional[int] = Query(None, ge=1900, le=2030),
    year_max: Optional[int] = Query(None, ge=1900, le=2030),
):
    """Perform semantic search over movie database."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG service not available")

    try:
        retriever = get_retriever()

        # Build filters
        filters = {}
        if genre:
            filters["genre"] = genre
        if year_min:
            filters["year_min"] = year_min
        if year_max:
            filters["year_max"] = year_max

        # Search
        results = retriever.retrieve(
            query, top_k=top_k, filters=filters if filters else None
        )

        return {
            "query": query,
            "results": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "genre": doc.genre,
                    "year": doc.year,
                    "rating": doc.rating,
                    "director": doc.director,
                    "relevance_score": score,
                }
                for doc, score in results
            ],
            "total": len(results),
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-test/{experiment}", summary="Get A/B test variant")
async def get_ab_test_variant(
    experiment: str,
    user_id: Optional[str] = Query(None),
):
    """Get A/B test variant assignment for an experiment."""
    variant = get_ab_variant(experiment, user_id)

    return {
        "experiment": experiment,
        "variant": variant,
        "user_id": user_id,
    }


@router.get("/ab-test-config", summary="Get A/B test configuration")
async def get_ab_test_config():
    """Get current A/B test configuration."""
    return _ab_test_config
