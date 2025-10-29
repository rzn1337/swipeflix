"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_version: Optional[str] = Field(None, description="Current model version")
    environment: str = Field(default="development")


class MetadataResponse(BaseModel):
    """Service metadata response schema."""

    app_name: str
    version: str
    model_name: str
    model_version: str
    features: List[str]


class MovieRecommendation(BaseModel):
    """Single movie recommendation."""

    movie_id: str = Field(..., description="Movie identifier")
    title: str = Field(..., description="Movie title")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    genres: Optional[List[str]] = Field(None, description="Movie genres")


class PredictRequest(BaseModel):
    """Prediction request schema."""

    user_id: str = Field(..., description="User identifier (e.g., 'user_123')")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Number of recommendations to return"
    )
    candidate_movies: Optional[List[str]] = Field(
        None, description="Optional list of candidate movie IDs to rank"
    )

    @validator("user_id")
    def validate_user_id(cls, v):
        """Validate user_id format."""
        if not v or not isinstance(v, str):
            raise ValueError("user_id must be a non-empty string")
        return v


class PredictResponse(BaseModel):
    """Prediction response schema."""

    user_id: str = Field(..., description="User identifier")
    recommendations: List[MovieRecommendation] = Field(
        ..., description="List of movie recommendations"
    )
    model_version: str = Field(..., description="Model version used for inference")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

