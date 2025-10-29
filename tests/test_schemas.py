"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from swipeflix.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    MovieRecommendation,
    PredictRequest,
    PredictResponse,
)


def test_health_response_schema():
    """Test HealthResponse schema."""
    data = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "model_loaded": True,
        "model_version": "v1",
        "environment": "test",
    }

    response = HealthResponse(**data)

    assert response.status == "healthy"
    assert response.model_loaded is True
    assert response.model_version == "v1"


def test_movie_recommendation_schema():
    """Test MovieRecommendation schema."""
    data = {
        "movie_id": "123",
        "title": "Test Movie",
        "score": 0.85,
        "genres": ["Action", "Drama"],
    }

    rec = MovieRecommendation(**data)

    assert rec.movie_id == "123"
    assert rec.title == "Test Movie"
    assert rec.score == 0.85
    assert rec.genres == ["Action", "Drama"]


def test_movie_recommendation_score_validation():
    """Test MovieRecommendation score validation."""
    # Valid score
    rec = MovieRecommendation(movie_id="1", title="Movie", score=0.5)
    assert rec.score == 0.5

    # Invalid score (too high)
    with pytest.raises(ValidationError):
        MovieRecommendation(movie_id="1", title="Movie", score=1.5)

    # Invalid score (negative)
    with pytest.raises(ValidationError):
        MovieRecommendation(movie_id="1", title="Movie", score=-0.1)


def test_predict_request_schema():
    """Test PredictRequest schema."""
    data = {"user_id": "user_123", "top_k": 10}

    request = PredictRequest(**data)

    assert request.user_id == "user_123"
    assert request.top_k == 10
    assert request.candidate_movies is None


def test_predict_request_default_top_k():
    """Test PredictRequest default top_k."""
    data = {"user_id": "user_123"}

    request = PredictRequest(**data)

    assert request.top_k == 10  # Default value


def test_predict_request_validation():
    """Test PredictRequest validation."""
    # Empty user_id should fail
    with pytest.raises(ValidationError):
        PredictRequest(user_id="", top_k=10)

    # Missing user_id should fail
    with pytest.raises(ValidationError):
        PredictRequest(top_k=10)

    # Invalid top_k (too low)
    with pytest.raises(ValidationError):
        PredictRequest(user_id="user_1", top_k=0)

    # Invalid top_k (too high)
    with pytest.raises(ValidationError):
        PredictRequest(user_id="user_1", top_k=101)


def test_predict_response_schema():
    """Test PredictResponse schema."""
    data = {
        "user_id": "user_123",
        "recommendations": [
            {
                "movie_id": "1",
                "title": "Movie 1",
                "score": 0.9,
                "genres": ["Action"],
            },
            {
                "movie_id": "2",
                "title": "Movie 2",
                "score": 0.8,
                "genres": ["Drama"],
            },
        ],
        "model_version": "v1",
        "inference_time_ms": 123.45,
        "timestamp": datetime.utcnow(),
    }

    response = PredictResponse(**data)

    assert response.user_id == "user_123"
    assert len(response.recommendations) == 2
    assert response.model_version == "v1"
    assert response.inference_time_ms == 123.45


def test_metadata_response_schema():
    """Test MetadataResponse schema."""
    data = {
        "app_name": "SwipeFlix",
        "version": "1.0.0",
        "model_name": "SwipeFlixModel",
        "model_version": "v1",
        "features": ["feature1", "feature2"],
    }

    response = MetadataResponse(**data)

    assert response.app_name == "SwipeFlix"
    assert response.version == "1.0.0"
    assert len(response.features) == 2


def test_error_response_schema():
    """Test ErrorResponse schema."""
    data = {
        "error": "Test error",
        "detail": "Detailed error message",
        "timestamp": datetime.utcnow(),
    }

    response = ErrorResponse(**data)

    assert response.error == "Test error"
    assert response.detail == "Detailed error message"
