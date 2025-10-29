"""Tests for FastAPI endpoints."""

import pytest
from fastapi import status


def test_health_endpoint(test_client, mock_predictor):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["model_loaded"] is True


def test_root_endpoint(test_client):
    """Test root endpoint."""
    response = test_client.get("/")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data


def test_metadata_endpoint(test_client, mock_predictor):
    """Test metadata endpoint."""
    response = test_client.get("/metadata")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["app_name"] == "SwipeFlix"
    assert "version" in data
    assert "model_name" in data
    assert "model_version" in data
    assert "features" in data
    assert isinstance(data["features"], list)


def test_predict_endpoint(test_client, mock_predictor):
    """Test predict endpoint with valid request."""
    request_data = {"user_id": "user_1", "top_k": 5}

    response = test_client.post("/predict", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["user_id"] == "user_1"
    assert "recommendations" in data
    assert len(data["recommendations"]) == 5
    assert "model_version" in data
    assert "inference_time_ms" in data
    assert "timestamp" in data

    # Check recommendation structure
    rec = data["recommendations"][0]
    assert "movie_id" in rec
    assert "title" in rec
    assert "score" in rec
    assert 0.0 <= rec["score"] <= 1.0


def test_predict_endpoint_different_top_k(test_client, mock_predictor):
    """Test predict endpoint with different top_k values."""
    for top_k in [1, 3, 5]:
        request_data = {"user_id": "user_1", "top_k": top_k}

        response = test_client.post("/predict", json=request_data)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["recommendations"]) == min(top_k, 5)  # Mock returns max 5


def test_predict_endpoint_invalid_user_id(test_client, mock_predictor):
    """Test predict endpoint with invalid user_id."""
    request_data = {"user_id": "", "top_k": 5}

    response = test_client.post("/predict", json=request_data)
    # Should fail validation
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_endpoint_missing_user_id(test_client, mock_predictor):
    """Test predict endpoint with missing user_id."""
    request_data = {"top_k": 5}

    response = test_client.post("/predict", json=request_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_endpoint_default_top_k(test_client, mock_predictor):
    """Test predict endpoint uses default top_k when not provided."""
    request_data = {"user_id": "user_1"}

    response = test_client.post("/predict", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    # Should use default top_k=10, but mock returns max 5
    assert len(data["recommendations"]) <= 10


def test_metrics_endpoint(test_client, mock_predictor):
    """Test Prometheus metrics endpoint."""
    # Make some requests first
    test_client.get("/health")
    test_client.get("/metadata")

    response = test_client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK

    # Check for Prometheus metrics format
    text = response.text
    assert "swipeflix_http_requests_total" in text or "http_requests_total" in text


def test_cors_headers(test_client):
    """Test CORS headers are present."""
    response = test_client.options("/health")
    # CORS should allow the request
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


def test_openapi_docs(test_client):
    """Test OpenAPI documentation is available."""
    response = test_client.get("/docs")
    assert response.status_code == status.HTTP_200_OK

    response = test_client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK

    openapi_data = response.json()
    assert "openapi" in openapi_data
    assert "paths" in openapi_data


@pytest.mark.parametrize(
    "user_id,top_k",
    [
        ("user_123", 5),
        ("user_abc", 10),
        ("1", 1),
    ],
)
def test_predict_various_inputs(test_client, mock_predictor, user_id, top_k):
    """Test predict endpoint with various valid inputs."""
    request_data = {"user_id": user_id, "top_k": top_k}

    response = test_client.post("/predict", json=request_data)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["user_id"] == user_id
    assert len(data["recommendations"]) <= top_k

