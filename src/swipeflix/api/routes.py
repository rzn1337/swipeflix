"""API route handlers."""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from swipeflix.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
)
from swipeflix.config import settings
from swipeflix.ml.predict import ModelPredictor
from swipeflix.monitoring.metrics import inference_duration_seconds, model_version_info
from swipeflix.monitoring.gpu_metrics import update_gpu_metrics, get_gpu_info

router = APIRouter()

# Global model predictor (lazy loaded)
_predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """Get or initialize model predictor."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing model predictor...")
        _predictor = ModelPredictor()
        _predictor.load_model()
        logger.info("Model predictor initialized successfully")
    return _predictor


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health() -> HealthResponse:
    """Check service health and model status."""
    try:
        # Update GPU metrics if available
        update_gpu_metrics()
        
        predictor = get_predictor()
        model_loaded = predictor.is_loaded()
        model_ver = predictor.get_model_version() if model_loaded else None

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            model_loaded=model_loaded,
            model_version=model_ver,
            environment=settings.environment,
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    status_code=status.HTTP_200_OK,
    tags=["Metadata"],
    summary="Get service metadata",
)
async def metadata() -> MetadataResponse:
    """Get service and model metadata."""
    try:
        predictor = get_predictor()

        return MetadataResponse(
            app_name=settings.app_name,
            version=settings.app_version,
            model_name=settings.model_name,
            model_version=predictor.get_model_version() or "unknown",
            features=[
                "hybrid_recommender",
                "collaborative_filtering",
                "content_based_filtering",
                "mlflow_integration",
                "prometheus_metrics",
            ],
        )
    except Exception as e:
        logger.error(f"Failed to get metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metadata: {str(e)}",
        )


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Get movie recommendations",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(request: PredictRequest) -> PredictResponse:
    """Get personalized movie recommendations for a user."""
    start_time = time.time()

    try:
        logger.info(
            f"Prediction request for user_id={request.user_id}, top_k={request.top_k}"
        )

        predictor = get_predictor()

        # Get recommendations
        recommendations = predictor.predict(
            user_id=request.user_id,
            top_k=request.top_k,
            candidate_movies=request.candidate_movies,
        )

        inference_time_ms = (time.time() - start_time) * 1000

        # Record metrics
        inference_duration_seconds.observe(inference_time_ms / 1000)
        model_version_info.labels(version=predictor.get_model_version()).set(1)

        logger.info(
            f"Prediction completed for user_id={request.user_id} "
            f"in {inference_time_ms:.2f}ms"
        )
        
        # Log to CloudWatch
        if settings.cloudwatch_enabled:
            from swipeflix.cloud.aws_utils import log_to_cloudwatch, send_metric
            log_to_cloudwatch(
                f"Prediction requested: user_id={request.user_id}, top_k={request.top_k}, "
                f"inference_time_ms={inference_time_ms:.2f}, model_version={predictor.get_model_version()}",
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            
            # Send metrics to CloudWatch
            send_metric(
                metric_name="PredictionRequests",
                value=1,
                unit="Count",
                dimensions={"Endpoint": "/predict"}
            )
            send_metric(
                metric_name="InferenceLatency",
                value=inference_time_ms,
                unit="Milliseconds",
                dimensions={"Model": predictor.get_model_version() or "unknown"}
            )

        return PredictResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version=predictor.get_model_version() or "unknown",
            inference_time_ms=inference_time_ms,
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        logger.warning(f"Invalid prediction request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )

