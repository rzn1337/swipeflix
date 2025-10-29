"""Model prediction and loading utilities."""

from typing import List, Optional

import mlflow
from loguru import logger

from swipeflix.api.schemas import MovieRecommendation
from swipeflix.config import settings


class ModelPredictor:
    """Loads and serves predictions from MLflow model."""

    def __init__(self):
        """Initialize predictor."""
        self.model = None
        self.model_version = None
        self._loaded = False

    def load_model(self) -> None:
        """Load model from MLflow."""
        try:
            logger.info("Loading model from MLflow...")
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            # Determine model URI
            if settings.model_uri:
                model_uri = settings.model_uri
            else:
                # Load from registry
                model_uri = f"models:/{settings.model_name}/{settings.model_version}"

            logger.info(f"Loading model from: {model_uri}")

            # Load as PyFunc
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = settings.model_version
            self._loaded = True

            logger.info(
                f"Model loaded successfully: {settings.model_name} v{self.model_version}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # For development, allow running without model
            if settings.environment == "development":
                logger.warning("Running in development mode without model")
                self._loaded = False
            else:
                raise

    def predict(
        self,
        user_id: str,
        top_k: int = 10,
        candidate_movies: Optional[List[str]] = None,
    ) -> List[MovieRecommendation]:
        """Generate recommendations for a user."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare input
        model_input = {"user_id": user_id, "top_k": top_k}

        if candidate_movies:
            model_input["candidate_movies"] = candidate_movies

        # Get predictions
        logger.debug(f"Predicting for user_id={user_id}, top_k={top_k}")
        predictions = self.model.predict(model_input)

        # Convert to schema
        recommendations = []
        for pred in predictions:
            recommendations.append(
                MovieRecommendation(
                    movie_id=pred["movie_id"],
                    title=pred["title"],
                    score=pred["score"],
                    genres=pred.get("genres", []),
                )
            )

        return recommendations

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_version(self) -> Optional[str]:
        """Get current model version."""
        return self.model_version
