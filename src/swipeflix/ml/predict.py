"""Model prediction and loading utilities."""

from pathlib import Path
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
        """Load model from local file or MLflow."""
        # Try loading from local file first
        local_model_path = Path("models/local_model.pkl")
        if local_model_path.exists():
            try:
                logger.info(f"Loading model from local file: {local_model_path}")
                import pickle

                # Import the class before unpickling to avoid "Can't get attribute" error
                # This is needed because pickle saved the class from __main__ context
                try:
                    from swipeflix.ml.train_local import HybridRecommenderModel
                except ImportError:
                    # Fallback: try to import from train module
                    try:
                        from swipeflix.ml.train import HybridRecommenderModel
                    except ImportError:
                        logger.error(
                            "Could not import HybridRecommenderModel from either train_local or train"
                        )
                        raise

                # Create a custom unpickler that maps __main__ to the actual module
                class ModelUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # If pickle is looking for the class in __main__, redirect to the actual module
                        if module == "__main__" and name == "HybridRecommenderModel":
                            return HybridRecommenderModel
                        # Otherwise, use the standard lookup
                        return super().find_class(module, name)

                with open(local_model_path, "rb") as f:
                    unpickler = ModelUnpickler(f)
                    self.model = unpickler.load()

                # Load metadata if available
                metadata_path = local_model_path.parent / "model_metadata.pkl"
                if metadata_path.exists():
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                        self.model_version = metadata.get("model_version", "local")
                else:
                    self.model_version = "local"

                self._loaded = True
                logger.info(
                    f"Model loaded successfully from local file: {self.model_version}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}. Trying MLflow...")

        # Fall back to MLflow
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
            # In development mode, return mock recommendations
            if settings.environment == "development":
                logger.warning(
                    "Model not loaded. Returning mock recommendations for development."
                )
                return self._get_mock_recommendations(user_id, top_k)
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Prepare input
        model_input = {"user_id": user_id, "top_k": top_k}

        if candidate_movies:
            model_input["candidate_movies"] = candidate_movies

        # Get predictions
        logger.debug(f"Predicting for user_id={user_id}, top_k={top_k}")

        # Handle both MLflow PyFunc model and local model
        if hasattr(self.model, "predict") and not isinstance(self.model, type):
            # Local model (direct call)
            predictions = self.model.predict(user_id=user_id, top_k=top_k)
        else:
            # MLflow PyFunc model (needs dict input)
            model_input = {"user_id": user_id, "top_k": top_k}
            if candidate_movies:
                model_input["candidate_movies"] = candidate_movies
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

    def _get_mock_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> List[MovieRecommendation]:
        """Generate mock recommendations for development/testing."""
        from pathlib import Path

        import pandas as pd

        # Try to load movies from CSV
        movies_path = Path(settings.data_dir) / settings.movies_file
        try:
            movies_df = pd.read_csv(movies_path)
            # Return top movies by rating or popularity
            if "rating" in movies_df.columns:
                top_movies = movies_df.nlargest(top_k, "rating")
            else:
                top_movies = movies_df.head(top_k)

            recommendations = []
            for _, row in top_movies.iterrows():
                recommendations.append(
                    MovieRecommendation(
                        movie_id=str(row.get("movieId", row.get("id", ""))),
                        title=str(row.get("title", "Unknown")),
                        score=float(row.get("rating", 4.0)) / 5.0,  # Normalize to 0-1
                        genres=row.get("genres", "").split("|")
                        if pd.notna(row.get("genres"))
                        else [],
                    )
                )
            return recommendations
        except Exception as e:
            logger.warning(f"Could not load movies for mock recommendations: {e}")
            # Return minimal mock data
            return [
                MovieRecommendation(
                    movie_id=f"mock_{i}",
                    title=f"Mock Movie {i+1}",
                    score=0.8 - (i * 0.1),
                    genres=["Action", "Drama"],
                )
                for i in range(min(top_k, 5))
            ]
