"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///test_mlflow.db"


@pytest.fixture
def sample_movies_data():
    """Create sample movies dataframe for testing."""
    return pd.DataFrame(
        {
            "movieId": ["1", "2", "3", "4", "5"],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
                "Father of the Bride Part II (1995)",
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Children|Fantasy",
                "Comedy|Romance",
                "Comedy|Drama|Romance",
                "Comedy",
            ],
        }
    )


@pytest.fixture
def sample_ratings_data():
    """Create sample ratings dataframe for testing."""
    np.random.seed(42)
    data = []
    for user_id in range(1, 11):
        for movie_id in range(1, 6):
            if np.random.random() > 0.3:  # 70% density
                data.append(
                    {
                        "userId": str(user_id),
                        "movieId": str(movie_id),
                        "rating": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0]),
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_files(tmp_path, sample_movies_data, sample_ratings_data):
    """Create temporary CSV files with sample data."""
    movies_file = tmp_path / "movies.csv"
    ratings_file = tmp_path / "ratings.csv"

    sample_movies_data.to_csv(movies_file, index=False)
    sample_ratings_data.to_csv(ratings_file, index=False)

    return {
        "movies_file": movies_file,
        "ratings_file": ratings_file,
        "data_dir": tmp_path,
    }


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from swipeflix.api.main import app

    return TestClient(app)


@pytest.fixture
def mock_predictor(monkeypatch):
    """Mock ModelPredictor for testing without actual model."""
    from swipeflix.api.schemas import MovieRecommendation
    from swipeflix.ml.predict import ModelPredictor

    class MockPredictor(ModelPredictor):
        def __init__(self):
            super().__init__()
            self._loaded = True
            self.model_version = "test-v1"

        def load_model(self):
            self._loaded = True

        def predict(self, user_id, top_k=10, candidate_movies=None):
            # Return mock recommendations
            recommendations = []
            for i in range(min(top_k, 5)):
                recommendations.append(
                    MovieRecommendation(
                        movie_id=f"movie_{i}",
                        title=f"Test Movie {i}",
                        score=0.9 - (i * 0.1),
                        genres=["Action", "Drama"],
                    )
                )
            return recommendations

        def is_loaded(self):
            return True

        def get_model_version(self):
            return "test-v1"

    def mock_get_predictor():
        return MockPredictor()

    monkeypatch.setattr("swipeflix.api.routes.get_predictor", mock_get_predictor)

    return MockPredictor()


@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Cleanup MLflow test artifacts after each test."""
    yield
    # Cleanup
    test_db = Path("test_mlflow.db")
    if test_db.exists():
        test_db.unlink()
