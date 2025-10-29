"""Tests for ML training and prediction modules."""

import pandas as pd

from swipeflix.ml.preprocessing import DataPreprocessor


def test_data_preprocessor_clean_data(sample_movies_data, sample_ratings_data):
    """Test data cleaning."""
    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    movies, ratings = preprocessor.clean_data()

    assert len(movies) > 0
    assert len(ratings) > 0
    assert not movies["id"].isna().any()
    assert not ratings["user_id"].isna().any()


def test_data_preprocessor_user_item_matrix(sample_movies_data, sample_ratings_data):
    """Test user-item matrix creation."""
    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    preprocessor.clean_data()

    matrix = preprocessor.create_user_item_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] > 0  # Has users
    assert matrix.shape[1] > 0  # Has movies
    assert matrix.index.name == "user_id"


def test_data_preprocessor_content_features(sample_movies_data, sample_ratings_data):
    """Test content feature extraction."""
    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    preprocessor.clean_data()

    tfidf_matrix, vectorizer = preprocessor.extract_content_features()

    assert tfidf_matrix.shape[0] == len(sample_movies_data)
    assert tfidf_matrix.shape[1] > 0
    assert vectorizer is not None


def test_data_preprocessor_sampling(sample_movies_data, sample_ratings_data):
    """Test data sampling."""
    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    preprocessor.clean_data()

    # Sample to smaller size
    sample_size = 5
    movies, ratings = preprocessor.sample_data(sample_size)

    # Should have at most sample_size users
    assert len(ratings["user_id"].unique()) <= sample_size


def test_hybrid_model_initialization():
    """Test HybridRecommenderModel initialization."""
    from swipeflix.ml.train import HybridRecommenderModel

    model = HybridRecommenderModel()

    assert model.svd_model is None
    assert model.user_embeddings is None
    assert model.item_embeddings is None


def test_hybrid_model_fit(sample_movies_data, sample_ratings_data):
    """Test model training."""
    from swipeflix.ml.train import HybridRecommenderModel

    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    movies, ratings = preprocessor.clean_data()
    user_item_matrix = preprocessor.create_user_item_matrix()
    tfidf_matrix, vectorizer = preprocessor.extract_content_features()

    model = HybridRecommenderModel()
    model.fit(user_item_matrix, movies, tfidf_matrix, vectorizer)

    assert model.svd_model is not None
    assert model.user_embeddings is not None
    assert model.item_embeddings is not None
    assert model.user_mapping is not None
    assert model.item_mapping is not None


def test_hybrid_model_predict(sample_movies_data, sample_ratings_data):
    """Test model prediction."""
    from swipeflix.ml.train import HybridRecommenderModel

    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    movies, ratings = preprocessor.clean_data()
    user_item_matrix = preprocessor.create_user_item_matrix()
    tfidf_matrix, vectorizer = preprocessor.extract_content_features()

    model = HybridRecommenderModel()
    model.fit(user_item_matrix, movies, tfidf_matrix, vectorizer)

    # Get a valid user_id from the matrix
    user_id = user_item_matrix.index[0]

    # Predict
    model_input = pd.DataFrame([{"user_id": user_id, "top_k": 3}])
    recommendations = model.predict(context=None, model_input=model_input)

    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3

    if len(recommendations) > 0:
        rec = recommendations[0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec
        assert 0.0 <= rec["score"] <= 1.0


def test_hybrid_model_predict_unknown_user(sample_movies_data, sample_ratings_data):
    """Test prediction for unknown user."""
    from swipeflix.ml.train import HybridRecommenderModel

    preprocessor = DataPreprocessor(sample_movies_data, sample_ratings_data)
    movies, ratings = preprocessor.clean_data()
    user_item_matrix = preprocessor.create_user_item_matrix()
    tfidf_matrix, vectorizer = preprocessor.extract_content_features()

    model = HybridRecommenderModel()
    model.fit(user_item_matrix, movies, tfidf_matrix, vectorizer)

    # Unknown user
    model_input = pd.DataFrame([{"user_id": "unknown_user_999", "top_k": 3}])
    recommendations = model.predict(context=None, model_input=model_input)

    # Should return popular movies as fallback
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0


def test_model_predictor_initialization():
    """Test ModelPredictor initialization."""
    from swipeflix.ml.predict import ModelPredictor

    predictor = ModelPredictor()

    assert predictor.model is None
    assert not predictor.is_loaded()
    assert predictor.get_model_version() is None


def test_data_cleaning_removes_duplicates():
    """Test that data cleaning removes duplicates."""
    movies_df = pd.DataFrame(
        {
            "id": ["1", "1", "2"],
            "title": ["Movie 1", "Movie 1 Dup", "Movie 2"],
            "genre": ["Action", "Action", "Comedy"],
            "plot": ["Plot 1", "Plot 1 Dup", "Plot 2"],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": ["1", "1", "2"],
            "movie_id": ["1", "1", "2"],
            "rating": [5.0, 5.0, 4.0],
        }
    )

    preprocessor = DataPreprocessor(movies_df, ratings_df)
    movies, ratings = preprocessor.clean_data()

    # Should remove duplicate movies and ratings
    assert len(movies) == 2
    assert len(ratings) == 2


def test_data_cleaning_handles_missing_values():
    """Test that data cleaning handles missing values."""
    movies_df = pd.DataFrame(
        {
            "id": ["1", "2", None],
            "title": ["Movie 1", None, "Movie 3"],
            "genre": ["Action", "Comedy", "Drama"],
            "plot": ["Plot 1", "Plot 2", "Plot 3"],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": ["1", "2", "3"],
            "movie_id": ["1", None, "3"],
            "rating": [5.0, 4.0, None],
        }
    )

    preprocessor = DataPreprocessor(movies_df, ratings_df)
    movies, ratings = preprocessor.clean_data()

    # Should remove rows with missing critical values
    assert not movies["id"].isna().any()
    assert not movies["title"].isna().any()
    assert not ratings["user_id"].isna().any()
    assert not ratings["movie_id"].isna().any()
    assert not ratings["rating"].isna().any()
