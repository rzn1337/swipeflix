"""Local model training script - saves model to disk without MLflow."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from swipeflix.config import settings
from swipeflix.ml.preprocessing import DataPreprocessor


class HybridRecommenderModel:
    """Hybrid recommender combining collaborative and content-based filtering."""

    def __init__(self):
        """Initialize model."""
        self.svd_model = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.content_similarity = None
        self.vectorizer = None
        self.movies_df = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_item_mapping = None
        self.collab_weight = settings.collab_weight
        self.content_weight = settings.content_weight

    def fit(
        self,
        user_item_matrix: pd.DataFrame,
        movies_df: pd.DataFrame,
        tfidf_matrix,
        vectorizer,
    ):
        """Train the hybrid model."""
        logger.info("Training hybrid recommender model...")

        # Store movie data and vectorizer
        self.movies_df = movies_df
        self.vectorizer = vectorizer

        # Create user/item mappings
        self.user_mapping = {
            user: idx for idx, user in enumerate(user_item_matrix.index)
        }
        self.item_mapping = {
            item: idx for idx, item in enumerate(user_item_matrix.columns)
        }
        self.reverse_item_mapping = {
            idx: item for item, idx in self.item_mapping.items()
        }

        # Collaborative filtering: SVD
        logger.info("Training collaborative filtering (SVD)...")
        self.svd_model = TruncatedSVD(
            n_components=min(settings.n_components, min(user_item_matrix.shape) - 1),
            random_state=settings.random_seed,
        )
        self.user_embeddings = self.svd_model.fit_transform(user_item_matrix.values)
        self.item_embeddings = self.svd_model.components_.T

        logger.info(f"User embeddings shape: {self.user_embeddings.shape}")
        logger.info(f"Item embeddings shape: {self.item_embeddings.shape}")

        # Content-based filtering: TF-IDF similarity
        logger.info("Computing content similarity...")
        self.content_similarity = cosine_similarity(tfidf_matrix)

        logger.info("Model training completed!")

    def predict(
        self,
        user_id: str,
        top_k: int = 10,
        candidate_movies=None,
    ):
        """Generate recommendations."""
        # Get user index
        if user_id not in self.user_mapping:
            # New user - use popularity-based recommendations
            logger.warning(
                f"Unknown user {user_id}, using popularity-based recommendations"
            )
            return self._get_popular_movies(top_k)

        user_idx = self.user_mapping[user_id]

        # Collaborative scores
        collab_scores = np.dot(self.user_embeddings[user_idx], self.item_embeddings.T)

        # Normalize scores
        collab_scores = (collab_scores - collab_scores.min()) / (
            collab_scores.max() - collab_scores.min() + 1e-10
        )

        # Content scores (average similarity to user's top items)
        content_scores = self._get_content_scores(user_idx)

        # Hybrid score
        hybrid_scores = (
            self.collab_weight * collab_scores + self.content_weight * content_scores
        )

        # Get top-k recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            movie_id = self.reverse_item_mapping[idx]
            movie_data = self.movies_df[self.movies_df["id"] == movie_id]

            if not movie_data.empty:
                recommendations.append(
                    {
                        "movie_id": movie_id,
                        "title": movie_data.iloc[0]["title"],
                        "score": float(hybrid_scores[idx]),
                        "genres": movie_data.iloc[0].get("genre", "").split("|")
                        if pd.notna(movie_data.iloc[0].get("genre"))
                        else [],
                    }
                )

        return recommendations

    def _get_content_scores(self, user_idx: int) -> np.ndarray:
        """Get content-based scores for a user."""
        # Get user's top-rated items
        user_ratings = self.user_embeddings[user_idx]
        top_item_indices = np.argsort(user_ratings)[::-1][:5]

        # Average similarity to top items
        content_scores = np.zeros(len(self.item_mapping))
        for item_idx in top_item_indices:
            if item_idx < self.content_similarity.shape[0]:
                content_scores += self.content_similarity[item_idx]

        content_scores /= len(top_item_indices)

        return content_scores

    def _get_popular_movies(self, top_k: int) -> list:
        """Get popular movies as fallback."""
        popular_movies = self.movies_df.head(top_k)
        recommendations = []

        for _, movie in popular_movies.iterrows():
            recommendations.append(
                {
                    "movie_id": str(movie["id"]),
                    "title": movie["title"],
                    "score": 0.5,
                    "genres": movie.get("genre", "").split("|")
                    if pd.notna(movie.get("genre"))
                    else [],
                }
            )

        return recommendations


def train_model_local(
    sample_size: int = None,
    seed: int = 42,
    model_save_path: str = "models/local_model.pkl",
) -> None:
    """Train and save model locally."""
    # Set seeds for reproducibility
    np.random.seed(seed)

    # Load data
    logger.info("Loading data...")
    from swipeflix.ml.data_loader import load_movies_data, load_ratings_data

    movies_df = load_movies_data()
    ratings_df = load_ratings_data()

    logger.info(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")

    # Preprocess
    preprocessor = DataPreprocessor(movies_df, ratings_df)

    if sample_size:
        movies_df, ratings_df = preprocessor.sample_data(sample_size)
        logger.info(f"Sampled to {sample_size} records")

    movies_df, ratings_df = preprocessor.clean_data()
    user_item_matrix = preprocessor.create_user_item_matrix()
    tfidf_matrix, vectorizer = preprocessor.extract_content_features()

    logger.info(f"User-item matrix shape: {user_item_matrix.shape}")

    # Train-test split
    train_matrix, test_matrix = train_test_split(
        user_item_matrix, test_size=settings.test_size, random_state=seed
    )

    # Train model
    logger.info("Training model...")
    model = HybridRecommenderModel()
    model.fit(train_matrix, movies_df, tfidf_matrix, vectorizer)

    # Evaluate
    logger.info("Evaluating model...")
    train_mse = np.mean(
        (
            train_matrix.values
            - np.dot(
                model.user_embeddings[: len(train_matrix)], model.item_embeddings.T
            )
        )
        ** 2
    )
    logger.info(f"Training MSE: {train_mse:.4f}")

    # Save model locally
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        "model_version": "1",
        "train_mse": float(train_mse),
        "n_movies": len(movies_df),
        "n_users": len(user_item_matrix),
        "n_components": settings.n_components,
        "collab_weight": settings.collab_weight,
        "content_weight": settings.content_weight,
        "sample_size": sample_size or "full",
        "seed": seed,
    }

    metadata_path = save_path.parent / "model_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"Model saved successfully to: {save_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Training completed! MSE: {train_mse:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train SwipeFlix recommender model locally"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for fast training (None for full dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/local_model.pkl",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    logger.info("Starting SwipeFlix local model training...")
    logger.info(f"Sample size: {args.sample_size or 'full'}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Model save path: {args.model_path}")

    train_model_local(
        sample_size=args.sample_size, seed=args.seed, model_save_path=args.model_path
    )


if __name__ == "__main__":
    main()
