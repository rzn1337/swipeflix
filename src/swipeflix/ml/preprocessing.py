"""Data preprocessing utilities."""

import os
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

from swipeflix.cloud.aws_utils import aws_manager
from swipeflix.config import settings


class DataPreprocessor:
    """Preprocess movie and ratings data."""

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Initialize with dataframes."""
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def clean_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Clean and validate data."""
        logger.info("Cleaning data...")

        # Remove duplicates
        self.movies_df = self.movies_df.drop_duplicates(subset=["id"])
        self.ratings_df = self.ratings_df.drop_duplicates(
            subset=["user_id", "movie_id"]
        )

        # Handle missing values
        self.movies_df = self.movies_df.dropna(subset=["id", "title"])
        self.ratings_df = self.ratings_df.dropna(subset=["user_id", "movie_id", "rating"])

        # Ensure proper data types
        self.movies_df["id"] = self.movies_df["id"].astype(str)
        self.ratings_df["movie_id"] = self.ratings_df["movie_id"].astype(str)
        self.ratings_df["user_id"] = self.ratings_df["user_id"].astype(str)

        logger.info(f"Movies: {len(self.movies_df)}, Ratings: {len(self.ratings_df)}")

        return self.movies_df, self.ratings_df

    def create_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix."""
        logger.info("Creating user-item matrix...")

        # Pivot to create matrix
        user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

        logger.info(
            f"User-item matrix shape: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} movies"
        )

        return user_item_matrix

    def extract_content_features(self) -> tuple[pd.DataFrame, TfidfVectorizer]:
        """Extract TF-IDF features from movie metadata."""
        logger.info("Extracting content features...")

        # Combine text features (use .fillna("") to handle missing values)
        self.movies_df["content"] = (
            self.movies_df["genre"].fillna("")
            + " "
            + self.movies_df["plot"].fillna("")
            + " "
            + self.movies_df["title"].fillna("")
        )

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(self.movies_df["content"].fillna(""))

        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        return tfidf_matrix, vectorizer

    def sample_data(self, sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sample data for faster training."""
        logger.info(f"Sampling {sample_size} users...")

        # Sample users
        unique_users = self.ratings_df["user_id"].unique()
        if len(unique_users) > sample_size:
            sampled_users = pd.Series(unique_users).sample(
                n=sample_size, random_state=42
            )
            self.ratings_df = self.ratings_df[
                self.ratings_df["user_id"].isin(sampled_users)
            ]

        # Keep only movies that have ratings in sampled data
        rated_movies = self.ratings_df["movie_id"].unique()
        self.movies_df = self.movies_df[self.movies_df["id"].isin(rated_movies)]

        logger.info(
            f"Sampled data: {len(self.movies_df)} movies, {len(self.ratings_df)} ratings"
        )

        return self.movies_df, self.ratings_df

