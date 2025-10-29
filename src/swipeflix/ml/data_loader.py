"""Data loading utilities with AWS S3 support."""

from pathlib import Path

import pandas as pd
from loguru import logger

from swipeflix.cloud.aws_utils import aws_manager, log_to_cloudwatch
from swipeflix.config import settings


def load_movies_data() -> pd.DataFrame:
    """
    Load movies data from AWS S3 or local file.
    
    Returns:
        DataFrame with movies data
    """
    if settings.use_aws_s3 and aws_manager.is_aws_enabled():
        # Load from AWS S3
        try:
            logger.info(f"Loading movies data from AWS S3: s3://{settings.aws_s3_bucket}/movies.csv")
            
            key = f"{settings.aws_s3_data_prefix}movies.csv" if settings.aws_s3_data_prefix else "movies.csv"
            df = aws_manager.load_csv_from_s3(settings.aws_s3_bucket, key)
            
            # Log to CloudWatch
            log_to_cloudwatch(
                f"Successfully loaded {len(df)} movies from S3: s3://{settings.aws_s3_bucket}/{key}",
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            
            logger.info(f"Loaded {len(df)} movies from AWS S3")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from S3, falling back to local: {e}")
            log_to_cloudwatch(
                f"ERROR: Failed to load movies from S3: {str(e)}. Falling back to local.",
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            # Fall through to local loading
    
    # Load from local file
    movies_path = settings.data_dir / settings.movies_file
    logger.info(f"Loading movies data from local file: {movies_path}")
    
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file not found: {movies_path}")
    
    df = pd.read_csv(movies_path)
    logger.info(f"Loaded {len(df)} movies from local file")
    return df


def load_ratings_data() -> pd.DataFrame:
    """
    Load ratings data from AWS S3 or local file.
    
    Returns:
        DataFrame with ratings data
    """
    if settings.use_aws_s3 and aws_manager.is_aws_enabled():
        # Load from AWS S3
        try:
            logger.info(f"Loading ratings data from AWS S3: s3://{settings.aws_s3_bucket}/ratings.csv")
            
            key = f"{settings.aws_s3_data_prefix}ratings.csv" if settings.aws_s3_data_prefix else "ratings.csv"
            df = aws_manager.load_csv_from_s3(settings.aws_s3_bucket, key)
            
            # Log to CloudWatch
            log_to_cloudwatch(
                f"Successfully loaded {len(df)} ratings from S3: s3://{settings.aws_s3_bucket}/{key}",
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            
            logger.info(f"Loaded {len(df)} ratings from AWS S3")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from S3, falling back to local: {e}")
            log_to_cloudwatch(
                f"ERROR: Failed to load ratings from S3: {str(e)}. Falling back to local.",
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            # Fall through to local loading
    
    # Load from local file
    ratings_path = settings.data_dir / settings.ratings_file
    logger.info(f"Loading ratings data from local file: {ratings_path}")
    
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    
    df = pd.read_csv(ratings_path)
    logger.info(f"Loaded {len(df)} ratings from local file")
    return df


def check_data_availability() -> dict:
    """
    Check data availability in S3 and local.
    
    Returns:
        Dictionary with availability status
    """
    status = {
        "s3_available": False,
        "s3_movies": False,
        "s3_ratings": False,
        "local_movies": False,
        "local_ratings": False,
    }
    
    # Check S3
    if settings.use_aws_s3 and aws_manager.is_aws_enabled():
        status["s3_available"] = True
        
        movies_key = f"{settings.aws_s3_data_prefix}movies.csv" if settings.aws_s3_data_prefix else "movies.csv"
        ratings_key = f"{settings.aws_s3_data_prefix}ratings.csv" if settings.aws_s3_data_prefix else "ratings.csv"
        
        status["s3_movies"] = aws_manager.check_s3_file_exists(settings.aws_s3_bucket, movies_key)
        status["s3_ratings"] = aws_manager.check_s3_file_exists(settings.aws_s3_bucket, ratings_key)
    
    # Check local
    movies_path = settings.data_dir / settings.movies_file
    ratings_path = settings.data_dir / settings.ratings_file
    
    status["local_movies"] = movies_path.exists()
    status["local_ratings"] = ratings_path.exists()
    
    return status

