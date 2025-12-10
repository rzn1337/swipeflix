"""Configuration management for SwipeFlix."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = "SwipeFlix"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Model
    model_name: str = "SwipeFlixModel"
    model_version: str = "1"
    model_stage: str = "Production"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_uri: Optional[str] = None

    # Data paths
    data_dir: Path = Path("data")
    movies_file: str = "movies.csv"
    ratings_file: str = "ratings.csv"

    # Training
    random_seed: int = 42
    test_size: float = 0.2
    n_components: int = 50
    content_weight: float = 0.3
    collab_weight: float = 0.7

    # Monitoring
    prometheus_enabled: bool = True
    evidently_enabled: bool = True
    log_level: str = "INFO"

    # Storage (MinIO/S3)
    s3_endpoint_url: Optional[str] = "http://localhost:9000"
    s3_access_key: Optional[str] = "minioadmin"
    s3_secret_key: Optional[str] = "minioadmin"
    s3_bucket: str = "mlflow"

    # AWS Integration (D9 - Cloud Integration)
    use_aws_s3: bool = False  # Set to True to use AWS S3 instead of local files
    aws_s3_bucket: str = "swipeflix"  # AWS S3 bucket for data storage
    aws_s3_data_prefix: str = ""  # Optional prefix for data files
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None  # AWS Access Key ID
    aws_secret_access_key: Optional[str] = None  # AWS Secret Access Key
    cloudwatch_enabled: bool = False  # Set to True to enable CloudWatch logging
    cloudwatch_log_group: str = "swipeflix-logs"
    cloudwatch_log_stream: str = "app"

    # Canary deployment
    canary: bool = False

    # LLM Settings (Milestone 2)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    llm_cache_enabled: bool = True
    llm_cache_dir: str = ".cache/gemini"
    llm_cache_ttl: int = 86400  # 24 hours

    # RAG Settings
    faiss_index_path: str = "data/faiss_index"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_min_score: float = 0.3

    # Guardrails Settings
    guardrails_enabled: bool = True
    pii_filter_enabled: bool = True
    injection_filter_enabled: bool = True
    toxicity_filter_enabled: bool = True
    toxicity_threshold: float = 0.7

    # Rate Limiting (Gemini Free Tier)
    llm_rpm_limit: int = 5  # 5 Requests Per Minute
    llm_tpm_limit: int = 250000  # 250K Tokens Per Minute
    llm_daily_limit: int = 20  # 20 Requests Per Day

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        protected_namespaces = ()  # Allow model_* field names


# Global settings instance
settings = Settings()
