"""FastAPI application entry point."""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app

from swipeflix.api.middleware import LoggingMiddleware, MetricsMiddleware
from swipeflix.api.routes import router
from swipeflix.config import settings

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"MLflow Tracking URI: {settings.mlflow_tracking_uri}")

    if settings.canary:
        logger.warning("🐤 Running in CANARY mode")
    
    # Initialize AWS CloudWatch logging
    if settings.cloudwatch_enabled:
        from swipeflix.cloud.aws_utils import aws_manager, log_to_cloudwatch
        
        if aws_manager.is_aws_enabled():
            # Create log group and stream
            aws_manager.create_log_group_if_not_exists(settings.cloudwatch_log_group)
            aws_manager.create_log_stream_if_not_exists(
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            
            # Send startup log
            startup_message = (
                f"SwipeFlix API started successfully. "
                f"Version: {settings.app_version}, Environment: {settings.environment}, "
                f"S3 Integration: {'Enabled' if settings.use_aws_s3 else 'Disabled'}, "
                f"MLflow: {settings.mlflow_tracking_uri}"
            )
            
            log_to_cloudwatch(
                startup_message,
                settings.cloudwatch_log_group,
                settings.cloudwatch_log_stream
            )
            
            logger.info("CloudWatch logging initialized and startup message sent")
        else:
            logger.warning("CloudWatch enabled but AWS credentials not available")

    yield

    # Send shutdown log
    if settings.cloudwatch_enabled:
        from swipeflix.cloud.aws_utils import log_to_cloudwatch
        log_to_cloudwatch(
            f"SwipeFlix API shutting down. Environment: {settings.environment}",
            settings.cloudwatch_log_group,
            settings.cloudwatch_log_stream
        )
    
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A hybrid movie recommendation system combining collaborative and content-based filtering",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(MetricsMiddleware)
if settings.debug:
    app.add_middleware(LoggingMiddleware)

# Include routes
app.include_router(router, prefix="")

# Mount Prometheus metrics endpoint
if settings.prometheus_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Prometheus metrics enabled at /metrics")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "swipeflix.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
    )

