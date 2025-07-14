"""
Core configuration settings for the FastAPI application.
Centralizes environment variables and provides sensible defaults.
"""

import os
from typing import Optional

class Settings:
    """Application settings with environment-based configuration."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")

    # Security
    SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # CORS
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

    # MLflow in local-file mode by default
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns_local"
    )
    MLFLOW_REGISTRY_URI: str = os.getenv(
        "MLFLOW_REGISTRY_URI",
        MLFLOW_TRACKING_URI
    )

    # Model training flags
    SKIP_BACKGROUND_TRAINING: bool = os.getenv("SKIP_BACKGROUND_TRAINING", "0") == "1"
    AUTO_TRAIN_MISSING: bool = os.getenv("AUTO_TRAIN_MISSING", "1") == "1"
    UNIT_TESTING: bool = os.getenv("UNIT_TESTING", "0") == "1"

settings = Settings() 
