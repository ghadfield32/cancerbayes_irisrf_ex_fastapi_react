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

    # Rate Limiting
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    RATE_LIMIT_DEFAULT: int = int(os.getenv("RATE_LIMIT_DEFAULT", "60"))
    RATE_LIMIT_CANCER: int = int(os.getenv("RATE_LIMIT_CANCER", "30"))
    RATE_LIMIT_LOGIN: int = int(os.getenv("RATE_LIMIT_LOGIN", "3"))
    RATE_LIMIT_TRAINING: int = int(os.getenv("RATE_LIMIT_TRAINING", "2"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    RATE_LIMIT_WINDOW_LIGHT: int = int(os.getenv("RATE_LIMIT_WINDOW_LIGHT", "300"))  # 5 minutes for light endpoint
    RATE_LIMIT_LOGIN_WINDOW: int = int(os.getenv("RATE_LIMIT_LOGIN_WINDOW", "20"))  # seconds

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

    # Debug flags
    DEBUG_RATELIMIT: bool = os.getenv("DEBUG_RATELIMIT", "0") == "1"

settings = Settings() 
