"""
Core configuration settings for the FastAPI application.
Centralizes environment variables and provides sensible defaults.
"""

import os
import socket
import time
from typing import Optional

def _maybe_fallback(uri: str) -> str:
    """Fall back to local file store if HTTP host is unreachable."""
    if uri.startswith("http"):
        host = uri.split("//", 1)[1].split("/", 1)[0].split(":")[0]
        try:
            t0 = time.perf_counter()
            socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            # resolved in < 50 ms → keep it
            if (time.perf_counter() - t0) < 0.05:
                return uri
        except socket.gaierror:
            pass   # unreachable
        return "file:./mlruns_local"
    return uri or "file:./mlruns_local"

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

    # MLflow Configuration - with smart fallback
    MLFLOW_TRACKING_URI: str = _maybe_fallback(os.getenv("MLFLOW_TRACKING_URI", ""))
    MLFLOW_REGISTRY_URI: str = _maybe_fallback(os.getenv("MLFLOW_REGISTRY_URI", ""))

    # MLflow Garbage Collection
    RETAIN_RUNS_PER_MODEL: int = int(os.getenv("RETAIN_RUNS_PER_MODEL", "5"))
    MLFLOW_GC_AFTER_TRAIN: bool = os.getenv("MLFLOW_GC_AFTER_TRAIN", "1") == "1"

    # Model Training Flags
    SKIP_BACKGROUND_TRAINING: bool = os.getenv("SKIP_BACKGROUND_TRAINING", "0") == "1"
    AUTO_TRAIN_MISSING: bool = os.getenv("AUTO_TRAIN_MISSING", "1") == "1"
    UNIT_TESTING: bool = os.getenv("UNIT_TESTING", "0") == "1"

    # Debug flags
    DEBUG_RATELIMIT: bool = os.getenv("DEBUG_RATELIMIT", "0") == "1"

settings = Settings() 
