"""
Concurrency limiting middleware for heavy endpoints.
Provides semaphore-based concurrency control to prevent resource exhaustion.
"""

import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class ConcurrencyLimiter(BaseHTTPMiddleware):
    """
    Middleware that limits concurrent requests to heavy endpoints.

    This is useful for CPU-intensive operations like Bayesian inference
    that could overwhelm the server if too many requests are processed simultaneously.
    """

    def __init__(self, app, max_concurrent: int = 4, heavy_endpoints: set = None):
        super().__init__(app)
        self._sem = asyncio.Semaphore(max_concurrent)
        self.heavy_endpoints = heavy_endpoints or {
            "/api/v1/cancer/predict",
            "/api/v1/iris/train", 
            "/api/v1/cancer/train"
        }
        logger.info(f"Concurrency limiter initialized with max {max_concurrent} concurrent requests")

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with concurrency limiting for heavy endpoints."""
        path = request.url.path

        # Only apply concurrency limiting to heavy endpoints
        if path in self.heavy_endpoints:
            try:
                async with self._sem:
                    logger.debug(f"Processing heavy endpoint {path} with concurrency control")
                    return await call_next(request)
            except asyncio.TimeoutError:
                logger.warning(f"Concurrency timeout for {path}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server is busy processing other requests. Please try again in a moment.",
                    headers={"Retry-After": "30"}
                )
        else:
            # Light endpoints bypass concurrency control
            return await call_next(request) 
