"""
Rate limiting dependencies for FastAPI endpoints.
✅ FIX: identifier must be async (fastapi-limiter expects await).
"""

from __future__ import annotations

from fastapi_limiter.depends import RateLimiter
from starlette.requests import Request
from ..core.config import settings

# ───────────────────────── helpers ──────────────────────────
async def _path_aware_ip(request: Request) -> str:
    """
    Return `<ip>:<path>` so each endpoint has its own bucket.
    Cheap & fully-sync, but declared *async* because fastapi-limiter
    always awaits the identifier.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    ip = (forwarded.split(",")[0].strip() if forwarded else request.client.host)
    return f"{ip}:{request.scope['path']}"

async def user_or_ip(request: Request) -> str:
    """
    Prefer JWT → keeps per-user buckets across NAT; otherwise fall back to IP+path.
    Also declared async for compatibility with fastapi-limiter.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return await _path_aware_ip(request)

# ──────────────────────── limiters ──────────────────────────

default_limit = RateLimiter(
    times=settings.RATE_LIMIT_DEFAULT,
    seconds=settings.RATE_LIMIT_WINDOW,
    identifier=user_or_ip,
)

light_limit = RateLimiter(                # /iris/predict
    times=120,
    seconds=settings.RATE_LIMIT_WINDOW_LIGHT,  # Use dedicated light window
    identifier=user_or_ip,                # ← switched to token-based
)

heavy_limit = RateLimiter(                # /cancer/predict
    times=settings.RATE_LIMIT_CANCER,
    seconds=settings.RATE_LIMIT_WINDOW,
    identifier=user_or_ip,
)

login_limit = RateLimiter(                # bad‑login attempts
    # `times` is exclusive – allow three failures, block 4th
    times=settings.RATE_LIMIT_LOGIN + 1,
    seconds=settings.RATE_LIMIT_LOGIN_WINDOW,
    identifier=_path_aware_ip,
)

training_limit = RateLimiter(             # /train endpoints
    times=settings.RATE_LIMIT_TRAINING,
    seconds=settings.RATE_LIMIT_WINDOW * 5,
    identifier=user_or_ip,
)

# Handy handle for debug & CI
def get_redis():
    from fastapi_limiter import FastAPILimiter as _L
    return _L.redis 
