import logging
import os
import asyncio
from fastapi import FastAPI, Request, Depends, BackgroundTasks, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
import time

from pydantic import BaseModel

# ── NEW: Fix ML backend configuration before any JAX imports ───────────────────────────
from .utils.env_sanitizer import fix_ml_backends
fix_ml_backends()
# ──────────────────────────────────────────────────────────────────────────

# ── NEW: Rate limiting imports ─────────────────────────────────────────────────────────
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
# ────────────────────────────────────────────────────────────────────────────────────────

# ── NEW: Concurrency limiting imports ────────────────────────────────────────────────
from .middleware.concurrency import ConcurrencyLimiter
# ────────────────────────────────────────────────────────────────────────────────────────

from .db import lifespan, get_db, get_app_ready
from .security import create_access_token, get_current_user, verify_password
from .crud import get_user_by_username
from .schemas.iris import IrisPredictRequest, IrisPredictResponse, IrisFeatures
from .schemas.cancer import CancerPredictRequest, CancerPredictResponse, CancerFeatures
from .services.ml.model_service import model_service
from .core.config import settings
from .deps.limits import default_limit, heavy_limit, login_limit, training_limit, light_limit
from .security import LoginPayload, get_credentials

# ── NEW: guarantee log directory exists ───────────────────────────
os.makedirs("logs", exist_ok=True)
# ──────────────────────────────────────────────────────────────────

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class Payload(BaseModel):
    count: int

class PredictionRequest(BaseModel):
    data: Payload

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    input_received: Payload  # Echo back the input for verification

class Token(BaseModel):
    access_token: str
    token_type: str

app = FastAPI(
    title="FastAPI + React ML App",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    swagger_ui_parameters={"persistAuthorization": True},
    lifespan=lifespan,  # register startup/shutdown events
)

# ── Rate limiting is now initialized in lifespan() ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────

# Configure CORS with environment-based origins
origins_env = settings.ALLOWED_ORIGINS
origins: list[str] = [o.strip() for o in origins_env.split(",")] if origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── NEW: Add concurrency limiting middleware ──────────────────────────────────────────
app.add_middleware(ConcurrencyLimiter, max_concurrent=4)
# ────────────────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure request time and add X-Process-Time header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """Basic health check - always returns 200 if server is running."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/v1/hello")
async def hello(current_user: str = Depends(get_current_user)):
    """Simple endpoint for token validation."""
    return {"message": f"Hello {current_user}!", "status": "authenticated"}

@app.get("/api/v1/ready")
async def ready():
    """Basic readiness check."""
    return {"ready": get_app_ready()}

@app.post("/api/v1/token", response_model=Token, dependencies=[Depends(login_limit)])
async def login(
    creds: LoginPayload = Depends(get_credentials),
    db: AsyncSession = Depends(get_db),
):
    """
    Issue a JWT. Accepts **either**
    • JSON {"username": "...", "password": "..."}  *or*
    • classic x‑www‑form‑urlencoded.
    """
    # 1️⃣ readiness gate
    if not get_app_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend still loading models. Try again in a moment.",
            headers={"Retry‑After": "10"},
        )

    # 2️⃣ verify credentials
    user = await get_user_by_username(db, creds.username)
    if not user or not verify_password(creds.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")

    # 3️⃣ issue token
    token = create_access_token(subject=user.username)
    return Token(access_token=token, token_type="bearer")

@app.get("/api/v1/ready/full")
async def ready_full() -> dict:
    """
    Extended readiness probe:
    - ready: API server is ready to accept requests (login allowed)
    - model_status: dict of {model_name: status} where status is 'loaded'|'training'|'failed'|'missing'
    - all_models_loaded: true if all models are in 'loaded' state
    """
    # Allow login if API is ready, regardless of model status
    ready_for_login = get_app_ready()

    expected = {"iris_random_forest", "breast_cancer_bayes"}
    loaded = set(model_service.models.keys())
    training = set(model_service.status.keys()) - loaded

    response = {
        "ready": ready_for_login,  # Allow login immediately
        "model_status": model_service.status,
        "all_models_loaded": all(s == "loaded" for s in model_service.status.values()),
        "models": {m: (m in loaded) for m in expected},
        "training": list(training)
    }

    logger.debug("READY endpoint – _app_ready=%s, response=%s", get_app_ready(), response)
    return response

# ── Alias routes (no auth, not shown in OpenAPI) ────────────────────────────
@app.get("/ready/full", include_in_schema=False)
async def ready_full_alias():
    """Alias for front-end calls that miss the /api/v1 prefix."""
    return await ready_full()

@app.get("/health", include_in_schema=False)
async def health_alias():
    """Alias for plain /health (SPA hits it before it knows the prefix)."""
    return await health_check()

@app.post("/token", include_in_schema=False)
async def login_alias(request: Request):
    """
    Alias: accept /token like /api/v1/token.
    Keeps the OAuth2PasswordRequestForm semantics without exposing clutter in docs.
    """
    from fastapi import Form

    # Parse form data manually to match OAuth2PasswordRequestForm behavior
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username and password are required"
        )

    # Create a mock OAuth2PasswordRequestForm object
    class MockForm:
        def __init__(self, username, password):
            self.username = username
            self.password = password

    mock_form = MockForm(username, password)

    # Reuse the existing login logic
    db = await get_db().__anext__()
    return await login(mock_form, db)

@app.post("/iris/predict", include_in_schema=False)
async def iris_predict_alias(request: Request):
    """Alias for /api/v1/iris/predict"""
    from .schemas.iris import IrisPredictRequest

    # Parse JSON body
    body = await request.json()
    iris_request = IrisPredictRequest(**body)

    # Reuse the existing prediction logic without authentication for testing
    background_tasks = BackgroundTasks()
    current_user = "test_user"  # Skip authentication for alias endpoints
    return await predict_iris(iris_request, background_tasks, current_user)

@app.post("/cancer/predict", include_in_schema=False)
async def cancer_predict_alias(request: Request):
    """Alias for /api/v1/cancer/predict"""
    from .schemas.cancer import CancerPredictRequest

    # Parse JSON body
    body = await request.json()
    cancer_request = CancerPredictRequest(**body)

    # Reuse the existing prediction logic without authentication for testing
    background_tasks = BackgroundTasks()
    current_user = "test_user"  # Skip authentication for alias endpoints
    return await predict_cancer(cancer_request, background_tasks, current_user)

# ----- on-demand training endpoints ----------------------------------
@app.post("/api/v1/iris/train", status_code=202, dependencies=[Depends(training_limit)])
async def train_iris(background_tasks: BackgroundTasks,
                     current_user: str = Depends(get_current_user)):
    background_tasks.add_task(model_service.train_iris)
    return {"status": "started"}

@app.post("/api/v1/cancer/train", status_code=202, dependencies=[Depends(training_limit)])
async def train_cancer(background_tasks: BackgroundTasks,
                       current_user: str = Depends(get_current_user)):
    background_tasks.add_task(model_service.train_cancer)
    return {"status": "started"}

@app.get("/api/v1/iris/ready")
async def iris_ready():
    """Check if Iris model is loaded and ready."""
    return {"loaded": "iris_random_forest" in model_service.models}

@app.get("/api/v1/cancer/ready")
async def cancer_ready():
    """Check if Cancer model is loaded and ready."""
    return {"loaded": "breast_cancer_bayes" in model_service.models}

@app.post(
    "/api/v1/iris/predict",
    response_model=IrisPredictResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(light_limit)]
)
async def predict_iris(
    request: IrisPredictRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
):
    """
    Predict iris species from measurements.

    Example request:
        {
            "model_type": "rf",
            "samples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            ]
        }
    """
    logger.info(f"User {current_user} called /iris/predict with {len(request.samples)} samples")
    logger.debug(f"→ Iris payload: {request.samples}")

    # Check if requested iris model is loaded; return 503 if not yet available
    if (
        request.model_type == "rf" and "iris_random_forest" not in model_service.models
    ) or (
        request.model_type == "logreg" and "iris_logreg" not in model_service.models
    ):
        logger.warning("Iris model not ready - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Iris model is still loading. Please try again in a few seconds.",
            headers={"Retry-After": "30"},
        )

    # Convert Pydantic models to dicts
    features = [sample.dict() for sample in request.samples]
    logger.debug(f"→ Iris features: {features}")

    # Get predictions
    predictions, probabilities = await model_service.predict_iris(
        features=features,
        model_type=request.model_type
    )
    logger.debug(f"← Iris predictions: {predictions}")
    logger.debug(f"← Iris probabilities: {probabilities}")

    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "input_received": request.samples
    }

    # Background task for audit logging
    background_tasks.add_task(
        logger.info,
        f"[audit] user={current_user} endpoint=iris input={request.samples} output={predictions}"
    )

    return IrisPredictResponse(**result)

@app.post(
    "/api/v1/cancer/predict",
    response_model=CancerPredictResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(heavy_limit)]
)
async def predict_cancer(
    request: CancerPredictRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
):
    """
    Predict breast cancer diagnosis from features.

    Example request:
        {
            "model_type": "bayes",
            "samples": [
                {
                    "mean_radius": 17.99,
                    "mean_texture": 10.38,
                    ...
                }
            ],
            "posterior_samples": 1000  # optional
        }
    """
    logger.info(f"User {current_user} called /cancer/predict with {len(request.samples)} samples")
    logger.debug(f"→ Cancer payload: {request.samples}")

    # No early 503 here – model_service will transparently fall back to stub if Bayes not yet ready

    # Convert Pydantic models to dicts
    features = [sample.dict() for sample in request.samples]
    logger.debug(f"→ Cancer features: {features}")

    # Get predictions
    predictions, probabilities, uncertainties = await model_service.predict_cancer(
        features=features,
        model_type=request.model_type,
        posterior_samples=request.posterior_samples
    )
    logger.debug(f"← Cancer predictions: {predictions}")
    logger.debug(f"← Cancer probabilities: {probabilities}")
    logger.debug(f"← Cancer uncertainties: {uncertainties}")

    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "uncertainties": uncertainties,
        "input_received": request.samples
    }

    # Background task for audit logging
    background_tasks.add_task(
        logger.info,
        f"[audit] user={current_user} endpoint=cancer input={request.samples} output={predictions}"
    )

    return CancerPredictResponse(**result) 

@app.get("/api/v1/debug/ready")
async def debug_ready():
    """Debug endpoint to check _app_ready status."""
    return {
        "app_ready": get_app_ready(),
        "model_service_initialized": model_service.initialized,
        "models": list(model_service.models.keys()),
        "status": model_service.status,
        "errors": {k: v for k, v in model_service.status.items() if k.endswith("_last_error")}
    }

@app.get("/api/v1/debug/compiler")
async def debug_compiler():
    """
    Debug endpoint to check JAX/NumPyro backend configuration.
    Returns information about the JAX backend setup.
    """
    try:
        import jax
        import numpyro
        import pymc as pm

        return {
            "backend": "jax_numpyro",
            "jax_version": jax.__version__,
            "numpyro_version": numpyro.__version__,
            "pymc_version": pm.__version__,
            "jax_devices": str(jax.devices()),
            "jax_platform": jax.default_backend(),
            "status": "jax_backend_configured"
        }
    except ImportError as e:
        return {
            "backend": "unknown",
            "error": f"Import error: {e}",
            "status": "missing_dependencies"
        }
    except Exception as e:
        return {
            "backend": "unknown", 
            "error": f"Configuration error: {e}",
            "status": "configuration_failed"
        } 

@app.get("/api/v1/test/401")
async def test_401():
    """Test endpoint that returns 401 for testing session expiry."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Test 401 response"
    )

# ── Debug‑only ratelimit helpers ─────────────────────────────────────────────
from .deps.limits import get_redis, user_or_ip

@app.post("/api/v1/debug/ratelimit/reset", include_in_schema=False)
async def rl_reset(request: Request):
    """
    Flush **all** rate‑limit counters bound to the caller (JWT _or_ IP).

    We match every fragment that contains the identifier to survive
    future changes in FastAPI‑Limiter's key schema.
    """
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Rate‑limiter not initialised")

    ident = await user_or_ip(request)
    keys = await r.keys(f"ratelimit:*{ident}*")        # <— broader pattern
    if keys:
        await r.delete(*keys)
    return {"reset": len(keys)}

if settings.DEBUG_RATELIMIT:          # OFF by default
    @app.get("/api/v1/debug/ratelimit/{bucket}", include_in_schema=False)
    async def rl_status(bucket: str, request: Request):
        """
        Inspect Redis keys for the current identifier + bucket.
        Handy for CI tests – **never enable in prod**.
        """
        key_prefix = f"ratelimit:{bucket}:{await user_or_ip(request)}"
        r = get_redis()
        keys = await r.keys(f"{key_prefix}*")
        values = await r.mget(keys) if keys else []
        return dict(zip(keys, values)) 
