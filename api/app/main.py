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

from .db import lifespan, get_db, get_app_ready
from .security import create_access_token, get_current_user, verify_password
from .crud import get_user_by_username
from .schemas.iris import IrisPredictRequest, IrisPredictResponse, IrisFeatures
from .schemas.cancer import CancerPredictRequest, CancerPredictResponse, CancerFeatures
from .services.ml.model_service import model_service
from .core.config import settings

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

@app.post("/api/v1/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate user and issue JWT."""
    if not get_app_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend still loading models. Try again in a moment.",
            headers={"Retry-After": "10"}
        )

    user = await get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
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

# ----- on-demand training endpoints ----------------------------------
@app.post("/api/v1/iris/train", status_code=202)
async def train_iris(background_tasks: BackgroundTasks,
                     current_user: str = Depends(get_current_user)):
    background_tasks.add_task(model_service.train_iris)
    return {"status": "started"}

@app.post("/api/v1/cancer/train", status_code=202)
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
    status_code=status.HTTP_200_OK
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

    # Check if iris model is ready
    if request.model_type == "rf" and "iris_random_forest" not in model_service.models:
        logger.warning("Iris model not ready - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Iris model is still loading. Please try again in a few seconds.",
            headers={"Retry-After": "30"}
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
    status_code=status.HTTP_200_OK
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

    # Check if cancer model is ready
    if request.model_type == "bayes" and "breast_cancer_bayes" not in model_service.models:
        logger.warning("Cancer model not ready - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Cancer model is still loading. Please try again in a few seconds.",
            headers={"Retry-After": "30"}
        )

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

@app.get("/api/v1/test/401")
async def test_401():
    """Test endpoint that returns 401 for testing session expiry."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Test 401 response"
    ) 
