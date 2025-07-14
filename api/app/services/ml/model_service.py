"""
Model service – self-healing startup with background training.
"""

from __future__ import annotations
import asyncio, logging, os, time, socket
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Optional

import mlflow, pandas as pd, numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from app.core.config import settings
from app.ml.builtin_trainers import (
    train_iris_random_forest,
    train_iris_logistic_regression,
    train_breast_cancer_bayes,
    train_breast_cancer_stub,
)

logger = logging.getLogger(__name__)

# Trainer mapping for self-healing
TRAINERS = {
    "iris_random_forest": train_iris_random_forest,
    "iris_logistic_regression": train_iris_logistic_regression,
    "breast_cancer_bayes": train_breast_cancer_bayes,
    "breast_cancer_stub": train_breast_cancer_stub,
}

# ── NEW: Column name mapping for cancer predictions ─────────────────────────
_BC_SNAKE_TO_SPACE = {
    # Mean features
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",
    # SE features (standard error)
    "se_radius": "radius error",
    "se_texture": "texture error",
    "se_perimeter": "perimeter error",
    "se_area": "area error",
    "se_smoothness": "smoothness error",
    "se_compactness": "compactness error",
    "se_concavity": "concavity error",
    "se_concave_points": "concave points error",
    "se_symmetry": "symmetry error",
    "se_fractal_dimension": "fractal dimension error",
    # Worst features
    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension",
}

_BC_TRAINING_COLS = list(_BC_SNAKE_TO_SPACE.values())

def _normalize_cancer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert snake_case API columns to the space-separated column names
    captured in the MLflow model signature.

    Raises
    ------
    ValueError
        When, after renaming, we still miss required columns.
    """
    logger.debug("Normalising cancer features – incoming cols=%s", list(df.columns))
    df = df.rename(columns=_BC_SNAKE_TO_SPACE)
    missing = [c for c in _BC_TRAINING_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required fields after normalisation: {missing}"
        )
    # keep only training columns and ordered correctly
    df = df[_BC_TRAINING_COLS]
    logger.debug("Normalised cols=%s", list(df.columns))
    return df

class ModelService:
    """
    Self-healing model service that loads existing models and schedules
    background training for missing ones.
    """

    _EXECUTOR = ThreadPoolExecutor(max_workers=2)

    def __init__(self) -> None:
        self._unit_test_mode = settings.UNIT_TESTING
        self.initialized = False

        # 🚫 Heavy clients only when NOT unit-testing
        self.client = None if self._unit_test_mode else None  # Will be set in initialize()
        self.mlflow_client = None

        self.models: Dict[str, Any] = {}
        self.status: Dict[str, str] = {
            "iris_random_forest": "missing",
            "iris_logistic_regression": "missing",
            "breast_cancer_bayes": "missing",
            "breast_cancer_stub": "missing",
        }

    async def initialize(self) -> None:
        """
        Connect to MLflow – fall back to local file store if the configured
        tracking URI is unreachable *or* the client is missing critical methods
        (e.g. when mlflow-skinny accidentally shadows the full package).
        """
        if self.initialized:
            return

        def _needs_fallback(client) -> bool:
            # any missing attr is a strong signal we are on mlflow-skinny
            return not callable(getattr(client, "list_experiments", None))

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            self.mlflow_client = MlflowClient(settings.MLFLOW_TRACKING_URI)

            if _needs_fallback(self.mlflow_client):
                raise AttributeError("list_experiments not implemented – skinny build detected")

            # minimal probe (cheap & always present)
            self.mlflow_client.search_experiments(max_results=1)
            logger.info("🟢  Connected to MLflow @ %s", settings.MLFLOW_TRACKING_URI)

        except (MlflowException, socket.gaierror, AttributeError) as exc:
            logger.warning("🔄  Falling back to local MLflow store – %s", exc)
            mlflow.set_tracking_uri("file:./mlruns_local")
            self.mlflow_client = MlflowClient("file:./mlruns_local")
            logger.info("📂  Using local file store ./mlruns_local")

        await self._load_models()
        self.initialized = True

    async def _load_models(self) -> None:
        """Load existing models from MLflow."""
        await self._try_load("iris_random_forest")
        await self._try_load("iris_logistic_regression")
        await self._try_load("breast_cancer_bayes")
        await self._try_load("breast_cancer_stub")

    async def startup(self, auto_train: bool | None = None) -> None:
        """
        Faster: serve stub immediately; heavy Bayesian job in background.
        """
        if self._unit_test_mode:
            logger.info("🔒 UNIT_TESTING=1 – skipping model loading")
            return                      # 👉 nothing else runs

        # Initialize MLflow connection first
        await self.initialize()

        if settings.SKIP_BACKGROUND_TRAINING:
            logger.warning("⏩ SKIP_BACKGROUND_TRAINING=1 – models will load on-demand")
            # We still *try* to load existing artefacts so prod works
            await self._try_load("iris_random_forest")
            await self._try_load("breast_cancer_bayes")
            return

        auto = auto_train if auto_train is not None else settings.AUTO_TRAIN_MISSING
        logger.info("🔄 Model-service startup (auto_train=%s)", auto)

        # 1️⃣ try to load whatever already exists
        await self._try_load("iris_random_forest")

        # 2️⃣ Load bayes – if exists we're done
        if not await self._try_load("breast_cancer_bayes"):
            # 3️⃣ Ensure stub is *synchronously* available
            if not await self._try_load("breast_cancer_stub"):
                logger.info("Training stub cancer model …")
                await asyncio.get_running_loop().run_in_executor(
                    self._EXECUTOR, train_breast_cancer_stub
                )
                await self._try_load("breast_cancer_stub")

            # 4️⃣ Fire full PyMC build in background unless disabled
            if not settings.SKIP_BACKGROUND_TRAINING:
                logger.info("Scheduling full Bayesian retrain in background")
                asyncio.create_task(
                    self._train_and_reload("breast_cancer_bayes", train_breast_cancer_bayes)
                )

        # 5️⃣ Train iris models if missing
        if not await self._try_load("iris_random_forest"):
            logger.info("Training iris random forest model …")
            await asyncio.get_running_loop().run_in_executor(
                self._EXECUTOR, train_iris_random_forest
            )
            await self._try_load("iris_random_forest")

        if not await self._try_load("iris_logistic_regression"):
            logger.info("Training iris logistic regression model …")
            await asyncio.get_running_loop().run_in_executor(
                self._EXECUTOR, train_iris_logistic_regression
            )
            await self._try_load("iris_logistic_regression")

    async def _try_load(self, name: str) -> None:
        """Try to load a model and update status."""
        model = await self._load_production_model(name)
        if model:
            self.models[name] = model
            self.status[name] = "loaded"
            logger.info("✅ %s loaded", name)
            return True
        return False

    async def _train_and_reload(self, name: str, trainer) -> None:
        """Train a model in background and reload it, with verbose phase logs."""
        try:
            t0 = time.perf_counter()
            logger.info("🏗️  BEGIN training %s", name)
            self.status[name] = "training"

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._EXECUTOR, trainer)

            logger.info("📦 Training %s complete in %.1fs – re-loading", name,
                        time.perf_counter() - t0)
            model = await self._load_production_model(name)
            if not model:
                raise RuntimeError(f"{name} trained but could not be re-loaded")

            self.models[name] = model
            self.status[name] = "loaded"
            logger.info("✅ %s trained & loaded", name)

        except Exception as exc:
            self.status[name] = "failed"
            logger.error("❌ %s failed: %s", name, exc, exc_info=True)  # ← keeps trace
            # NEW: persist last_error for UI / debug endpoint
            self.status[f"{name}_last_error"] = str(exc)

    async def _load_production_model(self, name: str) -> Optional[Any]:
        """
        1. Registry 'Production' stage → load.  
        2. Otherwise most recent run with runName == name.
        Returns None if not found.
        """
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if prod:
                uri = f"models:/{name}/{prod[0].version}"
                logger.info("↪︎  Loading %s from registry:%s", name, prod[0].version)
                return mlflow.pyfunc.load_model(uri)
        except MlflowException:
            pass

        # Fallback – scan experiments for latest run
        runs = []
        for exp in self.mlflow_client.search_experiments():
            runs.extend(self.mlflow_client.search_runs(
                [exp.experiment_id],
                f"tags.mlflow.runName = '{name}'",
                order_by=["attributes.start_time DESC"],
                max_results=1))
        if runs:
            uri = f"runs:/{runs[0].info.run_id}/model"
            logger.info("↪︎  Loading %s from latest run:%s", name, runs[0].info.run_id)
            return mlflow.pyfunc.load_model(uri)
        return None

    # Manual training endpoints (for UI)
    async def train_iris(self) -> None:
        await self._train_and_reload("iris_random_forest", TRAINERS["iris_random_forest"])

    async def train_cancer(self) -> None:
        await self._train_and_reload("breast_cancer_bayes", TRAINERS["breast_cancer_bayes"])

    # Predict methods (unchanged from your previous version)
    async def predict_iris(
        self,
        features: List[Dict[str, float]],
        model_type: str = "rf",
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Predict Iris species from measurements.

        Args:
            features: List of iris measurements as dictionaries
            model_type: Model type to use ('rf' or 'logreg')

        Returns:
            Tuple of (predicted_class_names, class_probabilities)
        """
        # Select model based on model_type
        if model_type == "rf":
            model_name = "iris_random_forest"
        elif model_type == "logreg":
            model_name = "iris_logistic_regression"
        else:
            raise ValueError("model_type must be 'rf' or 'logreg'")

        model = self.models.get(model_name)
        if not model:
            raise RuntimeError(f"{model_name} not loaded")

        # Convert to DataFrame with proper column names (matching training data)
        X_df = pd.DataFrame([{
            "sepal length (cm)": sample["sepal_length"],
            "sepal width (cm)": sample["sepal_width"], 
            "petal length (cm)": sample["petal_length"],
            "petal width (cm)": sample["petal_width"]
        } for sample in features])

        # Both model wrappers return probabilities via predict() method
        probs = model.predict(X_df)                  # shape (n, 3) - probabilities
        preds = probs.argmax(axis=1)                 # numerical class indices

        # Map numerical classes to species names
        class_names = ["setosa", "versicolor", "virginica"]
        pred_names = [class_names[i] for i in preds]

        return pred_names, probs.tolist()

    async def predict_cancer(
        self,
        features: List[Dict[str, float]],
        model_type: str = "bayes",
        posterior_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[float], Optional[List[Tuple[float, float]]]]:
        """
        Predict breast cancer diagnosis from features using hierarchical Bayesian model.
        Falls back to stub model if Bayesian model is not available.

        Args:
            features: List of cancer measurements as dictionaries
            model_type: Model type to use ('bayes' or 'stub')
            posterior_samples: Number of posterior samples for uncertainty (Bayesian only)

        Returns:
            Tuple of (predicted_labels, probabilities, uncertainty_intervals)
        """
        # 1️⃣  Select model (same logic as before) --------------------------
        if model_type == "bayes":
            model = self.models.get("breast_cancer_bayes")
            if not model:
                model = self.models.get("breast_cancer_stub")
                if not model:
                    raise RuntimeError("No cancer model available")
                logger.info("Using stub cancer model (Bayesian model not ready)")
        elif model_type == "stub":
            model = self.models.get("breast_cancer_stub")
            if not model:
                raise RuntimeError("Stub cancer model not loaded")
        else:
            raise ValueError("model_type must be 'bayes' or 'stub'")

        # 2️⃣  Build & normalise DataFrame ---------------------------------
        raw_df = pd.DataFrame(features)
        try:
            X_df = _normalize_cancer_features(raw_df)
        except ValueError as exc:
            logger.error("Schema validation failed: %s", exc)
            raise

        # 3️⃣  Predict ------------------------------------------------------
        if model_type == "bayes" and "breast_cancer_bayes" in self.models:
            probs = model.predict(X_df)           # returns shape (n,)
            labels = ["malignant" if p > 0.5 else "benign" for p in probs]
        else:
            probs = model.predict_proba(X_df)[:, 1]
            labels = ["malignant" if p > 0.5 else "benign" for p in probs]

        # 4️⃣  Credible intervals (optional) -------------------------------
        ci = None
        if (
            posterior_samples
            and model_type == "bayes"
            and "breast_cancer_bayes" in self.models
        ):
            try:
                python_model = model.unwrap_python_model()
                draws = python_model.trace.posterior
                αg = draws["α_group"].stack(samples=("chain", "draw"))
                β = draws["β"].stack(samples=("chain", "draw"))
                g = python_model._grp(X_df)  # Fixed method name
                Xs = python_model.scaler.transform(X_df)
                logits = αg.values[:, g] + np.dot(β.values.T, Xs.T)
                pp = 1 / (1 + np.exp(-logits))
                lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
                ci = list(zip(lo.tolist(), hi.tolist()))
            except Exception as e:
                logger.warning(f"Failed to compute uncertainty intervals: {e}")
                ci = None

        return labels, probs.tolist(), ci


# Global singleton
model_service = ModelService()
