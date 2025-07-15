"""
Model service – self-healing startup with background training.
"""

from __future__ import annotations
import asyncio, logging, os, time, socket, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import mlflow, pandas as pd, numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from app.core.config import settings
from app.ml.builtin_trainers import (
    train_iris_random_forest,
    train_iris_logreg,  # NEW
    train_breast_cancer_bayes,
    train_breast_cancer_stub,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cancer column mapping: Pydantic field names ➜ training column names
# ---------------------------------------------------------------------------
_CANCER_COLMAP: dict[str, str] = {
    # Means
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
    # SE
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
    # Worst
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

def _rename_cancer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame columns match the training schema used by MLflow artefacts.
    Unknown columns are left untouched so legacy models still work.
    """
    return df.rename(columns=_CANCER_COLMAP)

# Trainer mapping for self-healing
TRAINERS = {
    "iris_random_forest": train_iris_random_forest,
    "iris_logreg":        train_iris_logreg,  # NEW
    "breast_cancer_bayes": train_breast_cancer_bayes,
    "breast_cancer_stub":  train_breast_cancer_stub,
}

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
            "iris_logreg":        "missing",  # NEW
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

        # Log critical dependency versions for diagnostics
        try:
            import pytensor
            logger.info("📦 PyTensor version: %s", pytensor.__version__)
        except ImportError:
            logger.warning("⚠️  PyTensor not available")
        except Exception as e:
            logger.warning("⚠️  Could not determine PyTensor version: %s", e)

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
        for name in ["iris_random_forest", "iris_logreg",
                     "breast_cancer_bayes", "breast_cancer_stub"]:
            try:
                await self._try_load(name)
            except Exception as exc:
                logger.error("❌  load %s failed: %s", name, exc)

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
            await self._try_load("iris_logreg")
            await self._try_load("breast_cancer_bayes")
            return

        auto = auto_train if auto_train is not None else settings.AUTO_TRAIN_MISSING
        logger.info("🔄 Model-service startup (auto_train=%s)", auto)

        # 1️⃣ try to load whatever already exists
        await self._try_load("iris_random_forest")
        await self._try_load("iris_logreg")

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
            logger.info("Training iris random-forest …")
            await asyncio.get_running_loop().run_in_executor(
                self._EXECUTOR, train_iris_random_forest
            )
            await self._try_load("iris_random_forest")

        if not await self._try_load("iris_logreg"):
            logger.info("Training iris logistic-regression …")
            await asyncio.get_running_loop().run_in_executor(
                self._EXECUTOR, train_iris_logreg
            )
            await self._try_load("iris_logreg")

    async def _try_load(self, name: str) -> bool:
        """Try to load a model and update status."""
        try:
            model = await self._load_production_model(name)
            if model:
                self.models[name] = model
                self.status[name] = "loaded"
                logger.info("✅ %s loaded", name)
                return True
            self.status.setdefault(name, "missing")
            return False
        except Exception as exc:
            logger.error("❌  load %s failed: %s", name, exc)
            self.status[name] = "failed"
            self.status[f"{name}_last_error"] = str(exc)
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

            # Trigger retention clean‑up in background
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self._EXECUTOR,
                                 lambda: asyncio.run(self._cleanup_runs(name)))
            logger.info("✅ %s trained & loaded", name)

        except Exception as exc:
            self.status[name] = "failed"
            logger.error("❌ %s failed: %s", name, exc, exc_info=True)  # ← keeps trace
            # NEW: persist last_error for UI / debug endpoint
            self.status[f"{name}_last_error"] = str(exc)

    async def _load_production_model(self, name: str) -> Optional[Any]:
        """
        Return the Production-stage model if its *artifacts* are present.
        Falls back to the most recent run, and returns None when
        artifacts are missing instead of propagating MlflowException.
        """
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            if prod:
                uri = f"models:/{name}/{prod[0].version}"
                logger.info("↪︎  Loading %s from registry:%s", name, prod[0].version)
                return mlflow.pyfunc.load_model(uri)
        except (MlflowException, FileNotFoundError, OSError) as exc:
            # 404-style errors: artefact path vanished – log & continue
            logger.warning("🗂️  %s production artefact missing: %s", name, exc)
            return None

        # Fallback – latest run
        try:
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
        except (MlflowException, FileNotFoundError, OSError) as exc:
            logger.warning("🗂️  %s latest-run artefact missing: %s", name, exc)
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
            model_type: Model type to use (only 'rf' supported)

        Returns:
            Tuple of (predicted_class_names, class_probabilities)
        """
        if model_type not in ("rf", "logreg"):
            raise ValueError("model_type must be 'rf' or 'logreg'")

        model_name = "iris_random_forest" if model_type == "rf" else "iris_logreg"
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

        # Obtain probabilities in a backward-compatible way
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)
        else:
            # Legacy artefact – derive 1-hot probas from class indices
            preds_idx = model.predict(X_df)
            import numpy as _np
            probs = _np.zeros((len(preds_idx), 3), dtype=float)
            probs[_np.arange(len(preds_idx)), preds_idx.astype(int)] = 1.0

        # Ensure numpy array then list list
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
        # Determine which model to use
        if model_type == "bayes":
            model = self.models.get("breast_cancer_bayes")
            if not model:
                # Fall back to stub model
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

        # Convert to DataFrame with proper column names
        X_df_raw = pd.DataFrame(features)
        X_df = _rename_cancer_columns(X_df_raw)

        # Get predictions
        if model_type == "bayes" and "breast_cancer_bayes" in self.models:
            # Use Bayesian model with uncertainty
            probs = model.predict(X_df)
            labels = ["malignant" if p > 0.5 else "benign" for p in probs]
        else:
            # Use stub model (sklearn LogisticRegression)
            if hasattr(model, "predict_proba"):
                probs_full = model.predict_proba(X_df)
                probs = probs_full[:, 1]
            else:
                # Legacy artefact: model.predict returns hard class 0/1
                preds_bin = model.predict(X_df).astype(float)
                probs = preds_bin  # deterministic 0/1 acts as prob
            labels = ["malignant" if p > 0.5 else "benign" for p in probs]

        # Compute uncertainty intervals if requested (Bayesian model only)
        ci = None
        if posterior_samples and model_type == "bayes" and "breast_cancer_bayes" in self.models:
            try:
                # Access the underlying python model to get the trace
                python_model = model.unwrap_python_model()

                # Access posterior samples for uncertainty quantification
                draws = python_model.trace.posterior
                αg = draws["α_group"].stack(samples=("chain", "draw"))
                β = draws["β"].stack(samples=("chain", "draw"))

                # Get group indices and standardized features
                g = python_model._group_index(X_df)
                Xs = python_model.scaler.transform(X_df)

                # Compute posterior predictive samples
                logits = αg.values[:, g] + np.dot(β.values.T, Xs.T)      # shape (S, N)
                pp = 1 / (1 + np.exp(-logits))

                # Compute 95% credible intervals
                lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
                ci = list(zip(lo.tolist(), hi.tolist()))

            except Exception as e:
                logger.warning(f"Failed to compute uncertainty intervals: {e}")
                ci = None

        return labels, probs.tolist(), ci

    async def _cleanup_runs(self, model_name: str) -> None:
        """
        Keep the **newest N runs** for `model_name` and drop the rest, then
        optionally invoke `mlflow gc` to purge artifact folders.

        Runs marked *deleted* are still present on disk until GC executes,
        so we always run GC when `settings.MLFLOW_GC_AFTER_TRAIN` is True.
        """
        keep = max(settings.RETAIN_RUNS_PER_MODEL, 0)
        try:
            # 1️⃣ fetch runs newest→oldest
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id for exp in self.mlflow_client.search_experiments()],
                filter_string=f"tags.mlflow.runName = '{model_name}'",
                order_by=["attributes.start_time DESC"],
            )
            if len(runs) <= keep:
                logger.debug("No pruning needed for %s (runs=%d, keep=%d)",
                             model_name, len(runs), keep)
                return

            to_delete = runs[keep:]
            for r in to_delete:
                self.mlflow_client.delete_run(r.info.run_id)
            logger.info("🗑️  Pruned %d old %s runs; kept %d",
                        len(to_delete), model_name, keep)

            # 2️⃣ garbage‑collect artifacts
            if settings.MLFLOW_GC_AFTER_TRAIN:
                uri = mlflow.get_tracking_uri().removeprefix("file:")
                before = shutil.disk_usage(uri).used
                subprocess.run(
                    ["mlflow", "gc",
                     "--backend-store-uri", uri,
                     "--artifact-store", uri],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                after = shutil.disk_usage(uri).used
                logger.info("🧹 mlflow gc completed (%.2f MB → %.2f MB)",
                            before/1e6, after/1e6)

        except Exception as exc:
            logger.warning("Cleanup for %s failed: %s", model_name, exc)

    async def vacuum_store(self) -> None:
        """Force a *store‑wide* `mlflow gc` (use from cron jobs)."""
        try:
            uri = mlflow.get_tracking_uri().removeprefix("file:")
            before = shutil.disk_usage(uri).used
            subprocess.run(
                ["mlflow", "gc",
                 "--backend-store-uri", uri,
                 "--artifact-store", uri],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            after = shutil.disk_usage(uri).used
            logger.info("🧹 Store-wide vacuum completed (%.2f MB → %.2f MB)",
                        before/1e6, after/1e6)
        except Exception as exc:
            logger.warning("Store vacuum failed: %s", exc)


# Global singleton
model_service = ModelService()
