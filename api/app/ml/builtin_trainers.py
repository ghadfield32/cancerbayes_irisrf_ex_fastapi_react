# api/ml/builtin_trainers.py
"""
Built-in trainers for Iris RF and Breast-Cancer Bayesian LogReg.
Executed automatically by ModelService when a model is missing.
"""

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import mlflow, mlflow.sklearn, mlflow.pyfunc
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import tempfile
import pickle
import warnings
import subprocess
import os
import platform

# ── NEW: Configure MLflow to use local file storage ─────────────────────────
# Set MLflow to use local file storage instead of remote server
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns_local")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "file:./mlruns_local")

# Configure MLflow tracking URI immediately
mlflow.set_tracking_uri("file:./mlruns_local")
# ──────────────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT = "ml_fullstack_models"

# Only set experiment if not in unit test mode and after tracking URI is set
if os.getenv("UNIT_TESTING") != "1":
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception as e:
        logging.warning(f"Could not set MLflow experiment: {e}")

# -----------------------------------------------------------------------------
#  IRIS – point-estimate Random-Forest (enhanced with better parameters)
# -----------------------------------------------------------------------------
def train_iris_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42
) -> str:
    """
    Train + register a Random-Forest on the Iris data and push it to MLflow.
    Returns the run_id (string). Enhanced with better parameters and stratified split.
    """
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              stratify=y, random_state=random_state)

    # Enhanced Random Forest with better parameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle any class imbalance
    ).fit(X_tr, y_tr)

    preds = rf.predict(X_te)
    metrics = {
        "accuracy":  accuracy_score(y_te, preds),
        "f1_macro":  f1_score(y_te, preds, average="macro"),
        "precision_macro": precision_score(y_te, preds, average="macro"),
        "recall_macro":    recall_score(y_te, preds, average="macro"),
    }

    with mlflow.start_run(run_name="iris_random_forest") as run:
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Create a custom pyfunc wrapper that exposes both predict and predict_proba
        class IrisRFWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, model_input, params=None):
                # Return class probabilities for pyfunc interface
                # Convert to numpy array if it's a DataFrame
                if hasattr(model_input, 'values'):
                    X = model_input.values
                else:
                    X = model_input
                return self.model.predict_proba(X)

            def predict_proba(self, X):
                # Expose predict_proba for direct access
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict_proba(X)

            def predict_classes(self, X):
                # Expose class prediction
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict(X)

        iris_wrapper = IrisRFWrapper(rf)

        # Log model with proper signature
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=iris_wrapper,
            registered_model_name="iris_random_forest",
            input_example=X.head(),
            signature=mlflow.models.signature.infer_signature(X, iris_wrapper.predict(X))
        )
        return run.info.run_id

# -----------------------------------------------------------------------------
#  IRIS – Logistic Regression (multinomial for 3-class classification)
# -----------------------------------------------------------------------------
def train_iris_logistic_regression(
    max_iter: int = 400,
    C: float = 1.0,
    random_state: int = 42
) -> str:
    """
    Train + register a multinomial Logistic Regression on the Iris data and push it to MLflow.
    Returns the run_id (string).
    """
    from sklearn.linear_model import LogisticRegression

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # Multinomial logistic regression for 3-class classification
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        C=C,
        n_jobs=-1,
        random_state=random_state,
    ).fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "precision_macro": precision_score(y_te, preds, average="macro"),
        "recall_macro": recall_score(y_te, preds, average="macro"),
    }

    with mlflow.start_run(run_name="iris_logistic_regression") as run:
        # Log hyperparameters
        mlflow.log_params({
            "max_iter": max_iter,
            "C": C,
            "random_state": random_state
        })

        # Log metrics
        mlflow.log_metrics(metrics)

        # Create a custom pyfunc wrapper that exposes both predict and predict_proba
        class IrisLogRegWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, model_input, params=None):
                # Return class probabilities for pyfunc interface
                # Convert to numpy array if it's a DataFrame
                if hasattr(model_input, 'values'):
                    X = model_input.values
                else:
                    X = model_input
                return self.model.predict_proba(X)

            def predict_proba(self, X):
                # Expose predict_proba for direct access
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict_proba(X)

            def predict_classes(self, X):
                # Expose class prediction
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict(X)

        iris_wrapper = IrisLogRegWrapper(clf)

        # Log model with proper signature
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=iris_wrapper,
            registered_model_name="iris_logistic_regression",
            input_example=X.head(),
            signature=mlflow.models.signature.infer_signature(X, iris_wrapper.predict(X))
        )
        return run.info.run_id

# -----------------------------------------------------------------------------
#  BREAST-CANCER STUB – ultra-fast fallback model
# -----------------------------------------------------------------------------
def train_breast_cancer_stub(random_state: int = 42) -> str:
    """
    *Ultra-fast* fallback –  < 100 ms on any laptop.
    Trains vanilla LogisticRegression so the API can
    answer probability queries while the PyMC model cooks.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import mlflow, tempfile, pickle, pandas as pd

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=random_state)

    clf = LogisticRegression(max_iter=200, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))

    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_stub") as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            clf, "model",
            registered_model_name="breast_cancer_stub",
            input_example=X.head()
        )
        return run.info.run_id

# -----------------------------------------------------------------------------
#  BREAST-CANCER – hierarchical Bayesian logistic regression
# -----------------------------------------------------------------------------
class _HierBayesLogReg(mlflow.pyfunc.PythonModel):
    """
    Hierarchical Bayesian Logistic Regression wrapper for MLflow serving.
    Implements varying intercepts by mean_texture quintiles with global slopes.
    """

    def __init__(self, trace, scaler, group_edges, feature_names):
        self.trace = trace                # ArviZ InferenceData for posterior samples
        self.scaler = scaler              # sklearn StandardScaler for feature normalization
        self.group_edges = group_edges    # bin edges for creating group indices
        self.feature_names = feature_names # column names for proper DataFrame handling

    def _group_index(self, X_df):
        """Create group indices based on mean_texture quintiles."""
        tex = X_df["mean texture"].to_numpy()
        # Use same quintile edges as training, clipping to valid range
        return np.clip(np.digitize(tex, self.group_edges, right=False), 0, 4)

    def predict(self, model_input, params=None):
        """
        MLflow-required prediction method.

        Args:
            model_input: pandas.DataFrame with breast cancer features
            params: Optional parameters (unused)

        Returns:
            np.array: Probability of malignancy [0,1] for each sample
        """
        # Ensure we have a DataFrame with proper column order
        if isinstance(model_input, pd.DataFrame):
            X_df = model_input
        else:
            X_df = pd.DataFrame(model_input, columns=self.feature_names)

        # Standardize features using training scaler
        Xs = self.scaler.transform(X_df)

        # Get group indices for hierarchical structure
        g = self._group_index(X_df)

        # Extract posterior medians for prediction
        α = self.trace.posterior["α_group"].median(("chain", "draw")).values
        β = self.trace.posterior["β"].median(("chain", "draw")).values

        # Compute predictions: logit = α_group[g] + X @ β
        logits = α[g] + np.dot(Xs, β)

        # Convert to probabilities
        return 1 / (1 + np.exp(-logits))

def train_breast_cancer_bayes(
    draws: int = 500,      # Reduced from 800 for speed
    tune: int = 200,       # Reduced from 400 for speed  
    target_accept: float = 0.95,  # Increased for better convergence
) -> str:
    """
    Train a hierarchical Bayesian logistic-regression model with optimizations.

    Key optimizations:
    - Single-core sampling (Windows optimization)
    - Reduced draws/tune for speed
    - Higher target_accept for better convergence
    - Automatic backend selection (NumPyro > PyMC)
    - Environment cleanup for stability
    """
    # ------------------------------------------------------------------ Environment cleanup
    # Strip invalid flags that crash JAX
    os.environ.pop("XLA_FLAGS", None)  # defensive – avoid "--"

    # ------------------------------------------------------------------ Compiler configuration
    from app.ml.utils import find_compiler, configure_pytensor_compiler

    logger.info("🔧 Configuring PyTensor compiler...")
    cxx = find_compiler()
    if cxx is None:
        msg = "No compatible C/C++ compiler found – skipping Bayesian build."
        logger.warning("⚠️ %s", msg)
        raise RuntimeError(msg)

    if not configure_pytensor_compiler(cxx):
        msg = "PyTensor compiler configuration failed – skipping Bayesian build."
        logger.warning("⚠️ %s", msg)
        raise RuntimeError(msg)

    # ---------------------------------------------------------------- Optimized sampler selection
    def _jax_available():
        try:
            import jax
            jax.devices()
            return True
        except Exception:
            return False

    def _numpyro_available():
        try:
            import numpyro
            return True
        except ImportError:
            return False

    def _nutpie_available():
        try:
            import nutpie
            return True
        except ImportError:
            return False

    # Determine best available sampler with proper detection
    if _jax_available() and _numpyro_available():
        sampler_backend = "numpyro"
        logger.info("✅ Using NumPyro backend (fastest)")
    elif _nutpie_available():
        sampler_backend = "nutpie"
        logger.info("✅ Using NutPie backend (Stan-powered)")
    else:
        sampler_backend = "pymc"
        logger.info("⚠️ Using PyMC backend (fallback)")

    # ------------------------------------------------------------------ Modelling
    logger.info("🧠 Building optimized hierarchical Bayesian model (backend=%s)…", sampler_backend)

    # Import heavy libraries
    import pymc as pm
    import arviz as az
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    import mlflow
    import tempfile
    import pickle

    # Prepare data
    X_df, y = load_breast_cancer(as_frame=True, return_X_y=True)
    quint, edges = pd.qcut(X_df["mean texture"], 5, labels=False, retbins=True)
    g = quint.astype("int64").to_numpy()
    scaler = StandardScaler().fit(X_df)
    Xs = scaler.transform(X_df)

    logger.info("🧠 Building optimized hierarchical Bayesian model (backend=%s)…", sampler_backend)
    with pm.Model() as mdl:
        α_group = pm.Normal("α_group", mu=0, sigma=1, shape=5)
        β = pm.Normal("β", mu=0, sigma=1, shape=Xs.shape[1])
        logits = α_group[g] + pm.math.dot(Xs, β)
        pm.Bernoulli("obs", logit_p=logits, observed=y)

        # Optimized sampling parameters for Windows
        sample_kwargs = {
            "draws": draws,
            "tune": tune,
            "target_accept": target_accept,
            "chains": 2,
            "cores": 1,  # Single core to avoid Windows process spawn overhead
            "progressbar": False,
            "random_seed": 123,
        }

        # Only add nuts_sampler if not using default pymc
        if sampler_backend != "pymc":
            sample_kwargs["nuts_sampler"] = sampler_backend

        try:
            trace = pm.sample(**sample_kwargs)
        except Exception as e:
            logger.warning(f"Sampling with {sampler_backend} failed: {e}")
            logger.info("🔄 Falling back to standard PyMC sampler...")
            # Remove nuts_sampler for fallback
            sample_kwargs.pop("nuts_sampler", None)
            trace = pm.sample(**sample_kwargs)

    # ------------------------------------------------------------------ Optimized wrapper
    class _OptimizedHierBayesLogReg(mlflow.pyfunc.PythonModel):
        def __init__(self, trc, sc, edge, cols):
            self.trace, self.scaler, self.edges, self.cols = trc, sc, edge, cols

        def _grp(self, df):  # replicate training quintiles
            tex = df["mean texture"].to_numpy()
            return np.clip(np.digitize(tex, self.edges, right=False), 0, 4)

        def predict(self, model_input, params=None):
            df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input, columns=self.cols)
            Xs = self.scaler.transform(df)
            g = self._grp(df)
            αg = self.trace.posterior["α_group"].median(("chain", "draw")).values
            β = self.trace.posterior["β"].median(("chain", "draw")).values
            lg = αg[g] + np.dot(Xs, β)
            return 1 / (1 + np.exp(-lg))

    wrapper = _OptimizedHierBayesLogReg(trace, scaler, edges[1:-1], X_df.columns.tolist())
    preds = (wrapper.predict(X_df) > 0.5).astype(int)
    acc = float((preds == y).mean())

    # -------------------------------------------------------------------- MLflow logging
    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_bayes") as run:
        scaler_path = Path(td) / "scaler.pkl"
        with open(scaler_path, "wb") as fh:
            pickle.dump(scaler, fh)

        mlflow.log_params({
            "draws": draws, 
            "tune": tune, 
            "target_accept": target_accept,
            "sampler": sampler_backend,
            "optimized": True
        })
        mlflow.log_metric("accuracy", acc)
        mlflow.pyfunc.log_model(
            "model", python_model=wrapper,
            artifacts={"scaler": str(scaler_path)},
            registered_model_name="breast_cancer_bayes",
            input_example=X_df.head(),
            signature=mlflow.models.signature.infer_signature(X_df, wrapper.predict(X_df)),
        )
        logger.info("📦 Optimized Bayesian model logged – run_id=%s  acc=%.3f", run.info.run_id, acc)
        return run.info.run_id
