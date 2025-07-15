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

# Conditional imports for heavy dependencies
if os.getenv("UNIT_TESTING") != "1" and os.getenv("SKIP_BACKGROUND_TRAINING") != "1":
    import pymc as pm
    import arviz as az
else:
    pm = None
    az = None

# ------------------------------------------------------------------
# Honour whatever Settings or the shell already provided; then
# fall back if the host part cannot be resolved quickly.
# ------------------------------------------------------------------
from urllib.parse import urlparse
import socket, time

def _fast_resolve(uri: str) -> bool:
    if uri.startswith("http"):
        host = urlparse(uri).hostname
        try:
            t0 = time.perf_counter()
            socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            return (time.perf_counter() - t0) < 0.05
        except socket.gaierror:
            return False
    return True

uri = os.getenv("MLFLOW_TRACKING_URI")
if not uri or not _fast_resolve(uri):
    uri = "file:./mlruns_local"

os.environ["MLFLOW_TRACKING_URI"] = uri
os.environ.setdefault("MLFLOW_REGISTRY_URI", uri)

mlflow.set_tracking_uri(uri)
logger.info("📦 Trainers using MLflow @ %s", uri)

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
#  IRIS – logistic-regression trainer (NEW)
# -----------------------------------------------------------------------------

def train_iris_logreg(
    C: float = 1.0,
    max_iter: int = 400,
    random_state: int = 42,
) -> str:
    """
    Train and register a **multinomial Logistic Regression** model on the Iris
    dataset.  Returns the MLflow run_id so the caller can reload the model.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Load and split data (stratified)
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # Fit classifier
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    ).fit(X_tr, y_tr)

    # ------------------- wrap in consistent pyfunc --------------------------
    class IrisLogRegWrapper(mlflow.pyfunc.PythonModel):
        """Expose predict() as class probabilities so the service can rely on it."""

        def __init__(self, model):
            self.model = model

        def predict(self, model_input, params=None):  # noqa: D401 – MLflow signature
            X_ = model_input.values if hasattr(model_input, "values") else model_input
            return self.model.predict_proba(X_)

        # Explicit alias so hasattr(model, "predict_proba") works post-load
        def predict_proba(self, X):
            X_ = X.values if hasattr(X, "values") else X
            return self.model.predict_proba(X_)

    preds = clf.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "precision_macro": precision_score(y_te, preds, average="macro"),
        "recall_macro": recall_score(y_te, preds, average="macro"),
    }

    with mlflow.start_run(run_name="iris_logreg") as run:
        mlflow.log_params({"C": C, "max_iter": max_iter, "random_state": random_state})
        mlflow.log_metrics(metrics)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=IrisLogRegWrapper(clf),
            registered_model_name="iris_logreg",
            input_example=X.head(),
            signature=mlflow.models.signature.infer_signature(X, clf.predict_proba(X)),
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

    class CancerStubWrapper(mlflow.pyfunc.PythonModel):
        """Return P(malignant) both via predict() and predict_proba()."""

        def __init__(self, model):
            self.model = model

        def _pp(self, X):
            X_ = X.values if hasattr(X, "values") else X
            return self.model.predict_proba(X_)

        def predict(self, model_input, params=None):
            # Return 1-D array of malignant probabilities
            return self._pp(model_input)[:, 1]

        def predict_proba(self, X):
            return self._pp(X)

    acc = accuracy_score(yte, clf.predict(Xte))

    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_stub") as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.pyfunc.log_model(
            "model",
            python_model=CancerStubWrapper(clf),
            registered_model_name="breast_cancer_stub",
            input_example=X.head(),
            signature=mlflow.models.signature.infer_signature(X, clf.predict_proba(X)),
        )
        return run.info.run_id

# -----------------------------------------------------------------------------
#  BREAST-CANCER – hierarchical Bayesian logistic regression
# -----------------------------------------------------------------------------

def train_breast_cancer_bayes(
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.95,
) -> str:
    """
    Hierarchical Bayesian logistic‑regression with varying intercepts by
    **mean_texture quintile**.

    * Uses **NumPyro NUTS** backend → **no C compilation** on Windows.  
    * Logs model to MLflow exactly like before so FastAPI can reload it.
    """

    import pymc as pm                      # PyMC ≥5.9
    import pandas as pd, numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    import mlflow, tempfile, pickle
    from pathlib import Path

    # Note: PyTensor config is set by env_sanitizer before import
    # No runtime config changes needed - they're already applied

    # 1️⃣  data ----------------------------------------------------------------
    X_df, y = load_breast_cancer(as_frame=True, return_X_y=True)
    quint, edges = pd.qcut(X_df["mean texture"], 5, labels=False, retbins=True)
    g        = np.asarray(quint, dtype="int64")          # 0‥4
    scaler   = StandardScaler().fit(X_df)
    Xs       = scaler.transform(X_df)

    # 2️⃣  model ---------------------------------------------------------------
    coords = {"group": np.arange(5)}
    with pm.Model(coords=coords) as m:
        α     = pm.Normal("α", 0.0, 1.0, dims="group")   # varying intercepts
        β     = pm.Normal("β", 0.0, 1.0, shape=Xs.shape[1])
        logit = α[g] + pm.math.dot(Xs, β)
        pm.Bernoulli("obs", logit_p=logit, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            nuts_sampler="numpyro",        # <-- magic line
            target_accept=target_accept,
            progressbar=False,
        )

    # 3️⃣  lightweight pyfunc wrapper -----------------------------------------
    class _HierBayesWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, trace, sc, ed, cols):
            self.trace, self.scaler, self.edges, self.cols = trace, sc, ed, cols

        def _quint(self, df):
            tex = df["mean texture"].to_numpy()
            return np.clip(np.digitize(tex, self.edges, right=False), 0, 4)

        def predict(self, X, params=None):
            df  = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.cols)
            xs  = self.scaler.transform(df)
            g   = self._quint(df)
            αg  = self.trace.posterior["α"].median(("chain", "draw")).values
            β   = self.trace.posterior["β"].median(("chain", "draw")).values
            log = αg[g] + np.dot(xs, β)
            return 1.0 / (1.0 + np.exp(-log))

    wrapper = _HierBayesWrapper(idata, scaler, edges[1:-1], X_df.columns.tolist())
    acc     = float(((wrapper.predict(X_df) > .5).astype(int) == y).mean())

    # 4️⃣  MLflow logging (unchanged) -----------------------------------------
    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_bayes") as run:
        sc_path = Path(td) / "scaler.pkl"
        pickle.dump(scaler, open(sc_path, "wb"))
        mlflow.log_params(dict(draws=draws, tune=tune, target_accept=target_accept))
        mlflow.log_metric("accuracy", acc)
        mlflow.pyfunc.log_model(
            "model",
            python_model=wrapper,
            artifacts={"scaler": str(sc_path)},
            registered_model_name="breast_cancer_bayes",
            input_example=X_df.head(),
            signature=mlflow.models.signature.infer_signature(X_df, wrapper.predict(X_df)),
        )
        return run.info.run_id
