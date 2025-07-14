#!/usr/bin/env python3
"""
Hyperparameter tuning script for the Bayesian breast cancer model.
Uses Optuna to find optimal parameters for draws, tune, and target_accept.
"""

import os
import sys
import logging
import time
import json
import optuna
from pathlib import Path

# Add the api directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname).1s] %(asctime)s %(name)s ▶ %(message)s'
)
logger = logging.getLogger(__name__)

def objective(trial):
    """
    Optuna objective function for Bayesian hyperparameter tuning.

    Returns:
        float: 1 - accuracy (Optuna minimizes, so we return 1 - accuracy)
    """
    try:
        from app.ml.builtin_trainers import train_breast_cancer_bayes
        import mlflow
        from mlflow.tracking import MlflowClient

        # Suggest hyperparameters
        draws = trial.suggest_int("draws", 200, 1200, step=100)
        tune = trial.suggest_int("tune", 100, 800, step=100)
        target_accept = trial.suggest_float("target_accept", 0.8, 0.98)

        logger.info(f"🧠 Trial {trial.number}: draws={draws}, tune={tune}, target_accept={target_accept:.3f}")

        # Train the model
        start_time = time.time()
        run_id = train_breast_cancer_bayes(
            draws=draws,
            tune=tune,
            target_accept=target_accept
        )
        elapsed = time.time() - start_time

        # Get accuracy from MLflow
        client = MlflowClient()
        run = client.get_run(run_id)
        accuracy = float(run.data.metrics["accuracy"])

        logger.info(f"✅ Trial {trial.number} completed: accuracy={accuracy:.3f}, time={elapsed:.1f}s")

        # Return 1 - accuracy (Optuna minimizes)
        return 1 - accuracy

    except Exception as e:
        logger.error(f"❌ Trial {trial.number} failed: {e}")
        # Return a high value to indicate this trial failed
        return 1.0

def tune_hyperparameters(n_trials=20, timeout=3600):
    """
    Run hyperparameter tuning using Optuna.

    Args:
        n_trials: Number of trials to run
        timeout: Maximum time in seconds

    Returns:
        dict: Best parameters found
    """
    logger.info(f"🚀 Starting hyperparameter tuning with {n_trials} trials...")

    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name="bayesian_breast_cancer_tuning"
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Log results
    logger.info(f"🎯 Best trial: {study.best_trial.number}")
    logger.info(f"   Best value: {study.best_trial.value:.4f}")
    logger.info(f"   Best params: {study.best_params}")

    return study.best_params

def test_best_parameters(params):
    """
    Test the best parameters found by Optuna.

    Args:
        params: Dictionary of best parameters
    """
    logger.info("🧪 Testing best parameters...")

    try:
        from app.ml.builtin_trainers import train_breast_cancer_bayes
        import mlflow
        from mlflow.tracking import MlflowClient

        # Train with best parameters
        start_time = time.time()
        run_id = train_breast_cancer_bayes(**params)
        elapsed = time.time() - start_time

        # Get final metrics
        client = MlflowClient()
        run = client.get_run(run_id)
        accuracy = float(run.data.metrics["accuracy"])

        logger.info(f"✅ Best parameters test completed:")
        logger.info(f"   Accuracy: {accuracy:.3f}")
        logger.info(f"   Time: {elapsed:.1f}s")
        logger.info(f"   Run ID: {run_id}")

        return True

    except Exception as e:
        logger.error(f"❌ Best parameters test failed: {e}")
        return False

def save_results(params, filename="best_bayesian_params.json"):
    """
    Save the best parameters to a JSON file.

    Args:
        params: Dictionary of best parameters
        filename: Output filename
    """
    output_path = Path(filename)

    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info(f"💾 Best parameters saved to {output_path}")

def main():
    """Main function."""
    logger.info("🚀 Starting Bayesian hyperparameter tuning...")

    # Test if the basic setup works
    try:
        from app.ml.utils import configure_pytensor_compiler, find_compiler

        cxx = find_compiler()
        if not cxx:
            logger.error("❌ No compiler found - cannot run Bayesian training")
            return 1

        if not configure_pytensor_compiler(cxx):
            logger.error("❌ PyTensor configuration failed - cannot run Bayesian training")
            return 1

        logger.info("✅ Basic setup verified")

    except Exception as e:
        logger.error(f"❌ Setup verification failed: {e}")
        return 1

    # Run hyperparameter tuning
    try:
        best_params = tune_hyperparameters(n_trials=10, timeout=1800)  # 30 minutes max

        # Test the best parameters
        if test_best_parameters(best_params):
            # Save results
            save_results(best_params)
            logger.info("🎉 Hyperparameter tuning completed successfully!")
            return 0
        else:
            logger.error("❌ Best parameters test failed")
            return 1

    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main()) 
