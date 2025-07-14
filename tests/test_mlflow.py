import mlflow
from api.app.ml.builtin_trainers import train_iris_random_forest, train_breast_cancer_bayes

def test_mlflow():
    print("Setting up MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("ml_fullstack_models")

    print("\nTraining Iris Random Forest...")
    iris_run_id = train_iris_random_forest()
    print(f"Iris model trained. Run ID: {iris_run_id}")

    print("\nTraining Breast Cancer Bayesian Model...")
    cancer_run_id = train_breast_cancer_bayes()
    print(f"Cancer model trained. Run ID: {cancer_run_id}")

    print("\nLoading models...")
    iris_model = mlflow.pyfunc.load_model(f"models:/iris_random_forest/Production")
    print("Iris model loaded successfully")

    cancer_model = mlflow.pyfunc.load_model(f"models:/breast_cancer_bayes/Production")
    print("Cancer model loaded successfully")

if __name__ == "__main__":
    test_mlflow() 
