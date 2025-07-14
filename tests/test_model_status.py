#!/usr/bin/env python3
"""
Test script to verify the new model status system.
This script tests the backend startup, model loading, and status endpoints.
"""

import asyncio
import requests
import time
import json
from typing import Dict, Any

def test_backend_startup():
    """Test that the backend starts up correctly and provides status."""
    print("🔍 Testing backend startup...")

    # Wait for backend to be ready
    max_wait = 30
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            # Test basic health
            health_response = requests.get("http://localhost:8000/api/v1/health")
            if health_response.status_code == 200:
                print("✅ Backend health check passed")
                break
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for backend to start...")
            time.sleep(2)
    else:
        print("❌ Backend failed to start within 30 seconds")
        return False

    # Test readiness endpoint
    try:
        ready_response = requests.get("http://localhost:8000/api/v1/ready/full")
        if ready_response.status_code == 200:
            status_data = ready_response.json()
            print(f"✅ Readiness endpoint working: {json.dumps(status_data, indent=2)}")
            return True
        else:
            print(f"❌ Readiness endpoint failed: {ready_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing readiness: {e}")
        return False

def test_model_status_polling():
    """Test that model status updates correctly over time."""
    print("\n🔍 Testing model status polling...")

    status_history = []
    max_polls = 15  # Poll for up to 30 seconds

    for i in range(max_polls):
        try:
            response = requests.get("http://localhost:8000/api/v1/ready/full")
            if response.status_code == 200:
                status_data = response.json()
                status_history.append(status_data)

                print(f"Poll {i+1}: API ready={status_data.get('api_ready')}, "
                      f"All loaded={status_data.get('all_models_loaded')}")

                # Show individual model status
                model_status = status_data.get('model_status', {})
                for model, status in model_status.items():
                    print(f"  {model}: {status}")

                # Check if all models are loaded
                if status_data.get('all_models_loaded'):
                    print("✅ All models loaded successfully!")
                    return True

            time.sleep(2)
        except Exception as e:
            print(f"❌ Error polling status: {e}")
            time.sleep(2)

    print("❌ Models did not load within expected time")
    return False

def test_prediction_endpoints():
    """Test that prediction endpoints work when models are loaded."""
    print("\n🔍 Testing prediction endpoints...")

    # Test iris prediction
    iris_data = {
        "model_type": "rf",
        "samples": [{
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }]
    }

    try:
        # First get a token
        token_response = requests.post(
            "http://localhost:8000/api/v1/token",
            data={"username": "alice", "password": "supersecretvalue"}
        )

        if token_response.status_code != 200:
            print("❌ Failed to get authentication token")
            return False

        token_data = token_response.json()
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}

        # Test iris prediction
        iris_response = requests.post(
            "http://localhost:8000/api/v1/iris/predict",
            json=iris_data,
            headers=headers
        )

        if iris_response.status_code == 200:
            iris_result = iris_response.json()
            print(f"✅ Iris prediction successful: {iris_result['predictions']}")
        else:
            print(f"❌ Iris prediction failed: {iris_response.status_code}")
            return False

        # Test cancer prediction
        cancer_data = {
            "model_type": "bayes",
            "samples": [{
                "mean_radius": 14.13,
                "mean_texture": 19.26,
                "mean_perimeter": 91.97,
                "mean_area": 654.89,
                "mean_smoothness": 0.096,
                "mean_compactness": 0.104,
                "mean_concavity": 0.089,
                "mean_concave_points": 0.048,
                "mean_symmetry": 0.181,
                "mean_fractal_dimension": 0.063,
                "se_radius": 0.406,
                "se_texture": 1.216,
                "se_perimeter": 2.866,
                "se_area": 40.34,
                "se_smoothness": 0.007,
                "se_compactness": 0.025,
                "se_concavity": 0.032,
                "se_concave_points": 0.012,
                "se_symmetry": 0.020,
                "se_fractal_dimension": 0.004,
                "worst_radius": 16.27,
                "worst_texture": 25.68,
                "worst_perimeter": 107.26,
                "worst_area": 880.58,
                "worst_smoothness": 0.132,
                "worst_compactness": 0.254,
                "worst_concavity": 0.273,
                "worst_concave_points": 0.114,
                "worst_symmetry": 0.290,
                "worst_fractal_dimension": 0.084
            }]
        }

        cancer_response = requests.post(
            "http://localhost:8000/api/v1/cancer/predict",
            json=cancer_data,
            headers=headers
        )

        if cancer_response.status_code == 200:
            cancer_result = cancer_response.json()
            print(f"✅ Cancer prediction successful: {cancer_result['predictions']}")
            return True
        else:
            print(f"❌ Cancer prediction failed: {cancer_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing predictions: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting model status system tests...\n")

    # Test 1: Backend startup
    if not test_backend_startup():
        print("❌ Backend startup test failed")
        return

    # Test 2: Model status polling
    if not test_model_status_polling():
        print("❌ Model status polling test failed")
        return

    # Test 3: Prediction endpoints
    if not test_prediction_endpoints():
        print("❌ Prediction endpoints test failed")
        return

    print("\n🎉 All tests passed! The model status system is working correctly.")

if __name__ == "__main__":
    main() 
