#!/usr/bin/env python3
"""
Manual test script for the self-healing system.
This script assumes the backend is already running (e.g., via npm run dev).
"""

import requests
import time
import json

def test_backend_status():
    """Test basic backend status."""
    print("🔍 Testing backend status...")

    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/api/v1/health")
        if health_response.status_code == 200:
            print("✅ Backend health check passed")
        else:
            print(f"❌ Backend health failed: {health_response.status_code}")
            return False

        # Test readiness endpoint
        ready_response = requests.get("http://localhost:8000/api/v1/ready")
        if ready_response.status_code == 200:
            ready_data = ready_response.json()
            print(f"✅ Backend ready: {ready_data}")
        else:
            print(f"❌ Readiness check failed: {ready_response.status_code}")
            return False

        return True

    except requests.exceptions.ConnectionError:
        print("❌ Backend not running. Start it with: npm run dev")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_status():
    """Test model status endpoint."""
    print("\n🔍 Testing model status...")

    try:
        response = requests.get("http://localhost:8000/api/v1/ready/full")
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Model status: {json.dumps(status_data, indent=2)}")
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_login():
    """Test login functionality."""
    print("\n🔍 Testing login...")

    try:
        token_response = requests.post(
            "http://localhost:8000/api/v1/token",
            data={"username": "alice", "password": "supersecretvalue"}
        )

        if token_response.status_code == 200:
            token_data = token_response.json()
            print("✅ Login successful")
            return token_data['access_token']
        else:
            print(f"❌ Login failed: {token_response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_prediction(token):
    """Test prediction with authentication."""
    print("\n🔍 Testing prediction...")

    if not token:
        print("❌ No token available")
        return False

    headers = {"Authorization": f"Bearer {token}"}

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
        iris_response = requests.post(
            "http://localhost:8000/api/v1/iris/predict",
            json=iris_data,
            headers=headers
        )

        if iris_response.status_code == 200:
            iris_result = iris_response.json()
            print(f"✅ Iris prediction: {iris_result['predictions']}")
        elif iris_response.status_code == 503:
            print("✅ Iris prediction rejected (model still training)")
        else:
            print(f"❌ Iris prediction failed: {iris_response.status_code}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run manual tests."""
    print("🚀 Manual self-healing system tests\n")
    print("Make sure the backend is running with: npm run dev\n")

    # Test 1: Backend status
    if not test_backend_status():
        print("\n❌ Backend status test failed")
        return

    # Test 2: Model status
    if not test_model_status():
        print("\n❌ Model status test failed")
        return

    # Test 3: Login
    token = test_login()
    if not token:
        print("\n❌ Login test failed")
        return

    # Test 4: Prediction
    if not test_prediction(token):
        print("\n❌ Prediction test failed")
        return

    print("\n🎉 All manual tests passed!")
    print("\n📋 Summary:")
    print("✅ Backend is running and responsive")
    print("✅ Model status is being tracked")
    print("✅ Login works with authentication")
    print("✅ Predictions work (or are properly rejected)")

if __name__ == "__main__":
    main() 
