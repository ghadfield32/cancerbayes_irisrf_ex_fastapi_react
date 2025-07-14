#!/usr/bin/env python3
"""
Test script for the self-healing model system.
This script tests the new startup pattern and status tracking.
"""

import asyncio
import requests
import time
import json
import os
import shutil
import subprocess
import sys
import pathlib
from typing import Dict, Any

def start_backend():
    """
    Launch uvicorn in a subprocess, stream its output in real time and
    fail fast if it crashes or if port 8000 is already taken.
    """
    import socket, threading, os, pathlib, sys, subprocess, time, shutil

    # ── quick port-availability probe ──────────────────────────────
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Port 8000 already in use – aborting tests")

    print("🚀  Spawning backend …")

    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        "api.app.main:app",
        "--port", "8000",
        "--env-file", ".env",
        "--log-level", "info",
    ]

    env = os.environ.copy()
    # Ensure the project root is on PYTHONPATH so 'api' is importable
    env["PYTHONPATH"] = str(pathlib.Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        uvicorn_cmd,
        cwd=pathlib.Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,          # avoid Zombie children on ^C
        env=env,
    )

    # ── stream output in background thread ─────────────────────────
    def _pump(pipe):
        for ln in iter(pipe.readline, ""):
            print("[backend]", ln.rstrip())

    t = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    t.start()

    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Backend process exited early – see above log")

        try:
            import requests
            r = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=1)
            if r.status_code == 200:
                print("✅ Backend responded to /health")
                return proc
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(1)

    proc.terminate()
    raise RuntimeError("Backend did not become healthy within 60 seconds")

def cleanup_mlruns():
    """Remove mlruns directory to simulate fresh start."""
    mlruns_path = "mlruns"
    if os.path.exists(mlruns_path):
        print("🧹 Cleaning up mlruns directory...")
        shutil.rmtree(mlruns_path)
        print("✅ Cleanup complete")

def test_backend_startup():
    """Test that the backend starts up immediately."""
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

    # Test readiness endpoint - should be ready immediately
    try:
        ready_response = requests.get("http://localhost:8000/api/v1/ready")
        if ready_response.status_code == 200:
            ready_data = ready_response.json()
            if ready_data.get("ready"):
                print("✅ Backend is ready for requests immediately")
            else:
                print("❌ Backend not ready")
                return False
        else:
            print(f"❌ Readiness endpoint failed: {ready_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing readiness: {e}")
        return False

    return True

def test_model_status_evolution():
    """Test that model status evolves correctly over time."""
    print("\n🔍 Testing model status evolution...")

    status_history = []
    max_polls = 30  # Poll for up to 60 seconds

    for i in range(max_polls):
        try:
            response = requests.get("http://localhost:8000/api/v1/ready/full")
            if response.status_code == 200:
                status_data = response.json()
                status_history.append(status_data)

                print(f"Poll {i+1}: Ready={status_data.get('ready')}, "
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

def test_login_immediate():
    """Test that login works immediately even when models are training."""
    print("\n🔍 Testing immediate login capability...")

    try:
        # Try to get a token immediately after startup
        token_response = requests.post(
            "http://localhost:8000/api/v1/token",
            data={"username": "alice", "password": "supersecretvalue"}
        )

        if token_response.status_code == 200:
            token_data = token_response.json()
            print("✅ Login successful immediately after startup")
            return True
        else:
            print(f"❌ Login failed: {token_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing login: {e}")
        return False

def test_prediction_with_training_models():
    """Test prediction behavior when models are still training."""
    print("\n🔍 Testing prediction behavior during training...")

    try:
        # Get a token
        token_response = requests.post(
            "http://localhost:8000/api/v1/token",
            data={"username": "alice", "password": "supersecretvalue"}
        )

        if token_response.status_code != 200:
            print("❌ Failed to get authentication token")
            return False

        token_data = token_response.json()
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}

        # Try iris prediction (should work if model is loaded)
        iris_data = {
            "model_type": "rf",
            "samples": [{
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }]
        }

        iris_response = requests.post(
            "http://localhost:8000/api/v1/iris/predict",
            json=iris_data,
            headers=headers
        )

        if iris_response.status_code == 200:
            iris_result = iris_response.json()
            print(f"✅ Iris prediction successful: {iris_result['predictions']}")
        elif iris_response.status_code == 503:
            print("✅ Iris prediction correctly rejected (model still training)")
        else:
            print(f"❌ Unexpected iris response: {iris_response.status_code}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing predictions: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting self-healing system tests...\n")

    # Optional: Clean up for fresh start
    if input("Clean up mlruns directory for fresh start? (y/N): ").lower() == 'y':
        cleanup_mlruns()

    # 🔑 NEW: Launch backend in-process
    backend = start_backend()
    try:
        # Test 1: Backend startup
        if not test_backend_startup():
            print("❌ Backend startup test failed")
            return

        # Test 2: Immediate login
        if not test_login_immediate():
            print("❌ Immediate login test failed")
            return

        # Test 3: Model status evolution
        if not test_model_status_evolution():
            print("❌ Model status evolution test failed")
            return

        # Test 4: Prediction behavior during training
        if not test_prediction_with_training_models():
            print("❌ Prediction behavior test failed")
            return

        print("\n🎉 All tests passed! The self-healing system is working correctly.")
        print("\n📋 Summary:")
        print("✅ Backend starts immediately")
        print("✅ Login works immediately")
        print("✅ Models train in background")
        print("✅ Status updates in real-time")
        print("✅ Predictions work when models are ready")

    finally:
        # Clean shutdown
        print("\n🛑 Shutting down backend...")
        backend.terminate()
        backend.wait(timeout=10)
        print("✅ Backend shutdown complete")

if __name__ == "__main__":
    main() 
