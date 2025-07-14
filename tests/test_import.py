#!/usr/bin/env python3
"""
Enhanced test to verify that api.app.main can be imported without errors.
Captures detailed logs and implements unit test mode for fast imports.
"""

import importlib
import io
import logging
import os
import sys
import time
import traceback

def setup_test_environment():
    """Configure environment for fast, safe imports."""
    print("🔧 Setting up test environment...")

    # Tell the backend we are in unit-test-mode BEFORE we touch it
    os.environ["UNIT_TESTING"] = "1"
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file://./mlruns_tests")

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    print("✅ Test environment configured")

def capture_import_logs():
    """Capture all logs during import for debugging."""
    print("📝 Setting up log capture...")

    # Create a string buffer to capture all logs
    log_stream = io.StringIO()

    # Configure logging to capture everything
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(log_stream),
            logging.StreamHandler(sys.stdout)  # Also show in console
        ],
        force=True  # Override any existing config
    )

    return log_stream

def test_import_with_timing():
    """Test that the main module can be imported with timing and detailed logs."""
    print("🔍 Testing api.app.main import...")

    # Capture logs during import
    log_stream = capture_import_logs()

    # Time the import
    t0 = time.perf_counter()

    try:
        # Add the current directory to Python path
        sys.path.insert(0, os.getcwd())

        # Try to import the main module
        import api.app.main
        dt = time.perf_counter() - t0

        print(f"✅ api.app.main imported successfully in {dt:.3f}s")

        # Show captured logs if any
        log_content = log_stream.getvalue()
        if log_content.strip():
            print("📋 Import logs:")
            print(log_content)

        return True

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"❌ Import failed after {dt:.3f}s")
        print(f"❌ Error: {e}")

        # Show captured logs
        log_content = log_stream.getvalue()
        if log_content.strip():
            print("📋 Logs during failed import:")
            print(log_content)

        print("📋 Full traceback:")
        traceback.print_exc()
        return False

def test_logs_directory():
    """Test that logs directory exists."""
    print("🔍 Testing logs directory...")

    logs_dir = "logs"
    if os.path.exists(logs_dir):
        print(f"✅ Logs directory exists: {logs_dir}")
        return True
    else:
        print(f"❌ Logs directory missing: {logs_dir}")
        return False

def test_mlflow_config():
    """Test MLflow configuration."""
    print("🔍 Testing MLflow configuration...")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "not set")
    unit_testing = os.environ.get("UNIT_TESTING", "not set")

    print(f"✅ MLFLOW_TRACKING_URI: {tracking_uri}")
    print(f"✅ UNIT_TESTING: {unit_testing}")

    return True

def test_compiler_probe():
    """Test compiler detection functionality."""
    print("🔍 Testing compiler probe...")

    try:
        from api.app.ml.utils import find_compiler, test_compiler_availability

        # Test compiler availability
        compilers = test_compiler_availability()
        print(f"✅ Compiler test completed: {sum(compilers.values())}/{len(compilers)} available")

        # Test find_compiler
        compiler_path = find_compiler()
        if compiler_path:
            print(f"✅ Found compiler: {compiler_path}")
        else:
            print("⚠️ No compiler found (expected on CI or dev machines without build tools)")

        return True

    except Exception as e:
        print(f"❌ Compiler probe test failed: {e}")
        return False

def main():
    """Run comprehensive import tests."""
    print("🚀 Testing module imports with detailed diagnostics...\n")

    success = True

    # Test 1: Setup environment
    setup_test_environment()

    # Test 2: Logs directory
    if not test_logs_directory():
        success = False

    # Test 3: MLflow config
    if not test_mlflow_config():
        success = False

    # Test 4: Compiler probe
    if not test_compiler_probe():
        success = False

    # Test 5: Main module import (with timing and logs)
    if not test_import_with_timing():
        success = False

    if success:
        print("\n🎉 All import tests passed!")
    else:
        print("\n❌ Some import tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 
