#!/usr/bin/env python3
"""
GPU environment configuration script for JAX/NumPyro optimization.
Sets up environment variables and tests GPU availability.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname).1s] %(asctime)s %(name)s ▶ %(message)s'
)
logger = logging.getLogger(__name__)

def configure_jax_gpu():
    """
    Configure JAX for optimal GPU performance.

    Returns:
        bool: True if GPU is available and configured
    """
    logger.info("🔧 Configuring JAX GPU environment...")

    # Strip invalid flags that crash JAX
    os.environ.pop("XLA_FLAGS", None)  # defensive – avoid "--"

    # Set JAX environment variables for GPU optimization
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # Use 95% of GPU memory
    os.environ["JAX_PLATFORM_NAME"] = "gpu"

    # Additional optimizations
    os.environ["JAX_ENABLE_X64"] = "false"  # Use float32 for speed
    os.environ["JAX_DISABLE_JIT"] = "false"  # Enable JIT compilation

    logger.info("✅ JAX GPU environment variables set")
    return True

def test_gpu_availability():
    """
    Test if GPU is available and working with JAX.

    Returns:
        bool: True if GPU is available
    """
    try:
        import jax
        import jax.numpy as jnp

        # Test basic JAX functionality
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)
        logger.info(f"✅ JAX basic functionality: {y}")

        # Check available devices
        devices = jax.devices()
        logger.info(f"🔍 Available JAX devices: {len(devices)}")

        for i, device in enumerate(devices):
            logger.info(f"   Device {i}: {device}")

        # Check if GPU is available
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        if gpu_devices:
            logger.info(f"✅ GPU available: {len(gpu_devices)} device(s)")
            return True
        else:
            logger.warning("⚠️ No GPU devices found - will use CPU")
            return False

    except ImportError:
        logger.error("❌ JAX not available")
        return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False

def test_numpyro_gpu():
    """
    Test NumPyro GPU functionality.

    Returns:
        bool: True if NumPyro GPU works
    """
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        logger.info("🧪 Testing NumPyro GPU functionality...")

        # Simple NumPyro model test
        def model():
            x = numpyro.sample("x", dist.Normal(0, 1))
            numpyro.sample("y", dist.Normal(x, 1), obs=0.0)

        # Run MCMC with NumPyro
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100)
        mcmc.run(jax.random.PRNGKey(0))

        logger.info("✅ NumPyro GPU test passed")
        return True

    except Exception as e:
        logger.error(f"❌ NumPyro GPU test failed: {e}")
        return False

def optimize_bayesian_parameters():
    """
    Optimize Bayesian model parameters for GPU.

    Returns:
        dict: Optimized parameters
    """
    logger.info("🎯 Optimizing Bayesian parameters for GPU...")

    # GPU-optimized parameters
    params = {
        "draws": 1000,        # More draws for better convergence
        "tune": 500,          # More tuning for better mixing
        "target_accept": 0.95, # Higher acceptance rate for GPU
        "nuts_sampler": "numpyro"  # Use NumPyro backend
    }

    logger.info(f"✅ GPU-optimized parameters: {params}")
    return params

def create_gpu_config_file():
    """
    Create a configuration file with GPU-optimized settings.
    """
    config = {
        "jax_gpu_settings": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_ENABLE_X64": "false",
            "JAX_DISABLE_JIT": "false"
        },
        "bayesian_parameters": {
            "draws": 1000,
            "tune": 500,
            "target_accept": 0.95,
            "nuts_sampler": "numpyro"
        },
        "pytensor_settings": {
            "optimizer": "fast_compile",
            "mode": "FAST_COMPILE"
        }
    }

    import json
    config_path = Path("gpu_config.json")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"💾 GPU configuration saved to {config_path}")
    return config

def main():
    """Main function."""
    logger.info("🚀 Starting GPU environment configuration...")

    # Step 1: Configure JAX GPU
    if not configure_jax_gpu():
        logger.error("❌ JAX GPU configuration failed")
        return 1

    # Step 2: Test GPU availability
    gpu_available = test_gpu_availability()

    # Step 3: Test NumPyro if GPU is available
    if gpu_available:
        if not test_numpyro_gpu():
            logger.warning("⚠️ NumPyro GPU test failed - will use CPU fallback")
            gpu_available = False

    # Step 4: Optimize parameters
    params = optimize_bayesian_parameters()

    # Step 5: Create configuration file
    config = create_gpu_config_file()

    # Summary
    logger.info("📊 GPU Configuration Summary:")
    logger.info(f"   GPU Available: {gpu_available}")
    logger.info(f"   JAX Configured: True")
    logger.info(f"   NumPyro GPU: {gpu_available}")
    logger.info(f"   Optimized Parameters: {params}")

    if gpu_available:
        logger.info("🎉 GPU environment configured successfully!")
        return 0
    else:
        logger.warning("⚠️ GPU not available - will use CPU fallback")
        return 0  # Not a critical failure

if __name__ == "__main__":
    exit(main()) 
