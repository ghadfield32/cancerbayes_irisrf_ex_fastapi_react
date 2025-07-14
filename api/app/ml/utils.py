"""
Utility functions for ML model training and compiler detection.
"""

import shutil
import logging
import os
import subprocess
import shlex
import platform
import glob
import sys

log = logging.getLogger(__name__)

def find_compiler() -> str | None:
    """
    Return absolute path to a working C/C++ compiler or None.

    Searches in order:
    1. Explicit override via PYTENSOR_CXX environment variable
    2. Common compiler names (g++, gcc, cl.exe, cl)
    3. Windows Visual Studio BuildTools typical location (last resort)

    Returns:
        str | None: Absolute path to compiler executable, or None if not found
    """
    # 1️⃣ explicit override via env
    override = os.getenv("PYTENSOR_CXX")
    if override and shutil.which(override):
        log.info(f"Using compiler from PYTENSOR_CXX: {override}")
        return override

    # 2️⃣ try common names
    for exe in ("g++", "gcc", "cl.exe", "cl"):
        path = shutil.which(exe)
        if path:
            log.info(f"Found compiler: {path}")
            return path

    # 3️⃣ Windows VS BuildTools typical location (last resort)
    if platform.system() == "Windows":
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            try:
                log.debug("Probing for Visual Studio BuildTools via vswhere...")
                out = subprocess.check_output(
                    [vswhere, "-latest", "-products", "*", "-requires", 
                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", 
                     "-property", "installationPath"],
                    text=True,
                    timeout=5,
                ).strip()

                if out:
                    # Look for cl.exe in the typical location
                    cand = rf"{out}\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
                    matches = glob.glob(cand)
                    if matches:
                        log.info(f"Found Visual Studio compiler: {matches[0]}")
                        return matches[0]
                    else:
                        log.debug(f"VS installation found at {out} but cl.exe not found")
                else:
                    log.debug("vswhere found no Visual Studio installations")

            except subprocess.TimeoutExpired:
                log.debug("vswhere probe timed out")
            except subprocess.CalledProcessError as exc:
                log.debug(f"vswhere probe failed with return code {exc.returncode}")
            except Exception as exc:
                log.debug(f"vswhere probe failed: {exc}")

    log.warning("No C/C++ compiler found")
    return None

def configure_pytensor_compiler(compiler_path: str | None = None) -> bool:
    """
    Configure PyTensor for ahead-of-time C-thunks *before* its first import.
    Safe to call repeatedly; turns into a no-op if PyTensor is already live.

    Returns
    -------
    bool
        True when a compiler was configured (or already present); False when
        we had to fall back to default (pure-python ops).
    """
    import sys, os, platform, shutil, logging, subprocess, glob
    log = logging.getLogger(__name__)

    # ── CASE 1: PyTensor already imported – try a *runtime* tweak ────────────
    if "pytensor" in sys.modules:
        import pytensor
        if hasattr(pytensor, "change_flags"):
            try:
                pytensor.change_flags(optimizer="fast_compile")
                log.debug("PyTensor already initialised – changed flags (runtime)")
            except Exception as exc:                       # noqa: broad-except
                log.warning("PyTensor change_flags failed → %s", exc)
        else:
            log.debug("PyTensor.change_flags missing – probably ≤2.13 build")
        return True  # Either way, nothing else to do

    # ── CASE 2: module not yet imported – prepare env vars first ─────────────
    # Use find_compiler directly (it's in the same module)
    compiler_path = compiler_path or find_compiler()
    if not compiler_path:
        log.warning("No C/C++ compiler available – PyTensor will be pure-python.")
        # Set empty cxx to silence warnings
        os.environ["PYTENSOR_FLAGS"] = "cxx=,optimizer=fast_compile"
        return False

    system_is_windows = platform.system() == "Windows"

    # ── CRITICAL: Clean up problematic environment variables ───────────────────
    # Remove XLA_FLAGS that contains invalid "--" value
    if os.getenv("XLA_FLAGS") == "--":
        os.environ.pop("XLA_FLAGS", None)
        log.info("🧹 Cleaned up invalid XLA_FLAGS")

    # Clear any existing PyTensor environment variables to avoid conflicts
    pytensor_env_vars = ["PYTENSOR_FLAGS", "PYTENSOR_CXX", "PYTENSOR_CXXFLAGS", 
                        "THEANO_FLAGS", "THEANO_CXXFLAGS"]
    for var in pytensor_env_vars:
        if var in os.environ:
            del os.environ[var]

    # ── Set up PyTensor environment variables ─────────────────────────────────
    if system_is_windows and "cl" in os.path.basename(compiler_path).lower():
        # MSVC configuration
        os.environ["PYTENSOR_CXX"] = f'"{compiler_path}"'
        os.environ["PYTENSOR_CXXFLAGS"] = "/wd4100 /wd4244 /wd4267 /wd4996"
        os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_compile"
        log.info(f"🔧 Configured MSVC: {compiler_path}")
    else:
        # GCC/Clang configuration
        os.environ["PYTENSOR_CXX"] = compiler_path
        os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_compile"
        log.info(f"🔧 Configured GCC/Clang: {compiler_path}")

    # ── Import now: env vars take effect ─────────────────────────────────────
    import pytensor  # noqa: E402
    log.info("PyTensor configured with %s (fast_compile, device=%s)",
             compiler_path, pytensor.config.device)
    return True

def test_compiler_availability() -> dict:
    """
    Test what compilers are available on the system.

    Returns:
        dict: Mapping of compiler names to availability status
    """
    compilers = ["g++", "gcc", "cl.exe", "cl"]
    available = {}

    for compiler in compilers:
        try:
            result = subprocess.run([compiler, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            available[compiler] = result.returncode == 0
            if result.returncode == 0:
                log.info(f"✅ {compiler}: Available")
                log.debug(f"   Version: {result.stdout.split()[0] if result.stdout else 'Unknown'}")
            else:
                log.debug(f"❌ {compiler}: Not available (return code: {result.returncode})")
        except FileNotFoundError:
            log.debug(f"❌ {compiler}: Not found")
            available[compiler] = False
        except subprocess.TimeoutExpired:
            log.debug(f"⏰ {compiler}: Timeout")
            available[compiler] = False
        except Exception as e:
            log.debug(f"❌ {compiler}: Error - {e}")
            available[compiler] = False

    return available 
