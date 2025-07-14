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
    Configure PyTensor to use a specific compiler with MSVC-safe flags.

    Args:
        compiler_path: Path to compiler executable. If None, will search for one.

    Returns:
        bool: True if compiler was configured successfully, False otherwise
    """
    try:
        import pytensor  # late import so function can be called very early
    except ImportError:
        log.warning("PyTensor not available – cannot configure compiler")
        return False

    # 1️⃣ Resolve the compiler path ------------------------------------------------
    if compiler_path is None:
        compiler_path = find_compiler()
    if compiler_path is None:
        log.warning("No compiler found – PyTensor will fall back to defaults")
        return False

    # 2️⃣ Write settings into PyTensor's global config -----------------------------
    system_is_windows = platform.system() == "Windows"
    basename = os.path.basename(compiler_path).lower()

    if system_is_windows:
        # Quote path so spaces in "Program Files (x86)" don't break the command line
        pytensor.config.cxx = f'"{compiler_path}"'

        # If this *is* MSVC, strip every GCC flag and substitute safe disables
        if "cl" in basename:
            # MSVC understands /wdXXXX but not -Wno-…  ➜  map the common ones
            pytensor.config.cxxflags = "/wd4100 /wd4244 /wd4267 /wd4996"
            log.info("✅ Configured MSVC with safe warning suppressions")
    else:
        pytensor.config.cxx = compiler_path  # GCC / Clang path is fine

    # 3️⃣ NUCLEAR OPTION: Blank ALL environment variables that PyTensor uses to inject flags
    # PyTensor checks these environment variables in multiple places
    flag_vars = [
        "PYTENSOR_FLAGS",
        "THEANO_FLAGS",  # Legacy but still checked
        "PYTENSOR_CXXFLAGS",
        "THEANO_CXXFLAGS",  # Legacy but still checked
    ]

    for var in flag_vars:
        os.environ[var] = "cxxflags="

    # 4️⃣ Additional PyTensor config overrides to prevent flag injection
    if system_is_windows and "cl" in basename:
        # Disable PyTensor's automatic flag injection
        pytensor.config.mode = "FAST_COMPILE"  # Avoid some optimizations that inject flags
        pytensor.config.optimizer = "fast_compile"  # Use simpler optimizer

        # Set additional config to prevent GCC flag injection
        pytensor.config.cmodule__compilation_warning = False
        pytensor.config.cmodule__warn_no_version = False

        # Force PyTensor to use our flags only
        pytensor.config.cxxflags = "/wd4100 /wd4244 /wd4267 /wd4996"

        log.info("🛡️ Applied nuclear option: disabled all GCC flag injection")

    # 5️⃣ Optional verbose diagnostics --------------------------------------------
    if os.getenv("DEBUG_COMPILER") == "1":
        log.debug("PyTensor.cxx      = %s", pytensor.config.cxx)
        log.debug("PyTensor.cxxflags = %s", getattr(pytensor.config, "cxxflags", ""))
        log.debug("PYTENSOR_FLAGS    = %s", os.getenv("PYTENSOR_FLAGS", "NOT_SET"))
        log.debug("THEANO_FLAGS      = %s", os.getenv("THEANO_FLAGS", "NOT_SET"))

    log.info("🛠 PyTensor now uses compiler: %s", compiler_path)
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
