#!/usr/bin/env python3
import sys, importlib, os, textwrap

CRIT = []
WARN = []

def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

def _msg_box(title: str, body: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}\n{body}\n")

def _probe_jax() -> None:
    """
    Probe JAX availability and devices without relying on private internals.
    Adds context about import paths and provides actionable, uv-friendly hints.
    """
    import importlib, os, textwrap

    jp = os.environ.get("JAX_PLATFORM_NAME", "<unset>")
    print(f"   JAX_PLATFORM_NAME: {jp}")

    try:
        jax = importlib.import_module("jax")
    except Exception as e:
        WARN.append(f"jax not importable: {e!r}")
        _msg_box(
            "Action: JAX not importable",
            "• Ensure JAX is installed into /app/.venv via: uv pip install jax\n"
            "• Avoid using bare 'pip'; prefer 'uv pip' with UV_PROJECT_ENVIRONMENT=/app/.venv ."
        )
        return

    # Identify import locations
    try:
        jaxlib = importlib.import_module("jaxlib")
        print(f"   jax: {getattr(jax,'__version__','?')} @ {getattr(jax,'__file__','?')}")
        print(f"   jaxlib @ {getattr(jaxlib,'__file__','?')}")
    except Exception as e:
        WARN.append(f"jaxlib import failed: {e!r}")

    # Device probe (no private APIs)
    try:
        devs = jax.devices()
        print(f"   jax {getattr(jax,'__version__','?')} devices: {devs}")
    except Exception as e:
        WARN.append(f"jax.devices() raised: {e!r}")
        _msg_box(
            "Action: Fix JAX GPU backend",
            textwrap.dedent("""\
                • If you intend to use GPU with CUDA, install the PJRT CUDA plugin:
                  uv pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
                • Ensure installs target /app/.venv (use 'uv pip', not bare 'pip').
            """),
        )
        return

    # CPU-only hints
    gpu = [d for d in devs if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
    if not gpu:
        WARN.append("JAX imported but reports CPU-only devices.")
        _msg_box(
            "Info: JAX is CPU-only right now",
            textwrap.dedent("""\
                Likely causes (check in order):
                1) CUDA plugin not installed in this venv:
                   uv pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
                2) Plugin/driver/CUDA version mismatch (common on very new drivers):
                   uv pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
                3) Conflicting libraries from other frameworks in a different site-packages.
            """),
        )


def main():
    print("## Python & library diagnostics ##")
    print("Python:", sys.version.split()[0])
    print("sys.executable:", sys.executable)
    print("sys.prefix:", sys.prefix)
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV","<unset>"))
    print("PATH head:", os.environ.get("PATH","").split(":")[:3])

    if not sys.executable.startswith("/app/.venv/"):
        CRIT.append("Interpreter is not /app/.venv — uv env not active for this process")

    jlab_ok = _have("jupyterlab")
    print(f" - jupyterlab: {'OK' if jlab_ok else 'MISSING'}")

    torch_ok = _have("torch")
    print(f" - torch: {'OK' if torch_ok else 'MISSING'}")
    if torch_ok:
        try:
            import torch
            print("   torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
        except Exception as e:
            WARN.append(f"torch import ok but CUDA probe errored: {e}")

    print(f" - jax: {'OK' if _have('jax') else 'MISSING'}")
    _probe_jax()

    if CRIT:
        _msg_box("Critical failures", "\n".join(f"• {m}" for m in CRIT))
        sys.exit(1)

    if WARN:
        _msg_box("Warnings (non-blocking)", "\n".join(f"• {m}" for m in WARN))

    print("✅ verify_env completed (warnings above are informational).")
    sys.exit(0)

if __name__ == "__main__":
    main()
