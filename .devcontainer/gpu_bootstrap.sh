#!/usr/bin/env bash
set -euxo pipefail

echo "[gpu-bootstrap] BEGIN"

PY="/app/.venv/bin/python"
export UV_PROJECT_ENVIRONMENT="/app/.venv"

_have_cmd() { command -v "$1" >/dev/null 2>&1; }

seed_pip() {
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[gpu-bootstrap] Seeding pip into /app/.venv (ensurepip)"
    "$PY" -m ensurepip --upgrade || true
  fi
}

PIP() {
  if _have_cmd uv; then
    echo "[gpu-bootstrap] Using: uv pip $*"
    uv pip "$@"
  else
    seed_pip
    echo "[gpu-bootstrap] Using: $PY -m pip $*"
    "$PY" -m pip "$@"
  fi
}

PIP_SHOW() {
  if _have_cmd uv; then uv pip show "$@" || true; else seed_pip; "$PY" -m pip show "$@" || true; fi
}

pick_torch_index() {
  # Derive Torch CUDA wheel index from CUDA_TAG (defaults are safe)
  local tag="${CUDA_TAG:-12.8.0}"
  case "$tag" in
    12.1* ) echo "cu121" ;;
    12.4* ) echo "cu124" ;;
    12.5*|12.6*|12.7*|12.8*|12.9* ) echo "cu126" ;;
    * ) echo "cu126" ;;
  esac
}

echo "[gpu-bootstrap] whoami=$(whoami)"
echo "[gpu-bootstrap] PY=$PY"
echo "[gpu-bootstrap] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
$PY - <<'PY'
import sys, os
print("[gpu-bootstrap] sys.executable:", sys.executable)
print("[gpu-bootstrap] sys.prefix:", sys.prefix)
print("[gpu-bootstrap] VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV","<unset>"))
for m in ("jax","jaxlib","torch"):
    try:
        mod = __import__(m)
        print(f"[gpu-bootstrap] pre: import {m}: OK from", getattr(mod,"__file__","?"))
    except Exception as e:
        print(f"[gpu-bootstrap] pre: import {m}: FAIL -> {e.__class__.__name__}: {e}")
PY

if _have_cmd nvidia-smi; then
  echo "[gpu-bootstrap] nvidia-smi present"
  nvidia-smi || true
else
  echo "[gpu-bootstrap] nvidia-smi NOT present"
fi

unset JAX_PLATFORM_NAME || true

# --- 1) Ensure PyTorch (GPU) -------------------------------------------------
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[gpu-bootstrap] PyTorch with CUDA already present"
else
  if _have_cmd nvidia-smi; then
    IDX="$(pick_torch_index)"
    echo "[gpu-bootstrap] Installing PyTorch (${IDX}) wheels into /app/.venv"
    PIP install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${IDX}" || true
  else
    echo "[gpu-bootstrap] Skipping PyTorch GPU install (no nvidia-smi)"
  fi
fi

# --- 2) Ensure JAX (GPU) *after* Torch so final cuDNN >= 9.8 -----------------
JAX_OK=1
if ! $PY - <<'PY'
import sys
try:
    import jax
    devs=jax.devices()
    ok=any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs)
    print("[gpu-bootstrap] JAX devices pre:", devs)
    sys.exit(0 if ok else 1)
except Exception as e:
    print("[gpu-bootstrap] JAX import/probe error (pre):", e)
    sys.exit(2)
PY
then
  JAX_OK=0
fi

if [ "$JAX_OK" -ne 1 ] && _have_cmd nvidia-smi; then
  echo "[gpu-bootstrap] Installing/repairing JAX CUDA wheels into /app/.venv"
  # JAX 0.6.0 requires cuDNN >= 9.8; this will install the plugin + nvidia-* libs
  PIP install --no-cache-dir "jax[cuda12]==0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || true

  # Re-probe
  $PY - <<'PY'
import jax
print("[gpu-bootstrap] JAX:", jax.__version__, "devices:", jax.devices())
PY
fi

echo "[gpu-bootstrap] pip/uv show (post-install)"
PIP_SHOW jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt torch torchvision torchaudio || true

# Final snapshot
$PY - <<'PY'
for m in ("torch","jax","jaxlib"):
    try:
        mod = __import__(m)
        print(f"[gpu-bootstrap] post: import {m}: OK from", getattr(mod, "__file__", "?"))
    except Exception as e:
        print(f"[gpu-bootstrap] post: import {m}: FAIL -> {e.__class__.__name__}: {e}")
try:
    import jax
    print("[gpu-bootstrap] FINAL devices:", jax.devices())
except Exception as e:
    print("[gpu-bootstrap] FINAL devices probe error:", e)
PY

echo "[gpu-bootstrap] END"
