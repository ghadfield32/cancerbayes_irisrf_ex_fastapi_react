# .devcontainer/Dockerfile — image is built from .devcontainer context only
ARG CUDA_TAG=12.8.0
FROM nvidia/cuda:${CUDA_TAG}-cudnn-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=cancer_bayes_iris_fastapi_react
ARG JAX_PREALLOCATE=true
ARG JAX_MEM_FRAC=0.95
ARG JAX_ALLOCATOR=platform
ARG JAX_PREALLOC_LIMIT=8589934592
ENV DEBIAN_FRONTEND=noninteractive

# OS deps
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        bash curl ca-certificates git procps htop util-linux build-essential \
        python3 python3-venv python3-pip python3-dev \
        autoconf automake libtool m4 cmake pkg-config \
        jags iproute2 net-tools lsof \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Node (for remote UI bits)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get update && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*

# uv
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

WORKDIR /app

# Create venv + base tools
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}" && \
    uv pip install --no-cache-dir jupyterlab

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# CUDA symlink sanity
RUN set -e; \
    CUDA_REAL="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
    if [ -z "$CUDA_REAL" ] && [ -d /usr/local/cuda ]; then CUDA_REAL="/usr/local/cuda"; fi; \
    if [ -z "$CUDA_REAL" ]; then echo '❌ No CUDA toolkit found.' >&2; exit 1; fi; \
    if [ "$CUDA_REAL" != "/usr/local/cuda" ]; then ln -sfn "$CUDA_REAL" /usr/local/cuda; fi; \
    echo "🟢 CUDA toolkit: $CUDA_REAL"

# (no JAX or Torch installs here; bootstrap handles them)

# GPU / JAX envs
ENV XLA_PYTHON_CLIENT_PREALLOCATE=${JAX_PREALLOCATE}
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRAC}
ENV XLA_PYTHON_CLIENT_ALLOCATOR=${JAX_ALLOCATOR}
ENV XLA_FLAGS="--xla_force_host_platform_device_count=1"
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=${JAX_PREALLOC_LIMIT}
ENV LD_LIBRARY_PATH="/app/.venv/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
RUN echo 'cd /workspace' > /etc/profile.d/99-workspace-cd.sh
RUN mkdir -p /root/.ipython/profile_default/startup && \
    printf "import os, sys\nos.chdir('/workspace')\nsys.path.append('/workspace')\n" \
      > /root/.ipython/profile_default/startup/00-cd-workspace.py
RUN echo '. /app/.venv/bin/activate' > /etc/profile.d/10-uv-activate.sh

CMD ["bash"]
