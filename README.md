# FastAPI + React • Railway-ready Template

This repo contains a FastAPI back-end (`/api`) and a React (Vite) front-end (`/web`) with **JAX/NumPyro backend** for Bayesian machine learning models.  
Follow the steps below to run everything locally with **Railway CLI** and then deploy the two services on railway.com.

## 🧠 Machine Learning Features

- **JAX/NumPyro Backend**: No C compilation required - works seamlessly on Windows, Linux, and macOS
- **Hierarchical Bayesian Models**: Breast cancer diagnosis with varying intercepts by texture quintiles
- **Random Forest & Logistic Regression**: Iris species classification
- **Self-healing Model Service**: Automatically trains missing models in the background
- **MLflow Integration**: Model versioning and deployment tracking
- **Rate Limiting**: Redis-backed token bucket rate limiting with configurable limits per endpoint type
- **Automated Garbage Collection**: Keeps Railway volumes tidy by pruning old runs and artifacts

### Why JAX/NumPyro?

- **Cross-platform**: No compiler dependencies - pure Python + JIT compilation
- **Fast sampling**: NumPyro NUTS sampler is significantly faster than traditional PyMC
- **Windows-friendly**: Eliminates MSVC/GCC compilation issues
- **Production-ready**: Stable JAX 0.4.28 LTS with NumPyro 0.14.0

---

## 1 · Clone the template

```bash
git clone <repository-url>
cd <repository-name>
````

set up the environment variables in the .env file

Then set up and test frontend and backend tempaltes locally:

```bash
npm run install:all
npm run backend
npm run seed
cd web; npm run dev
```

tests to ensure it works:
backend:
curl -s http://127.0.0.1:8000/docs

frontend:
curl -s http://127.0.0.1:5173
```



---

## 2 · Install & link Railway CLI

```bash
curl -fsSL https://railway.com/install.sh | sh   # one-liner for macOS, Linux, WSL
railway login                                    # opens browser once
railway init -p <optional-existing-project-id>   # creates or links a project
```

---

## 3 · Create your `.env` from the template

```bash
cp .env.template .env
nano .env         # or code .env / vim .env
```

| Key                    | Sample value                   | Note                      |
| ---------------------- | ------------------------------ | ------------------------- |
| `SECRET_KEY`           | `super-secret-change-me`       | JWT signing key (backend) |
| `DATABASE_URL`         | `sqlite+aiosqlite:///./app.db` | or Postgres URI           |
| `VITE_API_URL`         | `http://localhost:8000`        | front-end → back-end URL  |
| `REDIS_URL`            | `redis://localhost:6379`       | Redis for rate limiting   |
| `RATE_LIMIT_DEFAULT`   | `60`                           | Default requests per minute |
| `RATE_LIMIT_CANCER`    | `30`                           | Cancer prediction limit   |
| `RATE_LIMIT_LOGIN`     | `3`                            | Login attempts per 20s    |
| `RETAIN_RUNS_PER_MODEL`| `5`                            | Keep N latest runs per model |
| `MLFLOW_GC_AFTER_TRAIN`| `1`                            | Run garbage collection after training |

---

## 4 · Install all local deps (one command)

```bash
npm run install:all     # sets up Python venv, uv, and Node modules
```

### ML Dependencies Setup

The project uses JAX/NumPyro for Bayesian modeling. Install the ML dependencies:

**Linux/macOS:**
```bash
chmod +x scripts/setup-jax.sh
./scripts/setup-jax.sh
```

**Windows:**
```cmd
scripts\setup-jax.bat
```

**Manual installation:**
```bash
pip install "jax[cpu]==0.4.28" "jaxlib==0.4.28" "numpyro==0.14.0"
```

### Test the ML Setup

```bash
# Test JAX/NumPyro backend
python test_pytensor_fix.py

# Test Bayesian training
python tests/test_bayesian_trainer.py

# Test garbage collection
python api/test_volume_cleanup.py
```

---

## 5 · Smoke-test locally *via Railway CLI*

```bash
# back-end
cd api
railway run uvicorn app.main:app --reload
# (new terminal) seed the DB
railway run python -m scripts.seed_user
# front-end
cd ../web
railway run npm run dev
```

* Front-end → [http://localhost:5173](http://localhost:5173)
* API → [http://localhost:8000/docs](http://localhost:8000/docs)

The `railway run` wrapper injects your `.env` so you're testing exactly what will run in the cloud.

---

## 6 · Prepare the repo for Railway

1. **Commit and push** everything to GitHub.

2. In the Railway dashboard create **two services** in the same project:

   | Service | Root directory (Settings → Root) |
   | ------- | -------------------------------- |
   | `api`   | `api`                            |
   | `web`   | `web`                            |

   (Root directories ensure each build only pulls the code it needs.)

3. **Copy env vars**

   * Open each service → Variables → "New Variable from File" → upload **.env**.
   * Delete `VITE_API_URL` from the `api` service and `SECRET_KEY`, `DATABASE_URL` from the `web` service so each side only keeps what it uses.
   * **Add Redis Plugin**: In the `api` service, add the Redis plugin for rate limiting functionality.

4. **Click Deploy** (or just push more commits; Railway auto-deploys).

---

## 7 · Verify production URLs

After the first deploy, Railway shows a unique domain for each service:

```text
https://api--<random>.up.railway.app
https://web--<random>.up.railway.app
```

Update **`VITE_API_URL`** in the *web* service variables to the API's final URL, redeploy the web service, and you're done.

---

### Recap — three-command workflow after the first push

```bash
# make changes …
git add .
git commit -m "feat: awesome change"
git push            # triggers two Railway builds
```

---

## Troubleshooting

* **401 token expired** – Refresh the token in `localStorage` or simply log out / back in; your FastAPI handler will now return a helpful hint.
* **Wrong root** – If the build log tries to install both back-end and front-end deps, re-check the "Root directory" for that service.
* **Need Docker instead of Nixpacks?** – Drop a `Dockerfile` in `api/` or `web/` and Railway will automatically build from it.

---

## 🗂️ Railway Volume Setup for MLflow Persistence

### Problem: Bayesian Models Missing in Production

The Bayesian breast-cancer models were "missing" in Railway but working fine locally due to:

1. **Ephemeral containers**: Railway containers are stateless by default - `/app/mlruns_local` vanishes on every deploy
2. **Missing artifacts**: MLflow registry entries exist but artifact directories are gone
3. **No graceful fallback**: Missing artifacts caused startup failures

### Solution: Railway Volume + Tolerant Loading

#### 1. Create Railway Volume

In your Railway dashboard:
1. Go to your `api` service
2. Navigate to **Volumes** → **New Volume**
3. Name it `mlruns` and mount at `/data/mlruns`
4. Set size to 1GB (sufficient for model artifacts)

#### 2. Update Environment Variables

In your `api` service variables, set:
```bash
MLFLOW_TRACKING_URI=file:/data/mlruns
MLFLOW_REGISTRY_URI=file:/data/mlruns
```

#### 3. Enhanced Model Loading

The service now handles missing artifacts gracefully:

- **Tolerant loading**: `FileNotFoundError` and `OSError` are caught and logged
- **Graceful fallback**: Missing Bayesian models fall back to stub models
- **Background training**: Heavy models train in background while serving traffic
- **Pre-training**: Models are pre-trained during container build phase

#### 4. Reduced PyMC Divergences

Updated `target_accept=0.95` (from 0.90) to dramatically reduce "divergences after tuning" warnings.

### Validation Checklist

After deploying with the Volume:

1. **Check model status**: `GET /ready/full` should show `breast_cancer_stub : loaded`
2. **Test predictions**: `POST /cancer/predict` should return valid probabilities
3. **Monitor training**: Bayesian model status should change from `training` to `loaded`

### Best Practices

- **Separate training and inference**: Consider training heavy models on beefier workers
- **Pin versions**: Keep MLflow and PyMC versions consistent in `pyproject.toml`
- **Monitor divergences**: Watch PyMC logs for convergence issues
- **Backup volumes**: Railway volumes can be backed up for disaster recovery

---

## 🧹 MLflow Garbage Collection

### Problem: Volume Space Management

MLflow persists every run forever, so your Railway volume will fill up unless you prune old runs and artifacts.

### Solution: Automated Cleanup

The service now includes automated garbage collection:

#### 1. Configuration

Set these environment variables in your Railway service:

```bash
RETAIN_RUNS_PER_MODEL=5      # Keep the latest 5 runs per model
MLFLOW_GC_AFTER_TRAIN=1      # Run garbage collection after each training
```

#### 2. How It Works

After each successful model training:

1. **List all runs** for that model (newest first)
2. **Keep the latest N** (default: 5) runs
3. **Delete older runs** via `MlflowClient.delete_run()`
4. **Run `mlflow gc`** to purge artifact folders
5. **Log disk usage** before/after cleanup

#### 3. Manual Cleanup

For store-wide cleanup (e.g., from Railway Cron Jobs):

```bash
python - <<'PY'
from api.app.services.ml.model_service import model_service
import asyncio, uvloop; uvloop.install()
asyncio.run(model_service.vacuum_store())
PY
```

#### 4. Railway Cron Job Setup

Add a daily cron job in Railway:

1. Go to your `api` service → **Cron Jobs** → **New Cron Job**
2. Schedule: `0 2 * * *` (daily at 2 AM)
3. Command:
   ```bash
   python -c "
   from app.services.ml.model_service import model_service
   import asyncio
   asyncio.run(model_service.vacuum_store())
   "
   ```

### Benefits

- **Automatic cleanup**: No manual intervention required
- **Configurable retention**: Adjust `RETAIN_RUNS_PER_MODEL` as needed
- **Non-blocking**: Cleanup runs in background thread pool
- **Resilient**: Failures never bubble up to API
- **Transparent**: Detailed logging of cleanup operations

### Monitoring

Watch for these log messages:

```
🗑️  Pruned 3 old iris_random_forest runs; kept 5
🧹 mlflow gc completed (45.23 MB → 12.45 MB)
```

### Testing

Test the garbage collection locally:

```bash
cd api
python test_volume_cleanup.py
```

### Volume Cleanup Verification

#### How Cleanup Works

1. **Retention Policy**: Keep the latest N runs per model (default: 5)
2. **Pruning**: Delete older runs via `MlflowClient.delete_run()`
3. **Garbage Collection**: Run `mlflow gc` to purge artifact folders
4. **Background Execution**: Cleanup runs in thread pool, never blocks API

#### Verification Steps

**Local Testing**:
```bash
# Measure volume size
du -sh mlruns_local

# Run cleanup
python -c "
from app.services.ml.model_service import model_service
import asyncio
asyncio.run(model_service.vacuum_store())
"

# Measure again
du -sh mlruns_local
```

**Python Script**:
```python
import os
import shutil
from app.services.ml.model_service import model_service
import asyncio

def folder_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

path = "mlruns_local"
print("Before:", folder_size(path), "bytes")
asyncio.run(model_service.vacuum_store())
print("After: ", folder_size(path), "bytes")
```

#### Demo Script

Run the cleanup demo to see it in action:

```bash
cd api
python demo_cleanup.py
```

This will:
1. Train multiple models to create runs
2. Measure volume size before/after
3. Run cleanup and show size reduction
4. Display detailed statistics

#### Railway Volume Monitoring

When deployed with Railway volumes:

- **Automatic cleanup**: Happens after each training
- **Periodic vacuum**: Daily cron job for extra assurance
- **Size monitoring**: Watch volume usage in Railway dashboard
- **Log monitoring**: Look for cleanup messages in logs

### Best Practices

- **Keep retention low**: 5-10 runs per model to avoid disk exhaustion
- **Monitor disk usage**: Set alerts for >80% volume capacity
- **Use file store for dev**: Prevents leftover Docker volumes
- **Test locally**: Use `test_volume_cleanup.py` to verify functionality

Happy shipping! 🚂
