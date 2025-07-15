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

Happy shipping! 🚂
