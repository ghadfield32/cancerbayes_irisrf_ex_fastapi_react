# FastAPI + React • Railway-ready Template

This repo contains a FastAPI back-end (`/api`) and a React (Vite) front-end (`/web`).  
Follow the steps below to run everything locally with **Railway CLI** and then deploy the two services on railway.com.

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

| Key            | Sample value                   | Note                      |
| -------------- | ------------------------------ | ------------------------- |
| `SECRET_KEY`   | `super-secret-change-me`       | JWT signing key (backend) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | or Postgres URI           |
| `VITE_API_URL` | `http://localhost:8000`        | front-end → back-end URL  |

---

## 4 · Install all local deps (one command)

```bash
npm run install:all     # sets up Python venv, uv, and Node modules
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

The `railway run` wrapper injects your `.env` so you’re testing exactly what will run in the cloud.

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

   * Open each service → Variables → “New Variable from File” → upload **.env**.
   * Delete `VITE_API_URL` from the `api` service and `SECRET_KEY`, `DATABASE_URL` from the `web` service so each side only keeps what it uses.

4. **Click Deploy** (or just push more commits; Railway auto-deploys).

---

## 7 · Verify production URLs

After the first deploy, Railway shows a unique domain for each service:

```text
https://api--<random>.up.railway.app
https://web--<random>.up.railway.app
```

Update **`VITE_API_URL`** in the *web* service variables to the API’s final URL, redeploy the web service, and you’re done.

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
* **Wrong root** – If the build log tries to install both back-end and front-end deps, re-check the “Root directory” for that service.
* **Need Docker instead of Nixpacks?** – Drop a `Dockerfile` in `api/` or `web/` and Railway will automatically build from it.

Happy shipping! 🚂

```

---

### Why these steps work

* **Railway CLI**: provides `init`, `link`, `run`, and `variables` for local parity and CI scripting. :contentReference[oaicite:1]{index=1}  
* **Environment variables**: `railway run` loads envs exactly as deployed, and `railway variables` lets you sync `.env` with the remote service store. :contentReference[oaicite:2]{index=2}  
* **Monorepo root directories**: setting the root prevents unnecessary installs and speeds builds. :contentReference[oaicite:3]{index=3}  
* **Nixpacks vs. Dockerfile**: Railway defaults to Nixpacks but switches automatically when it detects a `Dockerfile`. :contentReference[oaicite:4]{index=4}  
* **FastAPI/React precedent**: countless community templates follow the same flow (clone → env → commit → Railway). :contentReference[oaicite:5]{index=5}
::contentReference[oaicite:6]{index=6}
