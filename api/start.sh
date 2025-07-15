#!/usr/bin/env bash
set -euo pipefail

# ── sanity ─────────────────────────────────────────────────────────
if [[ -z "${PORT:-}" ]]; then
  echo "❌  PORT not set – Railway always provides it." >&2
  exit 1
fi

if [[ -z "${SECRET_KEY:-}" ]]; then
  echo "❌  SECRET_KEY is not set for the backend service – aborting." >&2
  exit 1
fi

echo "🚀  FastAPI boot; PORT=$PORT  PY=$(python -V)"
env | grep -E 'RAILWAY_|PORT|DATABASE_URL' | sed 's/SECRET_KEY=.*/SECRET_KEY=***/'

# ── optional local .env ------------------------------------------------------
[[ -f .env ]] && export $(grep -Ev '^#' .env | xargs)

# ── one-shot DB migrate + seed (blocks until done) ---------------------------
python -m scripts.seed_user

# ── pre-train models (optional, non-blocking) --------------------------------
echo "🏗️  Pre-training models..."
python -m api.scripts.ensure_models || echo "⚠️  pre-train failed – stub will load"

# ── run the API --------------------------------------------------------------
exec uvicorn app.main:app \
  --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips="*" --log-level info


