#!/bin/sh
# Apply database migrations, then start the API (which also serves the web app).
set -e
cd /app/backend
alembic upgrade head
exec uvicorn aura_api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
