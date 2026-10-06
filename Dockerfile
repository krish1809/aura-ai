# syntax=docker/dockerfile:1
# Aura - one image serving the API and the built web app (lite mode: CPU only).
# For the trained GPU models see docker/Dockerfile.gpu.

# ---- 1. build the React frontend -------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python app ---------------------------------------------------------------
FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir "uv>=0.5" && useradd --create-home --uid 1000 aura
WORKDIR /app

# dependencies first (cached unless the lockfile changes)
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
RUN mkdir -p ml/src/aura_ml backend/src/aura_api \
 && touch ml/src/aura_ml/__init__.py backend/src/aura_api/__init__.py \
 && uv sync --frozen --no-dev --no-install-workspace --package aura-api

COPY ml/src ml/src
COPY backend/src backend/src
COPY backend/alembic.ini backend/alembic.ini
COPY backend/migrations backend/migrations
RUN uv sync --frozen --no-dev --package aura-api

COPY artifacts/strategy artifacts/strategy
COPY artifacts/topic artifacts/topic
COPY artifacts/text_emotion artifacts/text_emotion
COPY --from=web /web/dist /app/web
COPY docker/entrypoint.sh /entrypoint.sh

ENV AURA_ENVIRONMENT=production \
    AURA_FRONTEND_DIST=/app/web \
    AURA_ARTIFACTS_DIR=/app/artifacts \
    AURA_AUTO_CREATE_TABLES=false \
    AURA_ML_MODE=lite
USER aura
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
ENTRYPOINT ["/entrypoint.sh"]
