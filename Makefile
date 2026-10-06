.PHONY: setup api web test lint format build docker train-cpu train-gpu

setup:            ## install Python workspace + frontend dependencies
	uv sync
	cd frontend && npm install

api:              ## backend on :8000 (lite mode, SQLite)
	cd backend && uv run uvicorn aura_api.main:app --reload --port 8000

web:              ## frontend on :5173 (proxies /api to :8000)
	cd frontend && npm run dev

test:             ## Python tests + frontend type-check
	uv run pytest -q
	cd frontend && npx tsc -b

lint:
	uv run ruff check .
	uv run ruff format --check .
	cd frontend && npx oxlint

format:
	uv run ruff check --fix .
	uv run ruff format .

build:            ## production frontend bundle
	cd frontend && npm run build

docker:           ## Postgres + app on :8000
	docker compose up --build

train-cpu:        ## the three CPU models (minutes, no GPU)
	uv run aura-ml train-text-emotion
	uv run aura-ml train-strategy
	uv run aura-ml train-topic

train-gpu:        ## everything that needs a GPU (see docs/TRAINING.md)
	uv run aura-ml build-ece && uv run aura-ml train-ece
	uv run aura-ml train-ser
	uv run aura-ml get-fer2013 && uv run aura-ml train-fer
	uv run aura-ml build-sft && uv run aura-ml train-llm
