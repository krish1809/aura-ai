"""FastAPI application factory.

uvicorn aura_api.main:app --reload            # lite mode (no GPU needed)
AURA_ML_MODE=full AURA_LLM_PROVIDER=local uvicorn aura_api.main:app
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aura_api import __version__
from aura_api.config import DEFAULT_SECRET, Settings, get_settings
from aura_api.db import Database
from aura_api.routers import auth, conversations, insights, memory, system, ws
from aura_api.services.chat import ChatOrchestrator
from aura_api.services.ml import MLService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.is_production and settings.jwt_secret == DEFAULT_SECRET:
        raise RuntimeError("Set AURA_JWT_SECRET to a long random value in production.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.db = Database(settings.database_url)
        if settings.auto_create_tables:
            await app.state.db.create_all()
        app.state.ml = MLService(settings)
        app.state.chat = ChatOrchestrator(app.state.ml, app.state.db, settings)
        loader = asyncio.create_task(app.state.ml.load())  # serve requests while big models load
        yield
        loader.cancel()
        await app.state.chat.drain()
        app.state.ml.shutdown()
        await app.state.db.dispose()

    app = FastAPI(title="Aura API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (auth, conversations, insights, memory, system, ws):
        app.include_router(module.router)

    dist = settings.frontend_dist
    if dist and Path(dist, "index.html").exists():
        app.mount("/assets", StaticFiles(directory=Path(dist, "assets")), name="assets")

        root = Path(dist).resolve()

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            candidate = (root / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
            return FileResponse(root / "index.html")  # client-side routes and anything outside dist

    return app


app = create_app()
