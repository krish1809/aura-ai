from __future__ import annotations

from fastapi import APIRouter, Request

from aura_ml.safety import DEFAULT_RESOURCES

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict:
    ml = request.app.state.ml
    return {"status": "ok", "models_ready": ml.ready, "mode": request.app.state.settings.ml_mode}


@router.get("/system/models")
async def models(request: Request) -> dict:
    """Which models are loaded and what each one fell back to - shown on the About page."""
    ml = request.app.state.ml
    return {
        "ready": ml.ready,
        "mode": request.app.state.settings.ml_mode,
        "components": ml.status(),
    }


@router.get("/system/resources")
async def resources() -> dict:
    return {"crisis_resources": DEFAULT_RESOURCES}
