# app/api/routes/health.py

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from app.config import settings

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    database_path: str

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Returns API status and component availability."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=request.app.state.detector is not None,
        database_path=settings.database_path
    )