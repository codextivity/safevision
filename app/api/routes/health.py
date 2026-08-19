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
    agent_loaded: bool
    database_exists: bool
    memory_mb: float

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    # Get current memory usage
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1e6
    except ImportError:
        memory_mb = 0.0

    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        model_loaded=request.app.state.detector is not None,
        agent_loaded=request.app.state.agent is not None,
        database_exists=Path(settings.database_path).exists(),
        memory_mb=round(memory_mb, 1)
    )