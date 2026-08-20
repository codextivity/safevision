# app/main.py — add Prometheus instrumentation

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

# Prometheus instrumentation
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import health, detect, query, violations
from app.config import settings
from app.core.database import initialize_database
from app.core.metrics import MODEL_LOADED_GAUGE, MEMORY_USAGE_BYTES

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting SafeVision API...")
    initialize_database()

    app.state.detector = None
    app.state.agent = None

    # Update system metrics at startup
    MODEL_LOADED_GAUGE.set(0)

    # Background warmup
    import asyncio

    async def warmup():
        await asyncio.sleep(2)
        try:
            from app.core.detector import PPEDetector
            from app.core.agent import build_safety_agent
            from app.config import settings

            if Path(settings.yolo_model_path).exists():
                print("Warming up detector...")
                app.state.detector = PPEDetector(settings.yolo_model_path)
                MODEL_LOADED_GAUGE.set(1)  # ← update metric
                print("Detector ready")

            print("Warming up agent...")
            app.state.agent = build_safety_agent()
            print("Agent ready")

        except Exception as e:
            print(f"Warmup failed: {e}")
            MODEL_LOADED_GAUGE.set(0)

    asyncio.create_task(warmup())

    # Background memory monitoring
    async def monitor_memory():
        import psutil
        import os
        process = psutil.Process(os.getpid())
        while True:
            MEMORY_USAGE_BYTES.set(process.memory_info().rss)
            await asyncio.sleep(30)  # update every 30 seconds

    asyncio.create_task(monitor_memory())

    yield
    print("Shutting down...")

app = FastAPI(
    title="SafeVision — PPE Compliance Intelligence API",
    description="AI-powered PPE detection with natural language queries",
    version="1.0.0",
    lifespan=lifespan
)

# ── Prometheus instrumentation ────────────────────────────────────────────────
# This single line automatically instruments ALL FastAPI endpoints with:
#   http_requests_total          (counter by method, path, status)
#   http_request_duration_seconds (histogram of latency)
#   http_requests_in_progress    (gauge of active requests)
# These are the standard HTTP metrics every production API should have.

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",   # Prometheus scrapes this URL
    include_in_schema=False  # hide from FastAPI docs
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(detect.router, prefix="/detect", tags=["Detection"])
app.include_router(query.router, prefix="/query", tags=["Agent"])
app.include_router(violations.router, prefix="/violations", tags=["Violations"])