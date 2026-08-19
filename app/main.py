# app/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.core.database import initialize_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting SafeVision API...")

    # Only initialize the database at startup
    # Everything else loads lazily on first request
    initialize_database()

    # Set placeholders — loaded on first request
    app.state.detector = None
    app.state.agent = None

    print("API ready — warming up models in background...")
     # Warmup runs after startup completes
    # Does not block the health check from passing
    import asyncio
    async def warmup():
        await asyncio.sleep(2)  # wait for server to fully start
        try:
            from app.core.detector import PPEDetector
            from app.core.agent import build_safety_agent
            from app.config import settings
            from pathlib import Path

            if Path(settings.yolo_model_path).exists():
                print("Warming up detector...")
                app.state.detector = PPEDetector(settings.yolo_model_path)
                print("Detector ready")

            print("Warming up agent...")
            app.state.agent = build_safety_agent()
            print("Agent ready")

        except Exception as e:
            print(f"Warmup failed: {e} — will load on first request")

    asyncio.create_task(warmup())

    yield
    print("Shutting down...")

app = FastAPI(
    title="SafeVision — PPE Compliance Intelligence API",
    description="AI-powered PPE detection with natural language queries",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import health, detect, query, violations
app.include_router(health.router, tags=["Health"])
app.include_router(detect.router, prefix="/detect", tags=["Detection"])
app.include_router(query.router, prefix="/query", tags=["Agent"])
app.include_router(violations.router, prefix="/violations", tags=["Violations"])