# app/main.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path

from app.api.routes import health, detect, query, violations
from app.config import settings
from app.core.database import initialize_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialize database and load models.
    Models are loaded once and reused across all requests.
    Loading YOLO on every request would add 200ms per call.
    """
    print("Starting SafeVision API...")

    # Initialize SQLite database
    initialize_database()

    onnx_path = Path("models/trained/best.onnx")
    print(f"ONNX exists: {onnx_path.exists()}")
    print(f"ONNX size: {onnx_path.stat().st_size if onnx_path.exists() else 'N/A'} bytes")

    # Load YOLO detector if model exists
    model_path = Path(settings.yolo_model_path)
    if model_path.exists():
        from app.core.detector import PPEDetector
        app.state.detector = PPEDetector(str(model_path))
        print(f"PPE detector loaded from {model_path}")
    else:
        app.state.detector = None
        print(f"Warning: No trained model at {model_path}")
        print("Run train.py first to train the model")

    # Build LangChain safety agent
    from app.core.agent import build_safety_agent
    app.state.agent = build_safety_agent()
    print("Safety agent ready")

    yield

    print("Shutting down SafeVision API...")

app = FastAPI(
    title="SafeVision — PPE Compliance Intelligence API",
    description=(
        "AI-powered construction site safety monitoring. "
        "Combines YOLOv11 PPE detection with GPT-4o verification "
        "and a LangChain agent for natural language safety queries."
    ),
    version="1.0.0",
    lifespan=lifespan
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