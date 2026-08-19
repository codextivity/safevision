# app/config.py — updated for new dataset class names

from pydantic_settings import BaseSettings
from pathlib import Path

ENV_FILE_PATH = Path(__file__).parent.parent / ".env"

class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "safevision"
    tavily_api_key: str = ""
    roboflow_api_key: str = ""

    # ── Model settings ────────────────────────────────────────────────────────
    openai_chat_model: str = "gpt-4o-mini"    # ← add this line
    yolo_base_model: str = "yolo11m.pt"
    yolo_model_path: str = "models/trained/best.pt"

    # ── Training settings ─────────────────────────────────────────────────────
    # imgsz=1280: Hardhat has 42.9% tiny objects — needs high resolution
    imgsz: int = 640

    # epochs=50: enough for convergence on 5646 images
    epochs: int = 50

    # batch=8: safe for RTX 5080 at 1280px
    # reduce to 4 if CUDA out of memory error occurs
    batch_size: int = 16

    workers: int = 4

    # ── Class names — match dataset exactly including case ────────────────────
    # These must match data.yaml class names exactly
    class_names: list[str] = [
        "Hardhat",
        "NO-Hardhat",
        "NO-Safety Vest",
        "Person",
        "Safety Vest"
    ]

    # Violation classes — detections that indicate non-compliance
    violation_classes: list[str] = ["NO-Hardhat", "NO-Safety Vest"]

    # Compliant PPE classes
    compliant_classes: list[str] = ["Hardhat", "Safety Vest"]

    # ── Inference settings ────────────────────────────────────────────────────
    # confidence_threshold: minimum to accept any detection
    confidence_threshold: float = 0.25

    # verification_threshold: below this confidence on violation classes,
    # send to GPT-4o for verification — false alarms on violations
    # are more costly than missing compliant detections
    verification_threshold: float = 0.70

    # ── Spatial association settings ──────────────────────────────────────────
    # How we pair person detections with PPE/violation detections
    # max_association_distance: maximum distance between person center
    # and PPE center, as ratio of person bounding box size
    max_association_distance: float = 2.0

    # iou_threshold for NMS during inference
    iou_threshold: float = 0.45

    # ── Storage ───────────────────────────────────────────────────────────────
    database_path: str = "data/violations.db"
    data_yaml_path: str = "data/raw/data.yaml"

    model_config = {
        "env_file": str(ENV_FILE_PATH),
        "extra": "ignore",
        "case_sensitive": False,
    }

settings = Settings()