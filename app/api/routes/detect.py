# app/api/routes/detect.py
# Handles image upload and PPE detection.

import tempfile
import os
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.core.database import store_frame_analysis
from app.core.bridge import frame_analysis_to_text

router = APIRouter()

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

class WorkerResult(BaseModel):
    worker_id: int
    is_compliant: bool
    has_hardhat: bool
    has_safety_vest: bool
    violations: list[str]
    needs_verification: bool
    confidence: float

class DetectionResponse(BaseModel):
    frame_id: int
    total_workers: int
    compliant_workers: int
    violation_workers: int
    compliance_rate: float
    workers: list[WorkerResult]
    summary: str
    annotated_image_base64: str = ""

@router.post("", response_model=DetectionResponse)
async def detect_ppe(
    request: Request,
    file: UploadFile = File(...),
    return_image: bool = True
):
    # Lazy load detector on first request
    if request.app.state.detector is None:
        from app.core.detector import PPEDetector
        from app.config import settings
        from pathlib import Path

        model_path = settings.yolo_model_path
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {model_path}"
            )
        print(f"Loading detector on first request: {model_path}")
        request.app.state.detector = PPEDetector(model_path)

    detector = request.app.state.detector

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="PPE detector not loaded. Run train.py first."
        )

    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. "
                   f"Use: {SUPPORTED_FORMATS}"
        )

    # Save to temp file — YOLO needs a file path not bytes
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file_ext
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run full detection pipeline
        frame_analysis = detector.analyze_frame(tmp_path)

        # Store in database
        frame_id = store_frame_analysis(frame_analysis)

        # Build response
        workers = [
            WorkerResult(
                worker_id=w.worker_id,
                is_compliant=w.is_compliant,
                has_hardhat=w.has_hardhat,
                has_safety_vest=w.has_safety_vest,
                violations=w.violations,
                needs_verification=w.needs_verification,
                confidence=w.person_detection.confidence
            )
            for w in frame_analysis.worker_analyses
        ]

        # Generate annotated image if requested
        annotated_b64 = ""
        if return_image:
            import cv2
            import numpy as np
            annotated = detector.draw_results(tmp_path, frame_analysis)
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return DetectionResponse(
            frame_id=frame_id,
            total_workers=frame_analysis.total_workers,
            compliant_workers=frame_analysis.compliant_workers,
            violation_workers=frame_analysis.violation_workers,
            compliance_rate=frame_analysis.compliance_rate,
            workers=workers,
            summary=frame_analysis_to_text(frame_analysis),
            annotated_image_base64=annotated_b64
        )

    finally:
        os.unlink(tmp_path)