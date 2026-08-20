# app/api/routes/detect.py — add metrics tracking

import time
import tempfile
import os
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from app.core.database import store_frame_analysis
from app.core.bridge import frame_analysis_to_text
from app.core.metrics import (
    DETECTION_REQUESTS_TOTAL,
    DETECTION_LATENCY_SECONDS,
    YOLO_INFERENCE_SECONDS,
    WORKERS_DETECTED_TOTAL,
    VIOLATIONS_DETECTED_TOTAL,
    VLM_VERIFICATIONS_TOTAL,
    COMPLIANCE_RATE_GAUGE,
    MODEL_LOADED_GAUGE,
)

router = APIRouter()

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

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

@router.post("", response_model=DetectionResponse)
async def detect_ppe(
    request: Request,
    file: UploadFile = File(...),
    return_image: bool = True
):
    """Upload an image for PPE compliance analysis."""

    # Start timing the entire request
    request_start = time.time()

    # Lazy load detector
    if request.app.state.detector is None:
        from app.core.detector import PPEDetector
        from app.config import settings

        model_path = settings.yolo_model_path
        if not Path(model_path).exists():
            DETECTION_REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {model_path}"
            )
        request.app.state.detector = PPEDetector(model_path)
        MODEL_LOADED_GAUGE.set(1)

    file_ext = "." + file.filename.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        DETECTION_REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}"
        )

    detector = request.app.state.detector

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Time only the YOLO inference part
        inference_start = time.time()
        frame_analysis = detector.analyze_frame(tmp_path)
        inference_duration = time.time() - inference_start

        # Record inference time
        YOLO_INFERENCE_SECONDS.observe(inference_duration)

        # Record worker and violation counts
        WORKERS_DETECTED_TOTAL.inc(frame_analysis.total_workers)

        for worker in frame_analysis.worker_analyses:
            for violation in worker.violations:
                # Extract base violation type without "(inferred)" suffix
                vtype = violation.replace(" (inferred)", "")
                VIOLATIONS_DETECTED_TOTAL.labels(
                    violation_type=vtype
                ).inc()

            if worker.needs_verification:
                VLM_VERIFICATIONS_TOTAL.labels(result="pending").inc()

        # Update current compliance rate
        COMPLIANCE_RATE_GAUGE.set(frame_analysis.compliance_rate)

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

        annotated_b64 = ""
        if return_image:
            import cv2
            annotated = detector.draw_results(tmp_path, frame_analysis)
            _, buffer = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # Record successful request
        DETECTION_REQUESTS_TOTAL.labels(status="success").inc()

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

    except Exception as e:
        DETECTION_REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)
        # Record total request duration
        total_duration = time.time() - request_start
        DETECTION_LATENCY_SECONDS.observe(total_duration)