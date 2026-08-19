# app/core/verifier.py
# GPT-4o vision verification for uncertain YOLO detections.
#
# When YOLO detects a potential violation with low confidence,
# we send the image region to GPT-4o for a second opinion.
#
# Why GPT-4o as a verifier rather than primary detector?
# YOLO: fast (30+ FPS), cheap ($0), runs locally
# GPT-4o: slow (2-3s), costs money, requires API call
# Best of both: YOLO screens all frames, GPT-4o only
# verifies the uncertain ones (typically 5-15% of frames)

import base64
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from app.core.detector import WorkerAnalysis, FrameAnalysis
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)

def crop_worker_region(
    image_path: str,
    worker: WorkerAnalysis,
    padding: float = 0.2
) -> str:
    """
    Crops the image region around a worker for focused VLM analysis.

    Why crop instead of sending the full image?
    1. Reduces tokens — smaller image = lower API cost
    2. Focuses attention — GPT-4o analyzes the specific worker
       rather than the entire construction site
    3. Better accuracy — zoomed-in view makes small PPE items clearer

    Args:
        image_path: source image path
        worker:     WorkerAnalysis with person bounding box
        padding:    fraction of bbox size to add as padding (default 20%)

    Returns:
        Path to the cropped temporary image file
    """
    import cv2
    import tempfile

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    x1, y1, x2, y2 = map(int, worker.person_detection.bbox)

    # Add padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    # Clamp to image boundaries
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    cropped = img[y1:y2, x1:x2]

    # Save to temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpg"
    ) as tmp:
        cv2.imwrite(tmp.name, cropped)
        return tmp.name

def encode_image(image_path: str) -> str:
    """Encodes image to base64 for OpenAI API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def verify_worker_compliance(
    image_path: str,
    worker: WorkerAnalysis
) -> dict:
    """
    Uses GPT-4o vision to verify compliance for an uncertain worker.

    Called only when YOLO flagged the worker as needing verification:
    - Low confidence detection
    - Conflicting signals (hardhat detected AND no-hardhat detected)
    - No PPE detected for high-confidence person

    Args:
        image_path: path to original image
        worker:     WorkerAnalysis flagged as needs_verification=True

    Returns:
        Dict with verified compliance status and reasoning
    """
    import os

    # Crop the worker region for focused analysis
    crop_path = crop_worker_region(image_path, worker)

    try:
        image_data = encode_image(crop_path)

        # Build context about what YOLO already detected
        yolo_context = f"""
YOLO detector findings for this worker:
- Has hardhat detected: {worker.has_hardhat}
- Has safety vest detected: {worker.has_safety_vest}
- NO-Hardhat detected: {worker.no_hardhat_detected}
- NO-Safety Vest detected: {worker.no_vest_detected}
- Reason for verification: {worker.verification_reason}
"""

        prompt = f"""You are a construction site safety inspector.
Examine this image of a construction worker and verify their PPE compliance.

{yolo_context}

Answer these questions based on what you can see:
1. Is the worker wearing a hardhat/helmet?
2. Is the worker wearing a safety/high-visibility vest?
3. Are there any other safety concerns visible?

Return ONLY a JSON object with exactly these fields:
{{
    "has_hardhat": true or false,
    "has_safety_vest": true or false,
    "violations": ["list", "of", "violations"],
    "is_compliant": true or false,
    "confidence": "high", "medium", or "low",
    "reasoning": "brief explanation of what you see"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            response_format={"type": "json_object"},
            max_tokens=500
        )

        result = json.loads(response.choices[0].message.content)
        result["verified_by"] = "gpt-4o"
        result["worker_id"] = worker.worker_id
        return result

    finally:
        # Always clean up temp file
        os.unlink(crop_path)

def verify_frame(
    image_path: str,
    frame_analysis: FrameAnalysis
) -> FrameAnalysis:
    """
    Verifies all workers flagged as needing verification in a frame.

    Updates the FrameAnalysis in-place with GPT-4o verified results.

    Args:
        image_path:     source image path
        frame_analysis: result from PPEDetector.analyze_frame()

    Returns:
        Updated FrameAnalysis with verified compliance statuses
    """
    workers_to_verify = [
        w for w in frame_analysis.worker_analyses
        if w.needs_verification
    ]

    if not workers_to_verify:
        return frame_analysis

    print(f"Verifying {len(workers_to_verify)} uncertain detections "
          f"with GPT-4o...")

    for worker in workers_to_verify:
        verification = verify_worker_compliance(image_path, worker)

        # Update worker analysis with verified results
        worker.has_hardhat = verification.get("has_hardhat", worker.has_hardhat)
        worker.has_safety_vest = verification.get(
            "has_safety_vest", worker.has_safety_vest
        )
        worker.violations = verification.get("violations", worker.violations)
        worker.is_compliant = verification.get("is_compliant", False)
        worker.needs_verification = False  # verified — no longer uncertain

        print(f"  Worker {worker.worker_id}: "
              f"{'COMPLIANT' if worker.is_compliant else 'VIOLATION'} "
              f"(confidence: {verification.get('confidence', 'unknown')})")

    # Recalculate frame statistics
    total = frame_analysis.total_workers
    compliant = sum(1 for w in frame_analysis.worker_analyses if w.is_compliant)
    violations = sum(1 for w in frame_analysis.worker_analyses if w.violations)

    frame_analysis.compliant_workers = compliant
    frame_analysis.violation_workers = violations
    frame_analysis.needs_verification = 0
    frame_analysis.compliance_rate = compliant / total if total > 0 else 1.0

    return frame_analysis