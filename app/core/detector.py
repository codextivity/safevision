# app/core/detector.py
# YOLO inference pipeline for PPE violation detection.
#
# This is the bridge between raw YOLO detections and
# meaningful safety events. It handles:
# 1. Running YOLO on an image
# 2. Parsing raw detections into structured objects
# 3. Spatial association — pairing each Person with nearby PPE
# 4. Classifying each worker as compliant or violating
# 5. Flagging uncertain detections for GPT-4o verification
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

from app.config import settings
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"PPEDetector using device: {DEVICE}")

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    """
    A single object detected by YOLO.

    bbox format: [x1, y1, x2, y2] in pixel coordinates
    confidence: 0.0 to 1.0
    class_name: matches dataset class names exactly
    """
    class_name: str
    confidence: float
    bbox: list[float]       # [x1, y1, x2, y2]
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

    @property
    def area(self) -> float:
        """Bounding box area in pixels."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def diagonal(self) -> float:
        """Bounding box diagonal — used for distance calculations."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return (w ** 2 + h ** 2) ** 0.5

    def distance_to(self, other: "Detection") -> float:
        """Euclidean distance between centers of two detections."""
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        return (dx ** 2 + dy ** 2) ** 0.5

@dataclass
class WorkerAnalysis:
    """
    Safety compliance analysis for a single detected worker.

    Each worker (Person detection) is analyzed to determine
    which PPE they are wearing and which violations exist.
    """
    worker_id: int
    person_detection: Detection
    associated_detections: list[Detection] = field(default_factory=list)

    # Compliance status per PPE type
    has_hardhat: bool = False
    has_safety_vest: bool = False
    no_hardhat_detected: bool = False
    no_vest_detected: bool = False

    # Overall compliance
    is_compliant: bool = False

    # Violations list
    violations: list[str] = field(default_factory=list)

    # Whether this analysis needs VLM verification
    needs_verification: bool = False
    verification_reason: str = ""

    def to_dict(self) -> dict:
        """Converts to dict for database storage and API response."""
        return {
            "worker_id": self.worker_id,
            "bbox": self.person_detection.bbox,
            "confidence": self.person_detection.confidence,
            "has_hardhat": self.has_hardhat,
            "has_safety_vest": self.has_safety_vest,
            "violations": self.violations,
            "is_compliant": self.is_compliant,
            "needs_verification": self.needs_verification,
            "verification_reason": self.verification_reason,
        }

@dataclass
class FrameAnalysis:
    """
    Complete safety analysis for a single image frame.

    Contains all worker analyses and frame-level statistics.
    """
    image_path: str
    total_workers: int
    compliant_workers: int
    violation_workers: int
    needs_verification: int
    worker_analyses: list[WorkerAnalysis]
    all_detections: list[Detection]
    compliance_rate: float

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "total_workers": self.total_workers,
            "compliant_workers": self.compliant_workers,
            "violation_workers": self.violation_workers,
            "needs_verification": self.needs_verification,
            "compliance_rate": self.compliance_rate,
            "workers": [w.to_dict() for w in self.worker_analyses],
            "violations_summary": self._summarize_violations(),
        }

    def _summarize_violations(self) -> dict:
        """Counts violation types across all workers."""
        summary = {"NO-Hardhat": 0, "NO-Safety Vest": 0}
        for worker in self.worker_analyses:
            for v in worker.violations:
                if v in summary:
                    summary[v] += 1
        return summary

# ── Detector class ────────────────────────────────────────────────────────────

class PPEDetector:
    """
    Main PPE detection and compliance analysis class.

    Wraps YOLO inference with domain-specific logic:
    - Spatial association of Person + PPE detections
    - Compliance scoring per worker
    - Uncertainty flagging for VLM verification

    Design decision: why a class instead of functions?
    The YOLO model is expensive to load (~200ms).
    Loading it once in __init__ and reusing across many
    inference calls is much faster than reloading per call.
    This is the standard pattern for ML model serving.
    """

    def __init__(self, model_path: str = None):
        """
        Loads the trained YOLO model.

        Args:
            model_path: path to trained .pt file.
                       Defaults to settings.yolo_model_path
        """
        model_path = model_path or settings.yolo_model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                f"Run train.py first."
            )

        print(f"Loading PPE detector from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = settings.class_names
        print(f"Detector ready. Classes: {self.class_names}")

    def detect(self, image_path: str) -> list[Detection]:
        """
        Runs YOLO inference on an image and returns raw detections.

        Args:
            image_path: path to image file

        Returns:
            List of Detection objects above confidence threshold
        """
        results = self.model(
            image_path,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
            device=DEVICE
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                class_name = self.class_names[class_id]

                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    class_id=class_id
                ))

        return detections

    def associate_ppe_with_workers(
        self,
        detections: list[Detection]
    ) -> list[WorkerAnalysis]:
        """
        Pairs each Person detection with nearby PPE detections.

        Algorithm:
        1. Separate detections into persons and PPE items
        2. For each person, find all PPE within max_association_distance
        3. Assign each PPE to the nearest person
        4. Analyze each person's associated PPE for compliance

        Why this approach?
        A simple overlap (IOU) check fails when PPE is worn on the head
        or chest — the helmet bbox may not overlap with the person bbox
        significantly. Distance-based association is more robust for
        construction site images where workers are often partially visible.

        Args:
            detections: raw YOLO detections from detect()

        Returns:
            List of WorkerAnalysis objects, one per detected person
        """
        # Separate persons from PPE
        persons = [d for d in detections if d.class_name == "Person"]
        ppe_items = [d for d in detections if d.class_name != "Person"]

        if not persons:
            return []

        # Initialize worker analyses
        workers = [
            WorkerAnalysis(worker_id=i, person_detection=p)
            for i, p in enumerate(persons)
        ]

        # Associate each PPE item with the nearest person
        for ppe in ppe_items:
            # Find nearest person
            nearest_worker = None
            min_distance = float("inf")

            for worker in workers:
                distance = ppe.distance_to(worker.person_detection)
                # Normalize distance by person diagonal
                # so association scales with person size in frame
                normalized_dist = distance / (
                    worker.person_detection.diagonal + 1e-6
                )

                if (normalized_dist < min_distance and
                        normalized_dist < settings.max_association_distance):
                    min_distance = normalized_dist
                    nearest_worker = worker

            if nearest_worker is not None:
                nearest_worker.associated_detections.append(ppe)

        return workers

    def analyze_compliance(
        self,
        workers: list[WorkerAnalysis]
    ) -> list[WorkerAnalysis]:
        """
        Determines compliance status for each worker based on
        their associated PPE detections.

        Compliance rules (from settings.required_ppe):
        - Every worker must have Hardhat
        - Every worker must have Safety Vest

        Uncertainty handling:
        If a worker has no PPE associated AND confidence is moderate,
        flag for VLM verification — YOLO may have missed small items.

        Args:
            workers: list from associate_ppe_with_workers()

        Returns:
            Same list with compliance fields populated
        """
        for worker in workers:
            associated_classes = [
                d.class_name for d in worker.associated_detections
            ]
            associated_confidences = [
                d.confidence for d in worker.associated_detections
            ]

            # Check PPE presence
            worker.has_hardhat = "Hardhat" in associated_classes
            worker.has_safety_vest = "Safety Vest" in associated_classes
            worker.no_hardhat_detected = "NO-Hardhat" in associated_classes
            worker.no_vest_detected = "NO-Safety Vest" in associated_classes

            # Determine violations
            violations = []

            # Hardhat check
            if worker.no_hardhat_detected:
                # Explicit violation detected
                violations.append("NO-Hardhat")
            elif not worker.has_hardhat:
                # No hardhat detected — could be missed by YOLO
                # Flag for verification if worker confidence is moderate
                if worker.person_detection.confidence < settings.verification_threshold:
                    worker.needs_verification = True
                    worker.verification_reason = (
                        "No hardhat detected but person confidence is low — "
                        "may be occluded or missed"
                    )
                else:
                    # High confidence person but no hardhat — likely violation
                    violations.append("NO-Hardhat (inferred)")

            # Safety vest check
            if worker.no_vest_detected:
                violations.append("NO-Safety Vest")
            elif not worker.has_safety_vest:
                if worker.person_detection.confidence < settings.verification_threshold:
                    worker.needs_verification = True
                    worker.verification_reason += (
                        " No safety vest detected."
                    )
                else:
                    violations.append("NO-Safety Vest (inferred)")

            # Uncertain detection — low confidence on violation class
            for detection in worker.associated_detections:
                if (detection.class_name in settings.violation_classes and
                        detection.confidence < settings.verification_threshold):
                    worker.needs_verification = True
                    worker.verification_reason = (
                        f"Low confidence ({detection.confidence:.2f}) on "
                        f"{detection.class_name} — needs visual verification"
                    )

            worker.violations = violations
            worker.is_compliant = len(violations) == 0 and not worker.needs_verification

        return workers

    def analyze_frame(self, image_path: str) -> FrameAnalysis:
        """
        Complete pipeline: detect → associate → analyze compliance.

        This is the single entry point for frame analysis.
        Returns a FrameAnalysis with full compliance information.

        Args:
            image_path: path to image file

        Returns:
            FrameAnalysis with per-worker compliance status
        """
        # Step 1: Raw YOLO detection
        detections = self.detect(image_path)

        # Step 2: Associate PPE with workers
        workers = self.associate_ppe_with_workers(detections)

        # Step 3: Analyze compliance per worker
        workers = self.analyze_compliance(workers)

        # Step 4: Frame-level statistics
        total = len(workers)
        compliant = sum(1 for w in workers if w.is_compliant)
        violations = sum(1 for w in workers if w.violations)
        needs_verify = sum(1 for w in workers if w.needs_verification)
        compliance_rate = compliant / total if total > 0 else 1.0

        return FrameAnalysis(
            image_path=image_path,
            total_workers=total,
            compliant_workers=compliant,
            violation_workers=violations,
            needs_verification=needs_verify,
            worker_analyses=workers,
            all_detections=detections,
            compliance_rate=compliance_rate
        )

    def draw_results(
        self,
        image_path: str,
        frame_analysis: FrameAnalysis,
        output_path: str = None
    ) -> np.ndarray:
        """
        Draws detection results on the image with color-coded boxes.

        Color coding:
        Green  → compliant worker
        Red    → worker with violations
        Yellow → worker needing verification
        Blue   → PPE item (compliant)
        Orange → violation item

        Args:
            image_path:    source image
            frame_analysis: result from analyze_frame()
            output_path:   if provided, saves annotated image here

        Returns:
            Annotated image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Draw worker boxes
        # app/core/detector.py — fix label positioning in draw_results

        for worker in frame_analysis.worker_analyses:
            bbox = worker.person_detection.bbox
            x1, y1, x2, y2 = map(int, bbox)

            if worker.is_compliant:
                color = (0, 255, 0)
                status = "COMPLIANT"
            elif worker.needs_verification:
                color = (0, 255, 255)
                status = "VERIFY"
            else:
                color = (0, 0, 255)
                status = "VIOLATION"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Fix: draw label INSIDE the box if worker is at top of image
            # so text is never cut off by image boundary
            label = f"Worker {worker.worker_id}: {status}"
            label_y = y1 + 25 if y1 < 30 else y1 - 10  # ← move inside if near top

            cv2.putText(
                img, label, (x1 + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Draw violations below the bottom of the box
            if worker.violations:
                violation_text = " | ".join(worker.violations)
                cv2.putText(
                    img, violation_text, (x1 + 5, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )

        # Draw PPE detections
        ppe_colors = {
            "Hardhat": (0, 255, 0),          # green
            "Safety Vest": (0, 255, 0),       # green
            "NO-Hardhat": (0, 0, 255),        # red
            "NO-Safety Vest": (0, 0, 255),    # red
        }

        for detection in frame_analysis.all_detections:
            if detection.class_name == "Person":
                continue  # already drawn above

            bbox = detection.bbox
            x1, y1, x2, y2 = map(int, bbox)
            color = ppe_colors.get(detection.class_name, (128, 128, 128))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        # Frame summary in top left
        summary = (
            f"Workers: {frame_analysis.total_workers} | "
            f"Compliant: {frame_analysis.compliant_workers} | "
            f"Violations: {frame_analysis.violation_workers} | "
            f"Rate: {frame_analysis.compliance_rate:.0%}"
        )
        cv2.putText(
            img, summary, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Annotated image saved to: {output_path}")

        return img