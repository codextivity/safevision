# app/api/routes/violations.py
# Violation history query endpoints.

from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.core.database import query_violations, get_compliance_summary

router = APIRouter()

class ViolationRecord(BaseModel):
    id: int
    detected_at: str
    worker_id: int
    violation_type: str
    confidence: float
    verified_by_vlm: bool

class ComplianceSummary(BaseModel):
    total_frames_analyzed: int
    total_workers_detected: int
    compliant_workers: int
    violation_workers: int
    avg_compliance_rate: float
    violations_by_type: dict

@router.get("/summary", response_model=ComplianceSummary)
async def get_summary(date_from: str = None):
    """
    Returns aggregate compliance statistics.
    Optionally filter by start date (ISO format).
    """
    summary = get_compliance_summary(date_from=date_from)
    return ComplianceSummary(**summary)

@router.get("", response_model=list[ViolationRecord])
async def list_violations(
    violation_type: str = Query(
        default=None,
        description="Filter by 'NO-Hardhat' or 'NO-Safety Vest'"
    ),
    limit: int = Query(default=50, le=500),
    date_from: str = Query(default=None)
):
    """
    Returns violation records with optional filters.
    """
    violations = query_violations(
        violation_type=violation_type,
        date_from=date_from,
        limit=limit
    )

    return [
        ViolationRecord(
            id=v["id"],
            detected_at=v["detected_at"],
            worker_id=v["worker_id"],
            violation_type=v["violation_type"],
            confidence=v["confidence"],
            verified_by_vlm=bool(v["verified_by_vlm"])
        )
        for v in violations
    ]