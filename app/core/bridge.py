# app/core/bridge.py
# Converts raw detection results into LLM-readable summaries.
#
# The bridge layer solves a fundamental problem:
# YOLO outputs numbers and coordinates.
# LLMs understand natural language.
# This module translates between the two.
#
# Why a separate bridge module?
# Keeps detector.py focused on CV logic.
# Keeps agent.py focused on LLM logic.
# The bridge is the seam between two different domains.

from app.core.detector import FrameAnalysis, WorkerAnalysis
from app.core.database import get_compliance_summary, query_violations
from datetime import datetime

def frame_analysis_to_text(frame_analysis: FrameAnalysis) -> str:
    """
    Converts a FrameAnalysis object into a natural language summary.

    This text is what the LangChain agent reads when asked about
    a specific frame. It must be detailed enough to answer follow-up
    questions without referring back to the raw data.

    Args:
        frame_analysis: result from PPEDetector.analyze_frame()

    Returns:
        Natural language description of the frame analysis
    """
    lines = []

    lines.append(
        f"Frame Analysis — {frame_analysis.image_path}"
    )
    lines.append(
        f"Analyzed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"")
    lines.append(
        f"Summary: {frame_analysis.total_workers} workers detected. "
        f"{frame_analysis.compliant_workers} compliant, "
        f"{frame_analysis.violation_workers} with violations, "
        f"{frame_analysis.needs_verification} need verification. "
        f"Overall compliance rate: {frame_analysis.compliance_rate:.1%}"
    )

    if not frame_analysis.worker_analyses:
        lines.append(
            "No workers detected in this frame. "
            "The image may not contain people or the confidence "
            "threshold filtered all detections."
        )
        return "\n".join(lines)

    lines.append(f"\nWorker Details:")

    for worker in frame_analysis.worker_analyses:
        lines.append(f"\n  Worker {worker.worker_id}:")
        lines.append(
            f"    Detection confidence: "
            f"{worker.person_detection.confidence:.2f}"
        )

        # PPE status
        ppe_status = []
        if worker.has_hardhat:
            ppe_status.append("wearing hardhat ✅")
        if worker.has_safety_vest:
            ppe_status.append("wearing safety vest ✅")
        if worker.no_hardhat_detected:
            ppe_status.append("NO hardhat detected ❌")
        if worker.no_vest_detected:
            ppe_status.append("NO safety vest detected ❌")

        if ppe_status:
            lines.append(f"    PPE status: {', '.join(ppe_status)}")

        # Compliance verdict
        if worker.is_compliant:
            lines.append(f"    Verdict: COMPLIANT — all required PPE present")
        elif worker.needs_verification:
            lines.append(
                f"    Verdict: UNCERTAIN — needs GPT-4o verification"
            )
            lines.append(
                f"    Reason: {worker.verification_reason}"
            )
        else:
            lines.append(
                f"    Verdict: VIOLATION — {', '.join(worker.violations)}"
            )

    return "\n".join(lines)

def compliance_summary_to_text(
    summary: dict,
    period_label: str = "all time"
) -> str:
    """
    Converts database compliance summary to natural language.

    Used by the LangChain agent when answering questions like
    "what is our compliance rate?" or "how many violations today?"

    Args:
        summary:      dict from get_compliance_summary()
        period_label: human-readable period description

    Returns:
        Natural language compliance report
    """
    total_frames = summary["total_frames_analyzed"]
    total_workers = summary["total_workers_detected"]
    compliant = summary["compliant_workers"]
    violations = summary["violation_workers"]
    avg_rate = summary["avg_compliance_rate"]
    by_type = summary["violations_by_type"]

    if total_frames == 0:
        return "No frames have been analyzed yet."

    lines = [
        f"Compliance Summary ({period_label}):",
        f"",
        f"  Frames analyzed:    {total_frames}",
        f"  Workers detected:   {total_workers}",
        f"  Compliant workers:  {compliant}",
        f"  Violation workers:  {violations}",
        f"  Avg compliance rate: {avg_rate:.1%}",
        f"",
        f"  Violations by type:",
    ]

    if not by_type:
        lines.append("    No violations recorded")
    else:
        for vtype, count in sorted(
            by_type.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"    {vtype:35} {count} incidents")

    # Add safety assessment
    lines.append("")
    if avg_rate >= 0.90:
        lines.append("  Assessment: ✅ Excellent compliance — site is safe")
    elif avg_rate >= 0.75:
        lines.append("  Assessment: ⚠ Good compliance — minor improvements needed")
    elif avg_rate >= 0.50:
        lines.append("  Assessment: ⚠ Moderate compliance — intervention recommended")
    else:
        lines.append("  Assessment: ❌ Poor compliance — immediate action required")

    return "\n".join(lines)