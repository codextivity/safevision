# app/core/database.py
# SQLite storage for violation history.
#
# Why SQLite instead of PostgreSQL?
# This is a portfolio project running on a single machine.
# SQLite requires zero setup, has no server to run,
# and handles thousands of records easily.
# The LangChain agent queries this database to answer
# questions like "how many violations occurred today?"
# Migrating to PostgreSQL later is straightforward.

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

from app.config import settings

@contextmanager
def get_connection():
    """
    Context manager for database connections.

    Why a context manager?
    Ensures the connection is always closed properly,
    even if an exception occurs during database operations.
    Prevents connection leaks in long-running applications.
    """
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def initialize_database():
    """
    Creates the database tables if they do not exist.

    Call this once at application startup.
    Uses IF NOT EXISTS so it is safe to call multiple times.
    """
    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript("""
            -- Frame-level analysis results
            CREATE TABLE IF NOT EXISTS frames (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at     TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                total_workers   INTEGER NOT NULL,
                compliant_workers INTEGER NOT NULL,
                violation_workers INTEGER NOT NULL,
                compliance_rate REAL NOT NULL,
                raw_analysis    TEXT NOT NULL  -- JSON blob of full FrameAnalysis
            );

            -- Individual worker violations
            -- One row per worker per frame where violation was detected
            CREATE TABLE IF NOT EXISTS violations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id        INTEGER NOT NULL REFERENCES frames(id),
                detected_at     TEXT NOT NULL,
                worker_id       INTEGER NOT NULL,
                violation_type  TEXT NOT NULL,
                confidence      REAL NOT NULL,
                bbox            TEXT NOT NULL,  -- JSON [x1,y1,x2,y2]
                verified_by_vlm INTEGER DEFAULT 0  -- 0=YOLO, 1=GPT-4o verified
            );

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_violations_detected_at
                ON violations(detected_at);

            CREATE INDEX IF NOT EXISTS idx_violations_type
                ON violations(violation_type);

            CREATE INDEX IF NOT EXISTS idx_frames_analyzed_at
                ON frames(analyzed_at);
        """)

    print(f"Database initialized at {settings.database_path}")

def store_frame_analysis(frame_analysis) -> int:
    """
    Stores a complete frame analysis result in the database.

    Args:
        frame_analysis: FrameAnalysis object from detector

    Returns:
        The frame_id of the inserted record
    """
    now = datetime.now().isoformat()

    with get_connection() as conn:
        # Insert frame record
        cursor = conn.execute(
            """INSERT INTO frames
               (analyzed_at, image_path, total_workers, compliant_workers,
                violation_workers, compliance_rate, raw_analysis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                frame_analysis.image_path,
                frame_analysis.total_workers,
                frame_analysis.compliant_workers,
                frame_analysis.violation_workers,
                frame_analysis.compliance_rate,
                json.dumps(frame_analysis.to_dict())
            )
        )
        frame_id = cursor.lastrowid

        # Insert individual violations
        for worker in frame_analysis.worker_analyses:
            for violation in worker.violations:
                conn.execute(
                    """INSERT INTO violations
                       (frame_id, detected_at, worker_id, violation_type,
                        confidence, bbox, verified_by_vlm)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        frame_id,
                        now,
                        worker.worker_id,
                        violation,
                        worker.person_detection.confidence,
                        json.dumps(worker.person_detection.bbox),
                        1 if not worker.needs_verification else 0
                    )
                )

    return frame_id

def query_violations(
    violation_type: str = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 100
) -> list[dict]:
    """
    Queries violation history with optional filters.

    Used by the LangChain agent to answer questions like:
    "How many NO-Hardhat violations occurred today?"
    "What was the compliance rate this week?"

    Args:
        violation_type: filter by "NO-Hardhat" or "NO-Safety Vest"
        date_from:      ISO format date string
        date_to:        ISO format date string
        limit:          maximum records to return

    Returns:
        List of violation dicts
    """
    conditions = []
    params = []

    if violation_type:
        conditions.append("violation_type = ?")
        params.append(violation_type)

    if date_from:
        conditions.append("detected_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("detected_at <= ?")
        params.append(date_to)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT * FROM violations
                {where}
                ORDER BY detected_at DESC
                LIMIT ?""",
            params
        ).fetchall()

    return [dict(row) for row in rows]

def get_compliance_summary(date_from: str = None) -> dict:
    """
    Returns aggregate compliance statistics.

    Used by the LangChain agent for summary questions:
    "What is our overall compliance rate?"
    "Which violation type is most common?"

    Args:
        date_from: optional start date for filtering

    Returns:
        Dict with compliance statistics
    """
    where = f"WHERE analyzed_at >= '{date_from}'" if date_from else ""

    with get_connection() as conn:
        # Overall stats
        frame_stats = conn.execute(
            f"""SELECT
                COUNT(*) as total_frames,
                SUM(total_workers) as total_workers,
                SUM(compliant_workers) as compliant_workers,
                SUM(violation_workers) as violation_workers,
                AVG(compliance_rate) as avg_compliance_rate
                FROM frames {where}"""
        ).fetchone()

        # Violation breakdown
        violation_counts = conn.execute(
            f"""SELECT violation_type, COUNT(*) as count
                FROM violations
                {where.replace('analyzed_at', 'detected_at')}
                GROUP BY violation_type
                ORDER BY count DESC"""
        ).fetchall()

    return {
        "total_frames_analyzed": frame_stats["total_frames"] or 0,
        "total_workers_detected": frame_stats["total_workers"] or 0,
        "compliant_workers": frame_stats["compliant_workers"] or 0,
        "violation_workers": frame_stats["violation_workers"] or 0,
        "avg_compliance_rate": frame_stats["avg_compliance_rate"] or 0,
        "violations_by_type": {
            row["violation_type"]: row["count"]
            for row in violation_counts
        }
    }