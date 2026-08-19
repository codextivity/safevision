# batch_test.py — place in project root
# Tests on 50 images and prints summary statistics

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.detector import PPEDetector
from app.core.database import (
    initialize_database,
    store_frame_analysis,
    get_compliance_summary
)

initialize_database()
detector = PPEDetector("models/trained/best.pt")

test_images = (
    list(Path("data/raw/test/images").glob("*.jpg")) +
    list(Path("data/raw/test/images").glob("*.png"))
)

print(f"Running batch test on {min(50, len(test_images))} images...\n")

Path("data/samples").mkdir(exist_ok=True)

frames_with_workers = 0
total_workers = 0
total_violations = 0
total_compliant = 0
needs_verification = 0
violation_types = {}

for img_path in test_images[:50]:
    result = detector.analyze_frame(str(img_path))

    if result.total_workers == 0:
        continue

    frames_with_workers += 1
    total_workers += result.total_workers
    total_violations += result.violation_workers
    total_compliant += result.compliant_workers
    needs_verification += result.needs_verification

    # Count violation types
    for worker in result.worker_analyses:
        for violation in worker.violations:
            violation_types[violation] = violation_types.get(violation, 0) + 1

    # Store in database
    store_frame_analysis(result)

print("=" * 60)
print("BATCH TEST RESULTS")
print("=" * 60)
print(f"\nFrames analyzed:        50")
print(f"Frames with workers:    {frames_with_workers}")
print(f"Frames without workers: {50 - frames_with_workers}")
print(f"\nWorker Statistics:")
print(f"  Total workers detected:  {total_workers}")
print(f"  Compliant workers:       {total_compliant}")
print(f"  Violation workers:       {total_violations}")
print(f"  Need VLM verification:   {needs_verification}")

if total_workers > 0:
    rate = total_compliant / total_workers * 100
    print(f"  Compliance rate:         {rate:.1f}%")

print(f"\nViolation Breakdown:")
for vtype, count in sorted(
    violation_types.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {vtype:30} {count}")

print(f"\nDatabase Summary:")
summary = get_compliance_summary()
print(f"  Total frames stored:     {summary['total_frames_analyzed']}")
print(f"  Total workers stored:    {summary['total_workers_detected']}")
print(f"  Avg compliance rate:     {summary['avg_compliance_rate']:.1%}")
print(f"  Violations by type:      {summary['violations_by_type']}")