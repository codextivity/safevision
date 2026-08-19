# test_detector_v2.py — place in project root
# Tests on multiple images to find one with workers

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.detector import PPEDetector
from app.core.database import initialize_database, store_frame_analysis

initialize_database()
detector = PPEDetector("models/trained/best.pt")

test_images = (
    list(Path("data/raw/test/images").glob("*.jpg")) +
    list(Path("data/raw/test/images").glob("*.png"))
)

print(f"Found {len(test_images)} test images")
print("Scanning for images with workers...\n")

Path("data/samples").mkdir(exist_ok=True)

found = 0
for img_path in test_images[:50]:  # check first 50
    result = detector.analyze_frame(str(img_path))

    if result.total_workers > 0:
        found += 1
        print(f"✅ {img_path.name}")
        print(f"   Workers: {result.total_workers} | "
              f"Compliant: {result.compliant_workers} | "
              f"Violations: {result.violation_workers}")

        for worker in result.worker_analyses:
            status = "COMPLIANT" if worker.is_compliant else "VIOLATION"
            print(f"   Worker {worker.worker_id}: {status} "
                  f"| hardhat={worker.has_hardhat} "
                  f"| vest={worker.has_safety_vest} "
                  f"| violations={worker.violations}")

        # Save annotated image for the first one with violations
        if result.violation_workers > 0:
            output = f"data/samples/violation_{img_path.stem}.jpg"
            detector.draw_results(str(img_path), result, output)
            print(f"   Saved: {output}")

            # Store and stop
            store_frame_analysis(result)
            print(f"\nFound a violation image — stopping scan")
            break

    else:
        print(f"  ⬜ {img_path.name} — no workers detected")

if found == 0:
    print("\nNo workers detected in first 50 images.")
    print("This suggests the confidence threshold is too high.")
    print("Run debug_detector.py to see raw detections.")