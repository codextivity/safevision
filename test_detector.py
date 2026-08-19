# test_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.detector import PPEDetector
from app.core.database import initialize_database, store_frame_analysis

# Initialize database
initialize_database()

# Load detector
detector = PPEDetector("models/trained/best.pt")

# Test on a sample image from the test set
test_images = list(Path("data/raw/test/images").glob("*.jpg"))
if not test_images:
    test_images = list(Path("data/raw/test/images").glob("*.png"))

if not test_images:
    print("No test images found")
else:
    test_image = str(test_images[0])
    print(f"Testing on: {test_image}")

    # Run full analysis
    result = detector.analyze_frame(test_image)

    print(f"\nFrame Analysis:")
    print(f"  Total workers:    {result.total_workers}")
    print(f"  Compliant:        {result.compliant_workers}")
    print(f"  Violations:       {result.violation_workers}")
    print(f"  Need verification:{result.needs_verification}")
    print(f"  Compliance rate:  {result.compliance_rate:.1%}")

    print(f"\nWorker Details:")
    for worker in result.worker_analyses:
        status = "COMPLIANT" if worker.is_compliant else "VIOLATION"
        print(f"  Worker {worker.worker_id}: {status}")
        print(f"    Hardhat: {worker.has_hardhat}")
        print(f"    Vest:    {worker.has_safety_vest}")
        if worker.violations:
            print(f"    Violations: {worker.violations}")
        if worker.needs_verification:
            print(f"    Needs verification: {worker.verification_reason}")

    # Save annotated image
    Path("data/samples").mkdir(exist_ok=True)
    detector.draw_results(
        test_image,
        result,
        output_path="data/samples/test_result.jpg"
    )

    # Store in database
    frame_id = store_frame_analysis(result)
    print(f"\nStored in database with frame_id: {frame_id}")
    print(f"Annotated image saved to: data/samples/test_result.jpg")