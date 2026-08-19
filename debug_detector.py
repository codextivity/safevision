# debug_detector.py — place in project root

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from ultralytics import YOLO
from app.config import settings

model = YOLO("models/trained/best.pt")

# Get the same test image the detector used
test_images = list(Path("data/raw/test/images").glob("*.jpg"))
if not test_images:
    test_images = list(Path("data/raw/test/images").glob("*.png"))

test_image = str(test_images[1])
print(f"Testing image: {test_image}")

# Run with very low confidence threshold to see ALL detections
results = model(
    test_image,
    conf=0.01,        # very low — show everything YOLO sees
    iou=0.45,
    verbose=True,
    device=0
)

print(f"\nAll detections (conf > 0.01):")
for result in results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("  No detections at all")
        continue

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = settings.class_names[class_id]
        print(f"  {class_name:20} conf={confidence:.3f}")

# Also save the raw YOLO visualization
results[0].save(filename="data/samples/debug_raw_yolo.jpg")
print(f"\nRaw YOLO visualization saved to: data/samples/debug_raw_yolo.jpg")
print(f"\nWith production threshold ({settings.confidence_threshold}):")

results2 = model(
    test_image,
    conf=settings.confidence_threshold,
    iou=0.45,
    verbose=False,
    device=0
)

for result in results2:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("  No detections survive the threshold")
    else:
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = settings.class_names[class_id]
            print(f"  {class_name:20} conf={confidence:.3f}  ✅ survives threshold")