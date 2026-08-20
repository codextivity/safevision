# mlflow_tracking/validate_data.py
# Stage 1 of the DVC pipeline — validates dataset before training.
#
# Why validate before training?
# Training on a corrupted or incomplete dataset wastes GPU time.
# This script catches problems early and documents dataset state.

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from datetime import datetime

def validate_dataset(data_root: str = "data/raw") -> dict:
    """
    Validates the PPE dataset and returns a report.
    Saves report to data/validation_report.json for DVC tracking.
    """
    data_root = Path(data_root)
    report = {
        "validated_at": datetime.now().isoformat(),
        "data_root": str(data_root),
        "status": "unknown",
        "issues": [],
        "stats": {}
    }

    # Check data.yaml exists
    yaml_files = list(data_root.rglob("data.yaml"))
    if not yaml_files:
        report["status"] = "failed"
        report["issues"].append("data.yaml not found")
        return report

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)

    report["stats"]["num_classes"] = config["nc"]
    report["stats"]["class_names"] = config["names"]

    # Check each split
    total_images = 0
    total_labels = 0

    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if not img_dir.exists():
            report["issues"].append(f"Missing directory: {img_dir}")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        report["stats"][f"{split}_images"] = len(images)
        report["stats"][f"{split}_labels"] = len(labels)

        total_images += len(images)
        total_labels += len(labels)

        # Check image-label correspondence
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_labels = img_stems - lbl_stems

        if missing_labels:
            report["issues"].append(
                f"{split}: {len(missing_labels)} images missing labels"
            )

    report["stats"]["total_images"] = total_images
    report["stats"]["total_labels"] = total_labels

    # Set overall status
    if report["issues"]:
        report["status"] = "warning"
    else:
        report["status"] = "passed"

    # Save report
    output_path = Path("data/validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validation status: {report['status']}")
    print(f"Total images: {total_images}")
    print(f"Issues found: {len(report['issues'])}")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ⚠ {issue}")

    return report

if __name__ == "__main__":
    report = validate_dataset()
    if report["status"] == "failed":
        sys.exit(1)
    print(f"Report saved to: data/validation_report.json")