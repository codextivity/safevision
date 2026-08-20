# mlflow_tracking/drift_detection.py
# Monitors data and model performance drift for SafeVision.
#
# Two reports generated:
# 1. Data drift report — compares training vs production image features
# 2. Model performance report — tracks detection metrics over time
#
# Run periodically (daily/weekly) in production:
#   python mlflow_tracking/drift_detection.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)

DB_PATH = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def extract_image_features(image_dir: str, sample_size: int = 200) -> pd.DataFrame:
    """
    Extracts numerical features from images for drift detection.

    We cannot compare raw pixels — too high dimensional.
    Instead we extract meaningful statistics that capture
    the distribution of image content:
    - Bounding box sizes (do objects appear larger or smaller?)
    - Class distribution (are there more violations than before?)
    - Image brightness (lighting conditions changed?)
    - Bounding box count per image (scene complexity)

    These features are what Evidently compares between
    training distribution and production distribution.
    """
    import yaml
    import cv2

    image_dir = Path(image_dir)
    label_dir = image_dir.parent.parent / "labels" / image_dir.name

    # Load class names
    yaml_files = list(Path("data/raw").rglob("data.yaml"))
    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    # Sample for efficiency
    import random
    if len(images) > sample_size:
        images = random.sample(images, sample_size)

    records = []

    for img_path in images:
        # Load image for brightness analysis
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        # Load corresponding label file
        label_path = label_dir / (img_path.stem + ".txt")
        boxes = []
        class_counts = {name: 0 for name in class_names}

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        area = bw * bh * 100  # as % of image
                        boxes.append(area)
                        class_counts[class_names[class_id]] += 1

        record = {
            "image_width":          w,
            "image_height":         h,
            "brightness":           brightness,
            "num_objects":          len(boxes),
            "mean_bbox_area_pct":   float(np.mean(boxes)) if boxes else 0,
            "max_bbox_area_pct":    float(np.max(boxes)) if boxes else 0,
            "min_bbox_area_pct":    float(np.min(boxes)) if boxes else 0,
            "std_bbox_area_pct":    float(np.std(boxes)) if boxes else 0,
        }

        # Add per-class counts
        for class_name, count in class_counts.items():
            record[f"count_{class_name.replace(' ', '_').replace('-', '_')}"] = count

        records.append(record)

    return pd.DataFrame(records)

def run_data_drift_report(
    reference_dir: str,
    current_dir: str,
    output_path: str = "data/drift_report.html"
) -> dict:
    """
    Compares image feature distributions between reference and current data.

    reference: training data distribution (the baseline)
    current:   production data or new test data

    Returns drift summary with overall drift detected flag
    and per-feature drift scores.
    """
    print(f"Extracting features from reference: {reference_dir}")
    reference_df = extract_image_features(reference_dir)

    print(f"Extracting features from current: {current_dir}")
    current_df = extract_image_features(current_dir)

    print(f"Reference samples: {len(reference_df)}")
    print(f"Current samples:   {len(current_df)}")

    # Build Evidently report
    # DataDriftPreset checks all columns for statistical drift
    # using appropriate tests per data type
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    # Save HTML report for human review
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    print(f"Drift report saved: {output_path}")

    # Extract summary metrics for MLflow logging
    report_dict = report.as_dict()

    # Get overall drift result
    drift_summary = {
        "report_generated_at":  datetime.now().isoformat(),
        "reference_samples":    len(reference_df),
        "current_samples":      len(current_df),
        "reference_dir":        reference_dir,
        "current_dir":          current_dir,
    }

    # Extract drift metrics from report
    try:
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                drift_summary["dataset_drift_detected"] = result.get(
                    "dataset_drift", False
                )
                drift_summary["drifted_columns"] = result.get(
                    "number_of_drifted_columns", 0
                )
                drift_summary["total_columns"] = result.get(
                    "number_of_columns", 0
                )
                drift_summary["share_drifted"] = result.get(
                    "share_of_drifted_columns", 0
                )
    except Exception as e:
        print(f"Could not extract drift metrics: {e}")
        drift_summary["dataset_drift_detected"] = None

    return drift_summary

def run_model_performance_report(
    metrics_history: list[dict],
    output_path: str = "data/performance_report.html"
) -> dict:
    """
    Tracks model performance metrics over time.

    metrics_history: list of metric dicts from different time periods
    Each dict should have keys like mAP50, precision, recall.

    Evidently detects when metrics drift from their historical baseline.
    """
    if len(metrics_history) < 2:
        print("Need at least 2 time periods to detect performance drift")
        return {}

    # Convert to DataFrames
    reference_metrics = pd.DataFrame([metrics_history[0]])
    current_metrics = pd.DataFrame([metrics_history[-1]])

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(
        reference_data=reference_metrics,
        current_data=current_metrics
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    print(f"Performance report saved: {output_path}")

    return report.as_dict()

def run_full_drift_analysis():
    """
    Runs complete drift analysis and logs results to MLflow.
    """
    print("=" * 60)
    print("SAFEVISION DRIFT DETECTION")
    print("=" * 60)

    mlflow.set_experiment("safevision-drift-monitoring")

    with mlflow.start_run(
        run_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ):
        # ── Data drift: train vs test ─────────────────────────────────────────
        # In production this would be train vs recent production frames
        # For now we use train vs test as a proxy
        print("\n1. Running data drift analysis (train vs test)...")

        train_img_dir = "data/raw/train/images"
        test_img_dir = "data/raw/test/images"

        if Path(train_img_dir).exists() and Path(test_img_dir).exists():
            drift_summary = run_data_drift_report(
                reference_dir=train_img_dir,
                current_dir=test_img_dir,
                output_path="data/drift_report.html"
            )

            # Log drift metrics to MLflow
            mlflow.log_metrics({
                "drifted_columns":    drift_summary.get("drifted_columns", 0),
                "total_columns":      drift_summary.get("total_columns", 0),
                "share_drifted":      drift_summary.get("share_drifted", 0),
            })

            mlflow.log_param(
                "dataset_drift_detected",
                drift_summary.get("dataset_drift_detected", "unknown")
            )

            mlflow.log_artifact("data/drift_report.html", "reports")

            print(f"\nDrift Summary:")
            print(f"  Drift detected: {drift_summary.get('dataset_drift_detected')}")
            print(f"  Drifted features: "
                  f"{drift_summary.get('drifted_columns')}/"
                  f"{drift_summary.get('total_columns')}")
            print(f"  Share drifted: "
                  f"{drift_summary.get('share_drifted', 0):.1%}")

        else:
            print("  Skipping — image directories not found")

        # ── Model performance tracking ────────────────────────────────────────
        print("\n2. Loading historical model metrics...")

        metrics_history = []

        # Load metrics from training runs
        eval_path = Path("data/eval_metrics.json")
        if eval_path.exists():
            with open(eval_path) as f:
                current_metrics = json.load(f)
                current_metrics["period"] = "current"
                metrics_history.append(current_metrics)
                print(f"  Current metrics: mAP50={current_metrics.get('test_mAP50', 0):.4f}")

        # Log current performance to MLflow for trend tracking
        if metrics_history:
            mlflow.log_metrics({
                f"current_{k}": v
                for k, v in metrics_history[-1].items()
                if isinstance(v, (int, float))
            })

        # Save drift summary
        summary_path = Path("data/drift_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "analyzed_at": datetime.now().isoformat(),
                "drift_summary": drift_summary if Path(train_img_dir).exists() else {},
                "metrics_history": metrics_history,
            }, f, indent=2)

        mlflow.log_artifact(str(summary_path), "reports")
        print(f"\nDrift summary saved: {summary_path}")
        print(f"View MLflow at: http://localhost:5000")

if __name__ == "__main__":
    run_full_drift_analysis()