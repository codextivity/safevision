# mlflow_tracking/drift_detection.py
# Updated for Evidently 0.7.x API

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

DB_PATH = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

def extract_image_features(
    image_dir: str,
    sample_size: int = 200
) -> pd.DataFrame:
    """
    Extracts numerical features from images for drift detection.

    Features extracted:
    - brightness: average pixel intensity
    - num_objects: number of annotated objects per image
    - mean_bbox_area_pct: average bounding box size as % of image
    - per-class object counts

    These statistics capture the distribution of image content
    without comparing raw pixels (too high dimensional).
    """
    import yaml
    import cv2

    image_dir = Path(image_dir)

    # Labels are in parallel directory structure
    # images/train → labels/train
    parts = image_dir.parts
    label_parts = list(parts)
    img_idx = None
    for i, p in enumerate(parts):
        if p == "images":
            img_idx = i
            break

    if img_idx is not None:
        label_parts[img_idx] = "labels"
    label_dir = Path(*label_parts)

    # Load class names from data.yaml
    yaml_files = list(Path("data/raw").rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError("data.yaml not found in data/raw")

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    # Collect all images
    images = (
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # Sample for efficiency
    import random
    random.seed(42)  # reproducible sampling
    if len(images) > sample_size:
        images = random.sample(images, sample_size)

    print(f"  Processing {len(images)} images from {image_dir.name}/")

    records = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Load labels
        label_path = label_dir / (img_path.stem + ".txt")
        boxes = []
        class_counts = {
            name.replace(" ", "_").replace("-", "_"): 0
            for name in class_names
        }

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        parts_line = line.split()
                        class_id = int(parts_line[0])
                        bw = float(parts_line[3])
                        bh = float(parts_line[4])
                        area = bw * bh * 100
                        boxes.append(area)
                        safe_name = (
                            class_names[class_id]
                            .replace(" ", "_")
                            .replace("-", "_")
                        )
                        class_counts[safe_name] += 1

        record = {
            "brightness":           brightness,
            "contrast":             contrast,
            "num_objects":          len(boxes),
            "mean_bbox_area_pct":   float(np.mean(boxes)) if boxes else 0.0,
            "max_bbox_area_pct":    float(np.max(boxes)) if boxes else 0.0,
            "std_bbox_area_pct":    float(np.std(boxes)) if boxes else 0.0,
        }
        record.update(class_counts)
        records.append(record)

    df = pd.DataFrame(records)
    print(f"  Extracted {len(df)} feature records")
    print(f"  Columns: {list(df.columns)}")
    return df

def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "data/drift_report.html"
) -> dict:
    """
    Runs statistical drift detection using scipy KS test.
    Works regardless of Evidently API version.
    """
    from scipy import stats

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    drift_summary = {
        "reference_samples": len(reference_df),
        "current_samples":   len(current_df),
        "analyzed_at":       datetime.now().isoformat(),
        "method":            "KS test (scipy)",
    }

    drift_results = {}
    drifted = 0

    for col in reference_df.columns:
        if col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna()
        cur_vals = current_df[col].dropna()

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        # Kolmogorov-Smirnov test
        # Tests whether two samples come from the same distribution
        # p_value < 0.05 → statistically significant drift detected
        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        is_drifted = bool(p_value < 0.05)

        drift_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":      round(float(p_value), 4),
            "drifted":      is_drifted,
            "ref_mean":     round(float(ref_vals.mean()), 4),
            "cur_mean":     round(float(cur_vals.mean()), 4),
            "mean_change":  round(float(cur_vals.mean() - ref_vals.mean()), 4),
        }

        if is_drifted:
            drifted += 1

    drift_summary["manual_drift"]   = drift_results
    drift_summary["drifted_columns"] = drifted
    drift_summary["total_columns"]   = len(drift_results)
    drift_summary["share_drifted"]   = (
        round(drifted / len(drift_results), 4) if drift_results else 0
    )

    # Save JSON report
    json_path = output_path.replace(".html", ".json")
    with open(json_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    print(f"  Drift report saved: {json_path}")

    return drift_summary

def run_full_drift_analysis():
    """Runs complete drift analysis and logs results to MLflow."""

    print("=" * 60)
    print("SAFEVISION DRIFT DETECTION")
    print("=" * 60)

    mlflow.set_experiment("safevision-drift-monitoring")

    with mlflow.start_run(
        run_name=f"drift_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ):
        # ── Data drift: train vs test ─────────────────────────────────────────
        print("\n1. Extracting image features...")

        train_img_dir = "data/raw/train/images"
        test_img_dir  = "data/raw/test/images"

        if not Path(train_img_dir).exists():
            print(f"  Directory not found: {train_img_dir}")
            return

        reference_df = extract_image_features(train_img_dir, sample_size=200)
        current_df   = extract_image_features(test_img_dir,  sample_size=200)

        print("\n2. Running drift analysis...")
        drift_summary = run_drift_report(
            reference_df=reference_df,
            current_df=current_df,
            output_path="data/drift_report.html"
        )

        # ── Log to MLflow ─────────────────────────────────────────────────────
        print("\n3. Logging to MLflow...")

        mlflow.log_params({
            "reference_dir":       train_img_dir,
            "current_dir":         test_img_dir,
            "reference_samples":   drift_summary["reference_samples"],
            "current_samples":     drift_summary["current_samples"],
            "api_version":         drift_summary.get("api_version", "unknown"),
        })

        # Log drift metrics if available
        if "drifted_columns" in drift_summary:
            mlflow.log_metrics({
                "drifted_columns":  drift_summary["drifted_columns"],
                "total_columns":    drift_summary["total_columns"],
                "share_drifted":    drift_summary["share_drifted"],
            })

        # Log per-feature drift if available (manual detection)
        if "manual_drift" in drift_summary:
            print("\nPer-feature drift results:")
            print(f"  {'Feature':30} {'KS stat':>8} {'p-value':>8} {'Drifted':>8}")
            print(f"  {'-'*60}")

            for col, result in drift_summary["manual_drift"].items():
                status = "⚠ YES" if result["drifted"] else "✅ no"
                print(
                    f"  {col:30} "
                    f"{result['ks_statistic']:>8.4f} "
                    f"{result['p_value']:>8.4f} "
                    f"{status:>8}"
                )

                mlflow.log_metric(
                    f"drift_ks_{col[:20]}",
                    result["ks_statistic"]
                )

        # Log model performance from eval metrics
        eval_path = Path("data/eval_metrics.json")
        if eval_path.exists():
            with open(eval_path) as f:
                eval_metrics = json.load(f)

            print(f"\nCurrent model performance:")
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
                    mlflow.log_metric(f"model_{k}", v)

        # Save and log summary
        summary_path = Path("data/drift_summary.json")
        with open(summary_path, "w") as f:
            json.dump(drift_summary, f, indent=2, default=str)

        # Log artifacts
        if Path("data/drift_report.html").exists():
            mlflow.log_artifact("data/drift_report.html", "reports")
        if Path("data/drift_report.json").exists():
            mlflow.log_artifact("data/drift_report.json", "reports")
        mlflow.log_artifact(str(summary_path), "reports")

        print(f"\n{'='*60}")
        print(f"Drift analysis complete")
        print(f"Drifted features: "
              f"{drift_summary.get('drifted_columns', '?')}/"
              f"{drift_summary.get('total_columns', '?')}")
        print(f"View MLflow at: http://localhost:5000")
        print(f"  → Experiment: safevision-drift-monitoring")

if __name__ == "__main__":
    run_full_drift_analysis()