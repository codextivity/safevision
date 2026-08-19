# mlflow_tracking/log_existing_runs.py
# Retroactively logs completed SafeVision training runs into MLflow.
#
# Why log retroactively?
# We completed two training runs before MLflow was set up.
# Logging them now gives us a complete experiment history
# so future runs can be compared against established baselines.
#
# Run this once from the project root:
#   python mlflow_tracking/log_existing_runs.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import mlflow
from datetime import datetime

# MLflow stores data in mlruns/ in the current directory.
# We set the tracking URI explicitly so it always uses
# the safevision project root regardless of where we run from.
DB_PATH = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

print(f"MLflow tracking URI: {DB_PATH}")
print(f"Open UI at: http://localhost:5000\n")

# ── Experiment: SafeVision PPE Detection ──────────────────────────────────────
# All training runs for this project go under one experiment.
# An experiment is a named group of related runs.

mlflow.set_experiment("safevision-ppe-detection")

# ── Run 1: YOLOv11m — baseline medium model ───────────────────────────────────
# This was our first training attempt.
# Strong accuracy but too large for free tier deployment.

print("Logging Run 1: YOLOv11m baseline...")

with mlflow.start_run(run_name="yolo11m-baseline"):

    # ── Parameters ────────────────────────────────────────────────────────────
    # Parameters are inputs that define the experiment.
    # They do not change during a run — they describe the setup.
    # In MLflow UI you can filter and sort runs by any parameter.

    mlflow.log_params({
        # Model configuration
        "model_architecture":   "yolo11m",
        "parameters_millions":  20.0,
        "gflops":               67.8,

        # Dataset
        "dataset":              "PPE-detection-construction",
        "num_classes":          5,
        "train_images":         5646,
        "val_images":           775,
        "test_images":          326,

        # Training hyperparameters
        "imgsz":                640,
        "epochs":               50,
        "batch_size":           16,
        "optimizer":            "AdamW",
        "lr0":                  0.001,
        "lrf":                  0.01,
        "cls_loss_weight":      1.5,
        "mosaic":               1.0,
        "degrees":              10.0,
        "patience":             15,

        # Hardware
        "device":               "CUDA RTX5080 16GB",
    })

    # ── Validation metrics ────────────────────────────────────────────────────
    # These were reported by YOLO during training on the val split.
    # The val split was seen indirectly during training (early stopping).

    mlflow.log_metrics({
        # Overall
        "val_mAP50":                0.742,
        "val_mAP50_95":             0.516,
        "val_precision":            0.802,
        "val_recall":               0.695,

        # Per-class mAP50 — critical for safety system assessment
        "val_mAP50_Hardhat":        0.890,
        "val_mAP50_NO_Hardhat":     0.462,
        "val_mAP50_NO_Safety_Vest": 0.508,
        "val_mAP50_Person":         0.879,
        "val_mAP50_Safety_Vest":    0.972,
    })

    # ── Test metrics ──────────────────────────────────────────────────────────
    # These were reported by evaluate.py on the held-out test split.
    # The test split was NEVER seen during training — true performance.
    # The gap between val and test reveals overfitting.

    mlflow.log_metrics({
        # Overall
        "test_mAP50":                0.549,
        "test_mAP50_95":             0.427,
        "test_precision":            0.814,
        "test_recall":               0.581,

        # Per-class mAP50 on test
        "test_mAP50_Hardhat":        0.702,
        "test_mAP50_NO_Hardhat":     0.121,
        "test_mAP50_NO_Safety_Vest": 0.335,
        "test_mAP50_Person":         0.693,
        "test_mAP50_Safety_Vest":    0.893,

        # Val-Test gap — higher gap = more overfitting
        "val_test_gap_mAP50":        0.742 - 0.549,
    })

    # ── Operational metrics ───────────────────────────────────────────────────
    # These matter for production decisions — not just accuracy

    mlflow.log_metrics({
        "training_time_hours":   0.814,
        "model_size_mb":         40.5,
        "inference_speed_ms":    2.4,
        "ram_required_mb":       800,   # why it failed on Render free tier
    })

    # ── Tags ──────────────────────────────────────────────────────────────────
    # Tags are string labels — useful for filtering in the UI
    # Unlike params, tags are not compared in charts

    mlflow.set_tags({
        "deployment_ready":      "false",
        "deployment_blocker":    "800MB RAM exceeds Render free tier 512MB",
        "model_format":          "pytorch",
        "framework":             "ultralytics",
        "decision":              "retrain with nano architecture",
    })

    # Log model artifact if it exists locally
    model_path = Path("models/trained/best.pt")
    if model_path.exists():
        mlflow.log_artifact(str(model_path), artifact_path="model/pytorch")
        print("  ✅ Model artifact logged: best.pt")
    else:
        print("  ⚠ best.pt not found — skipping artifact")

print("Run 1 logged ✅\n")

# ── Run 2: YOLOv11n — deployment model ───────────────────────────────────────
# This was our second attempt specifically for deployment.
# Smaller architecture — fits in 512MB RAM.
# Surprisingly outperforms medium on violation classes.

print("Logging Run 2: YOLOv11n deployment...")

with mlflow.start_run(run_name="yolo11n-deployment"):

    mlflow.log_params({
        # Model configuration
        "model_architecture":   "yolo11n",
        "parameters_millions":  2.6,
        "gflops":               6.4,

        # Dataset — same as Run 1
        "dataset":              "PPE-detection-construction",
        "num_classes":          5,
        "train_images":         5646,
        "val_images":           775,
        "test_images":          326,

        # Training hyperparameters
        # batch_size increased to 32 because nano uses less GPU memory
        "imgsz":                640,
        "epochs":               50,
        "batch_size":           32,
        "optimizer":            "AdamW",
        "lr0":                  0.001,
        "lrf":                  0.01,
        "cls_loss_weight":      1.5,
        "mosaic":               1.0,
        "degrees":              10.0,
        "patience":             15,

        "device":               "CUDA RTX5080 16GB",
    })

    mlflow.log_metrics({
        # Overall validation
        "val_mAP50":                0.755,
        "val_mAP50_95":             0.530,
        "val_precision":            0.799,
        "val_recall":               0.694,

        # Per-class validation
        "val_mAP50_Hardhat":        0.887,
        "val_mAP50_NO_Hardhat":     0.469,
        "val_mAP50_NO_Safety_Vest": 0.570,
        "val_mAP50_Person":         0.878,
        "val_mAP50_Safety_Vest":    0.970,
    })

    mlflow.log_metrics({
        "training_time_hours":   0.254,
        "model_size_mb":         5.5,
        "onnx_size_mb":          11.0,
        "inference_speed_ms":    0.5,
        "ram_required_mb":       120,
    })

    mlflow.set_tags({
        "deployment_ready":      "true",
        "deployment_platform":   "Render free tier + local Docker",
        "model_format":          "onnx",
        "framework":             "ultralytics",
        "key_finding": (
            "nano outperforms medium on violation classes — "
            "NO-Safety Vest 0.570 vs 0.508, "
            "NO-Hardhat 0.469 vs 0.462"
        ),
        "hypothesis": (
            "smaller capacity forces more generalizable features, "
            "less overfitting on minority violation classes"
        ),
    })

    # Log both model formats
    nano_pt = Path("models/trained/best_nano.pt")
    nano_onnx = Path("models/trained/best_nano.onnx")

    if nano_pt.exists():
        mlflow.log_artifact(str(nano_pt), artifact_path="model/pytorch")
        print("  ✅ Logged: best_nano.pt")

    if nano_onnx.exists():
        mlflow.log_artifact(str(nano_onnx), artifact_path="model/onnx")
        print("  ✅ Logged: best_nano.onnx")

print("Run 2 logged ✅\n")
print("=" * 50)
print("Both runs logged to MLflow")
print("Open http://localhost:5000 to see them")
print("Select both runs → click Compare to see side by side")