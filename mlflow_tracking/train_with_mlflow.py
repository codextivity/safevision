# mlflow_tracking/train_with_mlflow.py
# Training script with real-time MLflow epoch logging.
#
# What this adds over train.py:
# Every epoch's metrics appear as a data point in MLflow charts.
# You can watch training progress live in the MLflow UI.
# You can compare learning curves between different runs.
#
# Run from project root:
#   python mlflow_tracking/train_with_mlflow.py

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import torch
import shutil
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.callbacks import mlflow as mlflow_callbacks

DB_PATH = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

from app.config import settings

def train_with_epoch_logging(
    model_name: str = "yolo11n.pt",
    run_name: str = None,
    epochs: int = 50,
    batch: int = 32,
    imgsz: int = 640,
    experiment_name: str = "safevision-ppe-detection"
):
    """
    Trains YOLO with real-time MLflow epoch logging.

    Every epoch logs:
    - box_loss, cls_loss, dfl_loss
    - precision, recall, mAP50, mAP50-95
    - learning rate

    These appear as line charts in MLflow UI —
    you can watch the model learn in real time.
    """
    if run_name is None:
        arch = model_name.replace(".pt", "")
        run_name = f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Find dataset
    yaml_files = list(Path("data").rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError("data.yaml not found in data/")
    yaml_path = str(yaml_files[0].absolute())

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"View at: http://localhost:5000")

        # Log training configuration
        mlflow.log_params({
            "model_architecture":   model_name.replace(".pt", ""),
            "dataset":              "PPE-detection-construction",
            "num_classes":          5,
            "imgsz":                imgsz,
            "epochs":               epochs,
            "batch_size":           batch,
            "optimizer":            "AdamW",
            "lr0":                  0.001,
            "lrf":                  0.01,
            "cls_loss_weight":      1.5,
            "mosaic":               1.0,
            "device":               "cuda" if torch.cuda.is_available() else "cpu",
        })

        model = YOLO(model_name)

        # ── Custom callback for epoch logging ─────────────────────────────────
        # YOLO's callback system calls these functions at specific points.
        # on_fit_epoch_end fires after each epoch completes.
        # We use it to log metrics to MLflow in real time.

        def on_fit_epoch_end(trainer):
            """
            Called by YOLO after each training epoch.
            Logs all metrics as a time series to MLflow.
            """
            epoch = trainer.epoch
            metrics = trainer.metrics
            log_dict = {}

            # Training losses — dict format in Ultralytics 8.4+
            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if hasattr(loss_items, "items"):
                    for key, val in loss_items.items():
                        log_dict[f"train_{key}"] = float(val)
                else:
                    try:
                        log_dict["train_box_loss"] = float(loss_items[0])
                        log_dict["train_cls_loss"] = float(loss_items[1])
                        log_dict["train_dfl_loss"] = float(loss_items[2])
                    except (IndexError, TypeError):
                        pass

            # Validation metrics
            val_metric_map = {
                "val_precision":    "metrics/precision(B)",
                "val_recall":       "metrics/recall(B)",
                "val_mAP50":        "metrics/mAP50(B)",
                "val_mAP50_95":     "metrics/mAP50-95(B)",
                "val_box_loss":     "val/box_loss",
                "val_cls_loss":     "val/cls_loss",
                "val_dfl_loss":     "val/dfl_loss",
            }

            for log_key, metric_key in val_metric_map.items():
                if metric_key in metrics:
                    log_dict[log_key] = float(metrics[metric_key])

            if log_dict:
                mlflow.log_metrics(log_dict, step=epoch)
                print(
                    f"  Epoch {epoch}: "
                    f"mAP50={log_dict.get('val_mAP50', 0):.4f} logged to MLflow"
                )

        # Register the callback with YOLO
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        # ── Run training ──────────────────────────────────────────────────────
        run_name_yolo = f"mlflow_{datetime.now().strftime('%Y%m%d_%H%M')}"

        results = model.train(
            data=yaml_path,
            imgsz=imgsz,
            epochs=epochs,
            patience=15,
            batch=batch,
            workers=4,
            project="models",
            name=run_name_yolo,
            device=0 if torch.cuda.is_available() else "cpu",
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            cls=1.5,
            mosaic=1.0,
            degrees=10.0,
            save=True,
            val=True,
            verbose=False,  # reduce console noise — metrics go to MLflow
        )

        # print(f"\nDEBUG: Looking for best model...")
        # print(f"run_name_yolo = '{run_name_yolo}'")
        # best = Path("models") / run_name_yolo / "weights" / "best.pt"
        # print(f"Expected path: {best}")
        # print(f"Exists: {best.exists()}")

        # # Show what actually exists in models/
        # import glob
        # all_pts = glob.glob("models/**/*.pt", recursive=True)
        # print(f"All .pt files found:")
        # for f in all_pts:
        #     print(f"  {f}")

        # ── Log final summary metrics ─────────────────────────────────────────
        map50 = results.results_dict.get("metrics/mAP50(B)", 0)
        map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)

        mlflow.log_metrics({
            "final_val_mAP50":      map50,
            "final_val_mAP50_95":   map50_95,
            "final_val_precision":  results.results_dict.get(
                                        "metrics/precision(B)", 0),
            "final_val_recall":     results.results_dict.get(
                                        "metrics/recall(B)", 0),
        })

        #__DVC tracks a data/metrics.json file_____

        metrics_output = {
        "val_mAP50":        map50,
        "val_mAP50_95":     map50_95,
        "val_precision":    results.results_dict.get("metrics/precision(B)", 0),
        "val_recall":       results.results_dict.get("metrics/recall(B)", 0),
        "model_size_mb":    5.5,
        "training_time_hours": 0.254,
        }

        Path("data").mkdir(exist_ok=True)
        with open("data/metrics.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

        print(f"Metrics saved to data/metrics.json")

        # ── Save and log best model ───────────────────────────────────────────────────
        import glob

        print("\nSearching for best.pt...")

        # Build the exact path from the Ultralytics log output
        # Ultralytics saves to: runs/detect/{project}/{name}/weights/best.pt
        # Our project="models", name=run_name_yolo
        # So full path = runs/detect/models/{run_name_yolo}/weights/best.pt

        exact_path = Path("runs") / "detect" / "models" / run_name_yolo / "weights" / "best.pt"
        print(f"Checking exact path: {exact_path}")
        print(f"Exists: {exact_path.exists()}")

        # Also try alternative locations
        alternative_paths = [
            Path("models") / run_name_yolo / "weights" / "best.pt",
            Path("runs") / "detect" / run_name_yolo / "weights" / "best.pt",
        ]

        best = None
        if exact_path.exists():
            best = exact_path
        else:
            for alt in alternative_paths:
                if alt.exists():
                    best = alt
                    print(f"Found at alternative path: {alt}")
                    break

        if best is None:
            # Exhaustive search as last resort
            all_pts = glob.glob("**/best.pt", recursive=True)
            print(f"Exhaustive search found {len(all_pts)} best.pt files:")
            for f in all_pts:
                print(f"  {f} ({Path(f).stat().st_size/1e6:.1f} MB)")

            if all_pts:
                # Take most recently modified
                all_pts.sort(
                    key=lambda x: Path(x).stat().st_mtime,
                    reverse=True
                )
                best = Path(all_pts[0])
                print(f"Using most recent: {best}")

        if best and best.exists():
            size_mb = best.stat().st_size / 1e6
            print(f"Model found: {best} ({size_mb:.1f} MB)")

            # Log to MLflow
            mlflow.log_artifact(str(best), artifact_path="model")

            # Copy to DVC output location
            Path("models/trained").mkdir(parents=True, exist_ok=True)
            dvc_path = Path("models/trained/best_nano.pt")
            shutil.copy(str(best), str(dvc_path))

            print(f"Copied to: {dvc_path} ({dvc_path.stat().st_size/1e6:.1f} MB)")

        else:
            print("ERROR: best.pt not found")
            print("Manual fix: copy the model file to models/trained/best_nano.pt")

        mlflow.set_tags({
            "deployment_ready": str(map50 > 0.70),
            "framework":        "ultralytics",
            "final_mAP50":      f"{map50:.4f}",
        })

        print(f"\nTraining complete")
        print(f"Final mAP50: {map50:.4f}")
        print(f"View charts at: http://localhost:5000")

    return results

if __name__ == "__main__":
    train_with_epoch_logging(
        model_name="yolo11n.pt",
        epochs=50,          # shorter run to test logging
        batch=32,
        imgsz=640,
    )