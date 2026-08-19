# train.py

import torch
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import yaml
import shutil
from dotenv import load_dotenv
load_dotenv()

from app.config import settings

device=0 if torch.cuda.is_available() else "cpu"

def verify_dataset() -> str:
    """
    Confirms dataset exists and returns absolute path to data.yaml.
    Fails fast before wasting GPU time on a missing dataset.
    """
    candidates = list(Path("data").rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(
            "data.yaml not found. "
            "Check that dataset is extracted to data/raw/"
        )

    yaml_path = candidates[0]

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    print(f"Dataset: {config['nc']} classes — {config['names']}")
    print(f"data.yaml: {yaml_path}")
    return str(yaml_path.absolute())

def train():
    """
    Trains YOLOv11m on the PPE violation detection dataset.

    Key decisions from EDA:
    - imgsz=1280: 42.9% of hardhat objects are tiny
    - cls=1.5:    increases classification loss weight to help
                  with 5.6x class imbalance between Safety Vest
                  and NO-Safety Vest
    - mosaic=1.0: combines 4 images per training step — violation
                  examples appear more frequently in combined images
    - epochs=50:  sufficient for 5646 training images
    - patience=15: early stopping prevents overfitting
    """
    print("=" * 60)
    print("SAFEVISION YOLO11 TRAINING")
    print("=" * 60)

    yaml_path = verify_dataset()

    # Load base model — downloads automatically if not cached
    model = YOLO(settings.yolo_base_model)
    print(f"\nBase model loaded: {settings.yolo_base_model}")

    run_name = f"ppe_yolo11m_{datetime.now().strftime('%Y%m%d_%H%M')}"

    print(f"\nTraining configuration:")
    print(f"  Image size:   {settings.imgsz}px")
    print(f"  Epochs:       {settings.epochs}")
    print(f"  Batch size:   {settings.batch_size}")
    print(f"  cls weight:   1.5 (elevated for class imbalance)")
    print(f"  Run name:     {run_name}")

    results = model.train(
        data=yaml_path,
        imgsz=settings.imgsz,
        epochs=settings.epochs,
        patience=15,
        batch=settings.batch_size,
        workers=settings.workers,
        project="models",
        name=run_name,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,

        # Classification loss weight
        # Default 0.5 — increased to 1.5 to emphasize correct
        # classification of violation vs compliant classes
        cls=1.5,

        # Augmentation settings
        mosaic=1.0,      # combine 4 images — helps small objects
        degrees=10.0,    # slight rotation — cameras vary in angle
        flipud=0.3,      # vertical flip 30% of the time

        save=True,
        val=True,
        verbose=True,
    )

    # Save best model to standard location
    run_dir = Path("models/runs") / run_name
    best_model = run_dir / "weights" / "best.pt"

    if best_model.exists():
        Path("models/trained").mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model, settings.yolo_model_path)
        print(f"\nBest model saved to: {settings.yolo_model_path}")
    else:
        print(f"\nBest weights at: {run_dir / 'weights' / 'best.pt'}")

    # Print summary
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)

    print(f"\nFinal Results:")
    print(f"  mAP50:    {map50:.4f}")
    print(f"  mAP50-95: {map50_95:.4f}")

    if map50 > 0.75:
        print(f"  ✅ Ready for production testing")
    elif map50 > 0.60:
        print(f"  ⚠ Acceptable — consider more epochs")
    else:
        print(f"  ❌ Needs improvement")

    return results

if __name__ == "__main__":
    train()