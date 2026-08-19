# train_nano.py
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import shutil
from dotenv import load_dotenv
load_dotenv()

def train_nano():
    model = YOLO("yolo11n.pt")
    run_name = f"ppe_yolo11n_{datetime.now().strftime('%Y%m%d_%H%M')}"

    results = model.train(
        data=str(list(Path("data").rglob("data.yaml"))[0].absolute()),
        imgsz=640,
        epochs=50,
        patience=15,
        batch=32,
        workers=4,
        project="models",
        name=run_name,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cls=1.5,
        mosaic=1.0,
        save=True,
        val=True,
        verbose=True,
    )

    best = Path("models") / run_name / "weights" / "best.pt"
    if best.exists():
        Path("models/trained").mkdir(parents=True, exist_ok=True)
        shutil.copy(best, "models/trained/best_nano.pt")
        print(f"Nano model saved: models/trained/best_nano.pt")

    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    print(f"Nano mAP50: {map50:.4f}")

# ← This is the fix — required on Windows for multiprocessing
if __name__ == "__main__":
    train_nano()