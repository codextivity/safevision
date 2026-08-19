# evaluate.py
# Place in project root: D:\Projects\pytorch\safevision\
# Run after training: python evaluate.py

from ultralytics import YOLO
from pathlib import Path
import yaml
from dotenv import load_dotenv
load_dotenv()

from app.config import settings

def evaluate():
    """
    Runs evaluation on the held-out test set.
    Reports per-class metrics so we know which PPE items
    the model detects well and which need improvement.

    Why evaluate on test set separately from training?
    During training, YOLO reports validation metrics each epoch.
    The test set is held out completely — never seen during training.
    Evaluating on test gives us an unbiased measure of real-world
    performance. Using val metrics alone risks overfitting to val set.
    """
    model_path = Path(settings.yolo_model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Run train.py first and copy best.pt to models/trained/"
        )

    print("=" * 60)
    print("SAFEVISION MODEL EVALUATION — TEST SET")
    print("=" * 60)
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))

    # Find data.yaml
    yaml_files = list(Path("data").rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError("data.yaml not found in data/")
    yaml_path = str(yaml_files[0].absolute())
    print(f"Dataset: {yaml_path}")

    # Run validation on test split
    # split="test" ensures we use the held-out test set
    # not the validation set used during training
    results = model.val(
        data=yaml_path,
        split="test",
        imgsz=settings.imgsz,
        conf=settings.confidence_threshold,
        iou=settings.iou_threshold,
        device=0,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — TEST SET")
    print("=" * 60)

    # Overall metrics
    map50 = results.results_dict.get("metrics/mAP50(B)", 0)
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)

    print(f"\nOverall Performance:")
    print(f"  mAP50:        {map50:.4f}")
    print(f"  mAP50-95:     {map50_95:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")

    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"  {'Class':20} {'mAP50':>8} {'Assessment'}")
    print(f"  {'-'*50}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    class_map50s = {}
    if hasattr(results.box, "ap50") and results.ap_class_index is not None:
        for i, class_idx in enumerate(results.ap_class_index):
            class_name = class_names[class_idx]
            ap50 = float(results.box.ap50[i])
            class_map50s[class_name] = ap50

            if ap50 >= 0.75:
                assessment = "✅ Production ready"
            elif ap50 >= 0.50:
                assessment = "⚠ Acceptable"
            else:
                assessment = "❌ Needs improvement"

            print(f"  {class_name:20} {ap50:>8.4f} {assessment}")

    # Safety-specific assessment
    print(f"\nSafety System Assessment:")

    hardhat_ap = class_map50s.get("Hardhat", 0)
    no_hardhat_ap = class_map50s.get("NO-Hardhat", 0)
    vest_ap = class_map50s.get("Safety Vest", 0)
    no_vest_ap = class_map50s.get("NO-Safety Vest", 0)
    person_ap = class_map50s.get("Person", 0)

    print(f"\n  Worker Detection:")
    print(f"    Person mAP50: {person_ap:.4f} "
          f"{'✅' if person_ap > 0.75 else '⚠'}")

    print(f"\n  Compliant PPE Detection:")
    print(f"    Hardhat mAP50:      {hardhat_ap:.4f} "
          f"{'✅' if hardhat_ap > 0.75 else '⚠'}")
    print(f"    Safety Vest mAP50:  {vest_ap:.4f} "
          f"{'✅' if vest_ap > 0.75 else '⚠'}")

    print(f"\n  Violation Detection (critical for safety):")
    print(f"    NO-Hardhat mAP50:      {no_hardhat_ap:.4f} "
          f"{'✅' if no_hardhat_ap > 0.50 else '⚠ GPT-4o verification needed'}")
    print(f"    NO-Safety Vest mAP50:  {no_vest_ap:.4f} "
          f"{'✅' if no_vest_ap > 0.50 else '⚠ GPT-4o verification needed'}")

    print(f"\n  Overall Verdict:")
    if map50 >= 0.75:
        print(f"  ✅ Model ready for production testing (mAP50: {map50:.4f})")
        print(f"     Deploy with GPT-4o verification for low-confidence detections")
    elif map50 >= 0.60:
        print(f"  ⚠ Model acceptable for demo (mAP50: {map50:.4f})")
        print(f"     Consider additional training epochs or data augmentation")
    else:
        print(f"  ❌ Model needs improvement (mAP50: {map50:.4f})")

    print(f"\nInterview talking point:")
    print(f"  'The model achieves {map50:.0%} mAP50 overall, with strong")
    print(f"   performance on compliant PPE detection ({hardhat_ap:.0%} hardhat,")
    print(f"   {vest_ap:.0%} safety vest) and moderate violation detection")
    print(f"   ({no_hardhat_ap:.0%} NO-Hardhat, {no_vest_ap:.0%} NO-Safety Vest).")
    print(f"   Low-confidence violation detections are routed to GPT-4o")
    print(f"   vision verification before storing in the database.'")

    return results

if __name__ == "__main__":
    evaluate()