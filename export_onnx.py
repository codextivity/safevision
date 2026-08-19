# export_onnx.py — place in project root

import os
from pathlib import Path
from ultralytics import YOLO

def export_onnx():
    model_path = "models/trained/best.pt"
    onnx_path = "models/trained/best.onnx"

    # Verify source model exists and is valid
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    size_mb = Path(model_path).stat().st_size / 1e6
    print(f"Source model: {model_path} ({size_mb:.1f} MB)")

    if size_mb < 1:
        raise ValueError(
            f"Source model is too small ({size_mb:.1f} MB) — "
            f"may be a Git LFS pointer file, not the actual weights. "
            f"Run: git lfs pull"
        )

    # Delete existing ONNX if corrupted
    if Path(onnx_path).exists():
        existing_size = Path(onnx_path).stat().st_size / 1e6
        print(f"Removing existing ONNX ({existing_size:.1f} MB)...")
        os.remove(onnx_path)

    print("Loading model...")
    model = YOLO(model_path)

    print("Exporting to ONNX...")
    # Export returns the path to the exported file
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,
        opset=12,      # opset 12 is more widely compatible than 17
    )

    print(f"Export returned: {exported_path}")

    # Verify the exported file
    if exported_path and Path(exported_path).exists():
        onnx_size = Path(exported_path).stat().st_size / 1e6
        print(f"ONNX file size: {onnx_size:.1f} MB")

        if onnx_size < 1:
            raise ValueError(
                f"ONNX file is too small ({onnx_size:.1f} MB) — export failed"
            )

        # Move to correct location if needed
        if str(exported_path) != onnx_path:
            import shutil
            shutil.move(str(exported_path), onnx_path)
            print(f"Moved to: {onnx_path}")

        print(f"✅ ONNX export successful: {onnx_path}")

    else:
        raise RuntimeError("Export failed — no output file produced")

    # Verify it loads correctly
    print("Verifying ONNX model loads...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        inputs = session.get_inputs()
        print(f"✅ ONNX verified — input shape: {inputs[0].shape}")
    except ImportError:
        print("onnxruntime not installed — skipping verification")
        print("Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠ ONNX verification failed: {e}")

if __name__ == "__main__":
    export_onnx()