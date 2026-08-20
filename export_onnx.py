# export_onnx.py — full fixed version

import os
from pathlib import Path
from ultralytics import YOLO

def export_onnx():
    model_path = "models/trained/best_nano.pt"

    # Destination — where DVC expects the ONNX file
    output_dir = Path("models/deployed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_nano.onnx"

    # Verify source model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    size_mb = Path(model_path).stat().st_size / 1e6
    print(f"Source model: {model_path} ({size_mb:.1f} MB)")

    if size_mb < 1:
        raise ValueError(
            f"Source model too small ({size_mb:.1f} MB) — "
            f"may be a Git LFS pointer. Run: git lfs pull"
        )

    # Delete existing ONNX files to avoid conflicts
    # Ultralytics exports to same directory as the .pt file by default
    default_onnx = Path("models/trained/best_nano.onnx")
    if default_onnx.exists():
        print(f"Removing existing ONNX: {default_onnx}")
        default_onnx.unlink()

    if output_path.exists():
        print(f"Removing existing ONNX: {output_path}")
        output_path.unlink()

    print("Loading model...")
    model = YOLO(model_path)

    print("Exporting to ONNX...")
    exported = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,
        opset=12,
    )

    # exported is a Path object from Ultralytics
    exported_path = Path(exported)
    print(f"Ultralytics saved ONNX to: {exported_path}")
    print(f"ONNX exists: {exported_path.exists()}")

    if not exported_path.exists():
        raise RuntimeError(f"Export failed — file not found: {exported_path}")

    # Copy to DVC output location
    import shutil
    shutil.copy2(str(exported_path), str(output_path))
    print(f"Copied to DVC output: {output_path}")

    # Remove the Ultralytics default location copy
    if exported_path != output_path and exported_path.exists():
        exported_path.unlink()
        print(f"Removed intermediate file: {exported_path}")

    # Verify final output
    final_size = output_path.stat().st_size / 1e6
    print(f"Final ONNX: {output_path} ({final_size:.1f} MB)")

    if final_size < 1:
        raise ValueError(f"ONNX file too small: {final_size:.1f} MB")

    print("✅ ONNX export successful")
    return str(output_path)

if __name__ == "__main__":
    export_onnx()