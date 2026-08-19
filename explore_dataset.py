# explore_dataset.py

import os
import yaml
from pathlib import Path

def explore_dataset(data_root: str = "data/raw"):
    """
    Explores the downloaded dataset structure and prints a summary.

    YOLO datasets follow a strict structure:
    - data.yaml: class names, train/val/test paths
    - images/: actual image files
    - labels/: annotation files (one .txt per image)

    Each label file contains one line per detected object:
    class_id center_x center_y width height
    All values normalized to 0-1 relative to image dimensions.
    """
    data_root = Path(data_root)

    # Find data.yaml
    yaml_files = list(data_root.rglob("data.yaml"))
    if not yaml_files:
        print("No data.yaml found — check download")
        return

    yaml_path = yaml_files[0]
    print(f"Found data.yaml at: {yaml_path}")

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    print(f"\nDataset Configuration:")
    print(f"  Classes ({config['nc']} total): {config['names']}")
    print(f"  Train path: {config.get('train', 'not specified')}")
    print(f"  Val path:   {config.get('val', 'not specified')}")
    print(f"  Test path:  {config.get('test', 'not specified')}")

    # Count images and labels per split
    print(f"\nDataset Size:")
    for split in ["train", "val", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if img_dir.exists():
            images = list(img_dir.glob("*.jpg")) + \
                     list(img_dir.glob("*.jpeg")) + \
                     list(img_dir.glob("*.png"))
            labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
            print(f"  {split:10} {len(images):5} images | {len(labels):5} labels")

    return config

if __name__ == "__main__":
    config = explore_dataset()