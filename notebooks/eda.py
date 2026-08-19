# notebooks/eda.py
# Exploratory Data Analysis on the PPE dataset.
# Understand class distribution, image quality, and annotation quality
# before training — garbage in, garbage out.

import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

def load_dataset_config(data_root: str = "data/raw") -> dict:
    """Loads dataset configuration from data.yaml."""
    yaml_files = list(Path(data_root).rglob("data.yaml"))
    with open(yaml_files[0]) as f:
        return yaml.safe_load(f)

def analyze_class_distribution(data_root: str, config: dict) -> dict:
    """
    Counts instances of each class across all splits.

    Why this matters:
    Class imbalance is the most common cause of poor YOLO performance.
    If you have 5000 'helmet' annotations and only 200 'no_helmet',
    the model will be good at detecting helmets but miss violations.
    We need to know this before training so we can decide whether
    to use weighted loss, oversample minority classes, or augment.
    """
    class_names = config["names"]
    class_counts = defaultdict(int)
    split_counts = defaultdict(lambda: defaultdict(int))

    for split in ["train", "val", "test"]:
        label_dir = Path(data_root) / split / "labels"
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                lines = f.readlines()

            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_name = class_names[class_id]
                    class_counts[class_name] += 1
                    split_counts[split][class_name] += 1

    return dict(class_counts), dict(split_counts)

def analyze_image_properties(
    data_root: str,
    split: str = "train",
    sample_size: int = 100
) -> dict:
    """
    Analyzes image dimensions and quality for a sample of images.

    Why this matters:
    YOLO resizes all images to a fixed size (default 640x640).
    If your images are very different sizes, aggressive resizing
    may distort small objects. Knowing the size distribution
    helps us choose the right input resolution.
    """
    img_dir = Path(data_root) / split / "images"
    images = list(img_dir.glob("*.jpg")) + \
             list(img_dir.glob("*.jpeg")) + \
             list(img_dir.glob("*.png"))

    # Sample for efficiency
    import random
    sample = random.sample(images, min(sample_size, len(images)))

    widths, heights, aspects = [], [], []

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspects.append(w / h)

    return {
        "count": len(images),
        "width_mean": np.mean(widths),
        "width_std": np.std(widths),
        "height_mean": np.mean(heights),
        "height_std": np.std(heights),
        "aspect_mean": np.mean(aspects),
        "min_size": (min(widths), min(heights)),
        "max_size": (max(widths), max(heights)),
    }

def analyze_bbox_sizes(
    data_root: str,
    config: dict,
    split: str = "train"
) -> dict:
    """
    Analyzes bounding box sizes relative to image dimensions.

    Why this matters for PPE detection:
    Helmets and vests on distant workers appear very small in the image.
    If most bounding boxes are tiny (< 5% of image area), we may need
    to use a higher input resolution (1280 instead of 640) to detect them.
    This is a common mistake — training at 640px when objects are tiny
    gives poor results on the very violations you care most about.
    """
    class_names = config["names"]
    label_dir = Path(data_root) / split / "labels"

    bbox_areas = defaultdict(list)

    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            class_id = int(parts[0])
            # YOLO format: class cx cy w h (normalized 0-1)
            w, h = float(parts[3]), float(parts[4])
            # Area as percentage of image
            area = w * h * 100
            class_name = class_names[class_id]
            bbox_areas[class_name].append(area)

    stats = {}
    for class_name, areas in bbox_areas.items():
        stats[class_name] = {
            "mean_area_pct": np.mean(areas),
            "median_area_pct": np.median(areas),
            "min_area_pct": np.min(areas),
            "max_area_pct": np.max(areas),
            "tiny_objects_pct": sum(1 for a in areas if a < 1.0) / len(areas) * 100
        }

    return stats

def visualize_sample_annotations(
    data_root: str,
    config: dict,
    split: str = "train",
    num_samples: int = 4,
    output_path: str = "notebooks/sample_annotations.png"
):
    """
    Visualizes random samples with their bounding box annotations.
    Saves to a PNG file for inspection.

    Why visualize annotations?
    Label errors are common in crowd-sourced datasets.
    Visually inspecting a sample catches issues like:
    - Wrong class labels (helmet labeled as no_helmet)
    - Missing annotations (worker without annotation)
    - Poor quality annotations (bbox too loose or too tight)
    """
    class_names = config["names"]

    # Color per class for visualization
    colors = [
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 165, 0),  # orange
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 255, 0),  # yellow
    ]

    img_dir = Path(data_root) / split / "images"
    lbl_dir = Path(data_root) / split / "labels"

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    import random
    samples = random.sample(images, min(num_samples, len(images)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load corresponding label file
        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert from normalized YOLO to pixel coordinates
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)

                color = colors[class_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = class_names[class_id]
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        axes[i].imshow(img)
        axes[i].set_title(img_path.name, fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Sample annotations saved to: {output_path}")
    plt.close()

def run_full_eda():
    """Runs the complete EDA pipeline and prints a summary report."""

    print("=" * 60)
    print("SAFEVISION DATASET EDA")
    print("=" * 60)

    config = load_dataset_config()
    class_names = config["names"]

    print(f"\nClasses: {class_names}")

    # Class distribution
    print(f"\n{'─'*40}")
    print("CLASS DISTRIBUTION")
    print(f"{'─'*40}")

    class_counts, split_counts = analyze_class_distribution(
        "data/raw", config
    )

    total = sum(class_counts.values())
    for class_name, count in sorted(
        class_counts.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {class_name:20} {count:6} ({pct:5.1f}%) {bar}")

    # Image properties
    print(f"\n{'─'*40}")
    print("IMAGE PROPERTIES (train split)")
    print(f"{'─'*40}")

    img_props = analyze_image_properties("data/raw", "train")
    print(f"  Total images:  {img_props['count']}")
    print(f"  Avg width:     {img_props['width_mean']:.0f}px "
          f"(±{img_props['width_std']:.0f})")
    print(f"  Avg height:    {img_props['height_mean']:.0f}px "
          f"(±{img_props['height_std']:.0f})")
    print(f"  Size range:    {img_props['min_size']} to {img_props['max_size']}")
    print(f"  Avg aspect:    {img_props['aspect_mean']:.2f}")

    # Bounding box sizes
    print(f"\n{'─'*40}")
    print("BOUNDING BOX SIZES (% of image area)")
    print(f"{'─'*40}")

    bbox_stats = analyze_bbox_sizes("data/raw", config)
    for class_name, stats in sorted(bbox_stats.items()):
        print(f"\n  {class_name}:")
        print(f"    Mean area:     {stats['mean_area_pct']:.2f}%")
        print(f"    Median area:   {stats['median_area_pct']:.2f}%")
        print(f"    Tiny objects:  {stats['tiny_objects_pct']:.1f}% "
              f"(< 1% of image)")

    # Visualize samples
    print(f"\n{'─'*40}")
    print("GENERATING SAMPLE VISUALIZATIONS")
    print(f"{'─'*40}")

    visualize_sample_annotations("data/raw", config)

    # Training recommendations
    print(f"\n{'─'*40}")
    print("TRAINING RECOMMENDATIONS")
    print(f"{'─'*40}")

    # Check for class imbalance
    counts = list(class_counts.values())
    if counts:
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 5:
            print(f"  ⚠ High class imbalance detected (ratio: {imbalance_ratio:.1f}x)")
            print(f"    → Consider using class_weights in training config")
        else:
            print(f"  ✓ Class balance acceptable (ratio: {imbalance_ratio:.1f}x)")

    # Check for tiny objects
    tiny_classes = [
        name for name, stats in bbox_stats.items()
        if stats["tiny_objects_pct"] > 30
    ]
    if tiny_classes:
        print(f"  ⚠ Many tiny objects in: {tiny_classes}")
        print(f"    → Consider training at imgsz=1280 instead of 640")
    else:
        print(f"  ✓ Object sizes acceptable for imgsz=640")

    print(f"\nEDA complete. Check notebooks/sample_annotations.png")

if __name__ == "__main__":
    run_full_eda()