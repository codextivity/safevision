# audit_dataset.py
# Quantifies annotation quality issues found during visual inspection.
# Run this before training to understand data quality objectively.

import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# audit_dataset.py — fix case-insensitive person detection

def audit_annotations(data_root: str = "data/raw"):
    data_root = Path(data_root)
    yaml_files = list(data_root.rglob("data.yaml"))

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)

    class_names = config["names"]
    print(f"Classes: {class_names}")

    # Fix: case-insensitive person class detection
    person_idx = -1
    for i, name in enumerate(class_names):
        if name.lower() in ["person", "worker", "human", "people"]:
            person_idx = i
            print(f"Person class: '{name}' at index {i}")
            break

    if person_idx == -1:
        print("Person class: NOT FOUND")
    
    # Fix: case-insensitive PPE index detection
    ppe_indices = [
        i for i, n in enumerate(class_names)
        if n.lower() not in ["person", "worker", "human", "people"]
    ]

    results = {}

    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))
        total = len(label_files)

        no_annotations = 0
        no_person = 0
        person_only = 0
        has_person_and_ppe = 0
        ppe_without_person = []

        for label_file in label_files:
            with open(label_file) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                no_annotations += 1
                continue

            classes_in_image = [int(l.split()[0]) for l in lines]
            has_person = person_idx in classes_in_image
            has_ppe = any(c in ppe_indices for c in classes_in_image)

            if has_person and has_ppe:
                has_person_and_ppe += 1
            elif has_ppe and not has_person:
                no_person += 1
                ppe_without_person.append(label_file.name)
            elif has_person and not has_ppe:
                person_only += 1

        results[split] = {
            "total": total,
            "no_annotations": no_annotations,
            "no_person_label": no_person,
            "person_only": person_only,
            "has_person_and_ppe": has_person_and_ppe,
        }

        print(f"\n{split.upper()} split ({total} images):")
        print(f"  Complete annotations:     {has_person_and_ppe:5} "
              f"({has_person_and_ppe/total*100:.1f}%)")
        print(f"  PPE without person label: {no_person:5} "
              f"({no_person/total*100:.1f}%)")
        print(f"  Person without PPE:       {person_only:5} "
              f"({person_only/total*100:.1f}%)")
        print(f"  Empty label files:        {no_annotations:5} "
              f"({no_annotations/total*100:.1f}%)")

        if ppe_without_person:
            print(f"  Sample missing person:")
            for f in ppe_without_person[:3]:
                print(f"    {f}")

    return results


def check_class_confusion(data_root: str = "data/raw"):
    data_root = Path(data_root)
    yaml_files = list(data_root.rglob("data.yaml"))

    with open(yaml_files[0]) as f:
        config = yaml.safe_load(f)

    class_names = config["names"]
    cooccurrence = defaultdict(int)

    for split in ["train", "valid"]:
        lbl_dir = data_root / split / "labels"
        if not lbl_dir.exists():
            continue

        for label_file in lbl_dir.glob("*.txt"):
            with open(label_file) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            classes = list(set(int(l.split()[0]) for l in lines))

            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    pair = tuple(sorted([
                        class_names[classes[i]],
                        class_names[classes[j]]
                    ]))
                    cooccurrence[pair] += 1

    print("\nCLASS CO-OCCURRENCE:")
    for pair, count in sorted(
        cooccurrence.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {pair[0]:20} + {pair[1]:20}: {count:5} images")


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET ANNOTATION AUDIT")
    print("=" * 60)
    audit_annotations()
    check_class_confusion()