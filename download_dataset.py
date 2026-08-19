from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

def download_dataset():
    """
        Downloads PPE detection dataset from Roboflow.

        Why Roboflow?
        It handles dataset versioning, format conversion, and augmentation.
        The YOLOv11 format gives us the exact folder structure YOLO expects:
        - images/train, images/val, images/test
        - labels/train, labels/val, labels/test
        - data.yaml with class names and paths
    """
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    # Replace these with values from your chosen dataset
    # Found on the dataset page: workspace/project/version
    project = rf.workspace("construction-plxig").project("construction-ppe-detection-oiysp")
    version = project.version(2)
    dataset = version.download(
        "yolov11",
        location="data/raw")

    print(f"Dataset downloaded to: data/raw")
    print(f"Dataset downloaded to: {dataset.location}")

        # Check if files exist there
    from pathlib import Path
    loc = Path(dataset.location)
    print(f"Path exists: {loc.exists()}")
    if loc.exists():
        print("Contents:")
        for item in loc.rglob("*"):
            print(f"  {item}")

    return dataset

if __name__ == "__main__":
    download_dataset()