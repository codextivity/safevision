# generate_traffic.py — place in project root
# Sends test requests to the API to populate Prometheus metrics

import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

test_images = list(Path("data/raw/test/images").glob("*.jpg"))[:10]

print(f"Sending {len(test_images)} detection requests...")

for i, img_path in enumerate(test_images):
    # Detection request
    with open(img_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/detect",
            files={"file": (img_path.name, f, "image/jpeg")},
            params={"return_image": False}
        )

    if response.status_code == 200:
        data = response.json()
        print(f"  [{i+1}] Workers: {data['total_workers']} "
              f"| Compliance: {data['compliance_rate']:.1%}")
    else:
        print(f"  [{i+1}] Error: {response.status_code}")

    time.sleep(1)  # 1 request per second

# Query requests
questions = [
    "What is our compliance rate?",
    "Which violation type is most common?",
    "How many workers were detected?",
]

print("\nSending query requests...")
for q in questions:
    response = requests.post(
        f"{API_URL}/query",
        json={"question": q, "history": []}
    )
    if response.status_code == 200:
        print(f"  Q: {q[:50]}")
        print(f"  A: {response.json()['answer'][:100]}...")
    time.sleep(2)

print("\nDone. Check Grafana at http://localhost:3000")