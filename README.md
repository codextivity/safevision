# 🦺 SafeVision — PPE Compliance Intelligence System

AI-powered construction site safety monitoring that combines YOLOv11 object detection
with GPT-4o vision verification, a LangChain agent for natural language safety queries,
and a complete MLOps stack for production monitoring and reproducibility.

**GitHub:** https://github.com/codextivity/safevision

---

## What It Does

Upload a construction site image and get instant PPE compliance analysis:

```
POST /detect          → per-worker compliance report with annotated image
POST /query           → natural language safety insights from detection database
GET  /violations      → individual violation records with filters
GET  /violations/summary → aggregate compliance statistics
GET  /health          → service status and model load state
GET  /metrics         → Prometheus metrics endpoint
```

Ask questions in natural language:
```
"What is our compliance rate today?"
"Which violation type is most common?"
"How many workers were detected without hardhats?"
"What safety improvements should we make?"
```

---

## System Architecture

```
Construction Site Image
        │
        ▼
┌───────────────────────┐
│    YOLOv11n Detector  │  Detects: Hardhat, NO-Hardhat,
│    mAP50: 0.755       │           Safety Vest, NO-Safety Vest,
│    5.5MB ONNX model   │           Person
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Spatial Association  │  Pairs each Person with nearby PPE
│  Algorithm            │  Determines per-worker compliance
└──────────┬────────────┘
           │
           ├── High confidence ──► Store in SQLite directly
           │
           └── Uncertain ─────────► GPT-4o Vision Verification
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │   SQLite DB  │
                                    │  Violations  │
                                    └──────┬───────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  LangChain  │
                                    │  Agent      │
                                    │  6 DB tools │
                                    └─────────────┘
                                           │
                                           ▼
                                    Natural Language
                                    Safety Reports
```

---

## Key Engineering Decisions

**Why YOLO + GPT-4o instead of GPT-4o alone?**
YOLO processes images in 4ms at no API cost. GPT-4o takes 2-3 seconds and costs per
call. The tiered system uses YOLO for all frames and GPT-4o only for uncertain
detections — reducing cost by 85-95% while maintaining accuracy.

**Why spatial association instead of direct violation classification?**
Training a binary compliant/non-compliant classifier requires worker-level labels that
most datasets lack. Spatial association pairs person detections with nearby PPE
detections — making compliance logic configurable in code rather than baked into the
model. PPE requirements can be changed per site without retraining.

**Why LangChain agent over a fixed dashboard?**
Fixed dashboards answer predefined questions. The LangChain agent answers any question
grounded in real detection data. Safety managers can ask domain-specific questions
without waiting for a developer to add a new chart.

**Why YOLOv11n over YOLOv11m for deployment?**
The nano model (0.755 mAP50) outperformed the medium model (0.742 mAP50) on violation
classes — NO-Safety Vest 0.570 vs 0.508. The smaller capacity forced more generalizable
features. Additionally, nano uses 120MB RAM vs 800MB for medium, enabling deployment on
memory-constrained environments.

---

## Model Performance

Trained on 5646 construction site images — YOLOv11n, 50 epochs, imgsz=640:

| Class | mAP50 | Assessment |
|---|---|---|
| Safety Vest | 0.970 | Production ready |
| Hardhat | 0.887 | Strong |
| Person | 0.878 | Strong |
| NO-Safety Vest | 0.570 | GPT-4o verification |
| NO-Hardhat | 0.469 | GPT-4o verification |
| Overall | 0.755 | Demo ready |

Violation classes score lower due to class imbalance — only 9-14% of training instances
were violations. Low-confidence detections are automatically routed to GPT-4o vision
for verification.

---

## MLOps Stack

A complete MLOps pipeline built on top of the detection system for production
monitoring, reproducibility, and drift detection.

### MLflow — Experiment Tracking

Every training run is automatically logged with parameters, metrics, and per-epoch
learning curves. The parallel coordinates plot allows visual comparison of
configurations.

Experiments tracked:
- safevision-ppe-detection — 3 training runs with full metrics
- langchain-research-copilot — 2 RAG configuration experiments
- multimodal-doc-intelligence — 1 multimodal experiment
- safevision-drift-monitoring — periodic drift detection runs

Key finding from MLflow comparison:

```
                    YOLOv11n    YOLOv11m    Winner
val_mAP50:          0.755       0.742       nano (better accuracy)
model_size_mb:      5.5         40.5        nano (7x smaller)
training_hours:     0.25        0.81        nano (3x faster)
ram_required_mb:    120         800         nano (6x less RAM)
NO-Safety Vest:     0.570       0.508       nano (better on violations)
```

### DVC — Data and Model Versioning

5-stage reproducible pipeline managed by DVC:

```
data_validation
      │
      ▼
    train
    /   \
   /     \
evaluate  export
   \
    \
   monitor (drift detection)
```

```bash
dvc repro                     # reproduce the entire pipeline
dvc metrics diff HEAD~1 HEAD  # compare metrics between commits
dvc dag                       # visualize pipeline dependencies
dvc push                      # push data and models to remote storage
```

The dvc.lock file records the exact hash of every input and output at every stage —
guaranteeing reproducibility of any historical model.

### Evidently — Drift Detection

KS test across 11 image features comparing training vs production distributions.
Runs automatically as pipeline stage 5 and logs results to MLflow.

```
Feature                  KS stat  p-value  Status
brightness               0.1400   0.0396   DRIFT DETECTED
contrast                 0.0550   0.9238   stable
num_objects              0.0900   0.3935   stable
mean_bbox_area_pct       0.1350   0.0521   stable
Hardhat count            0.1250   0.0878   stable
NO_Hardhat count         0.0550   0.9238   stable
Safety_Vest count        0.0150   1.0000   stable

Result:  1 out of 11 features drifted
Finding: brightness distribution differs between train and test sets
Action:  In production, triggers alert to investigate camera calibration
```

### Prometheus + Grafana — Production Monitoring

7 custom metrics exposed at /metrics and scraped every 15 seconds:

```
DETECTION_REQUESTS_TOTAL       counter — by status (success/error)
DETECTION_LATENCY_SECONDS      histogram — end-to-end request latency
YOLO_INFERENCE_SECONDS         histogram — model inference time only
WORKERS_DETECTED_TOTAL         counter — total workers across all frames
VIOLATIONS_DETECTED_TOTAL      counter — by violation type
COMPLIANCE_RATE_GAUGE          gauge — current frame compliance rate
MEMORY_USAGE_BYTES             gauge — process memory usage
```

Grafana dashboard panels:
- Requests per second — traffic volume per endpoint
- Detection Latency P95 — performance SLA monitoring
- Violations detected per minute — real-time safety trend
- Current Compliance Rate — gauge 0-100%
- Memory Usage MB — resource monitoring

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Object detection | YOLOv11n | PPE and person detection |
| Vision verification | GPT-4o | Uncertain detection verification |
| LLM framework | LangChain + LangGraph | Agent with 6 database tools |
| Database | SQLite | Violation history storage |
| API | FastAPI | HTTP endpoints |
| Experiment tracking | MLflow | Training run comparison and logging |
| Data versioning | DVC | Pipeline and model versioning |
| Drift detection | Evidently + SciPy KS | Feature distribution monitoring |
| Metrics | Prometheus | Real-time metric scraping |
| Dashboards | Grafana | Live production monitoring |
| Observability | LangSmith | LangChain agent tracing |
| Deployment | Docker | Containerization |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service status and component state |
| POST | /detect | Upload image for PPE analysis |
| POST | /query | Natural language safety query |
| GET | /violations | List violation records with filters |
| GET | /violations/summary | Aggregate compliance statistics |
| GET | /metrics | Prometheus metrics endpoint |

---

## Quick Start

### Option 1: Run Locally

```bash
git clone https://github.com/codextivity/safevision
cd safevision
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs

### Option 2: Docker

```bash
git clone https://github.com/codextivity/safevision
cd safevision
cp .env.example .env
docker-compose up
```

### Option 3: MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000

### Option 4: Full DVC Pipeline

```bash
pip install dvc
dvc pull
dvc repro
cat data/drift_report.json
```

### Option 5: Prometheus + Grafana Monitoring

```bash
# Terminal 1 — start API
uvicorn app.main:app --port 8000

# Terminal 2 — start Prometheus
cd /path/to/prometheus
prometheus --config.file=prometheus.yml

# Terminal 3 — start Grafana
docker start grafana
```

Open http://localhost:3000

---

## Configuration

Copy .env.example to .env and fill in your keys:

```bash
OPENAI_API_KEY=your-openai-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=safevision
TAVILY_API_KEY=your-tavily-api-key
YOLO_MODEL_PATH=models/trained/best_nano.pt
CONFIDENCE_THRESHOLD=0.25
VERIFICATION_THRESHOLD=0.70
```

---

## Project Structure

```
safevision/
├── app/
│   ├── main.py                    # FastAPI entry point with lazy loading
│   ├── config.py                  # Typed settings via pydantic-settings
│   ├── api/routes/
│   │   ├── health.py              # GET /health
│   │   ├── detect.py              # POST /detect with Prometheus metrics
│   │   ├── query.py               # POST /query — LangChain agent
│   │   └── violations.py          # GET /violations
│   └── core/
│       ├── detector.py            # YOLO inference + spatial association
│       ├── verifier.py            # GPT-4o vision verification
│       ├── bridge.py              # CV outputs to LLM-readable text
│       ├── agent.py               # LangGraph agent with 6 tools
│       ├── database.py            # SQLite operations
│       └── metrics.py             # Prometheus metric definitions
│
├── mlflow_tracking/
│   ├── log_existing_runs.py       # Retroactive MLflow experiment logging
│   ├── log_langchain_experiments.py
│   ├── train_with_mlflow.py       # Training with real-time epoch logging
│   ├── validate_data.py           # DVC pipeline stage 1
│   └── drift_detection.py         # DVC pipeline stage 5
│
├── models/
│   ├── trained/
│   │   └── best_nano.pt           # Trained YOLOv11n weights (5.5MB)
│   └── deployed/
│       └── best_nano.onnx         # ONNX export for CPU deployment (10.6MB)
│
├── data/
│   ├── metrics.json               # DVC tracked training metrics
│   ├── eval_metrics.json          # DVC tracked evaluation metrics
│   ├── drift_report.json          # Drift detection results
│   └── violations.db              # SQLite violation history
│
├── dvc.yaml                       # 5-stage DVC pipeline definition
├── dvc.lock                       # Exact file hashes for reproducibility
├── params.yaml                    # Centralized training parameters
├── mlflow.db                      # MLflow experiment database
├── train.py                       # Standard training script
├── train_nano.py                  # Nano model training for deployment
├── evaluate.py                    # Test set evaluation
├── export_onnx.py                 # ONNX export pipeline
├── generate_traffic.py            # Prometheus metrics load testing
├── Dockerfile
└── docker-compose.yml
```

---

## Training Your Own Model

```bash
# Download dataset from Roboflow
python download_dataset.py

# Train with MLflow tracking
python mlflow_tracking/train_with_mlflow.py

# Evaluate on test set
python evaluate.py

# Export to ONNX
python export_onnx.py

# Run full DVC pipeline
dvc repro
```

---

## Deployment Notes

The full system requires approximately 400-500MB RAM to run the ONNX inference model
alongside the LangChain agent.

Local deployment is recommended for full functionality:

```bash
docker-compose up
```

Cloud deployment options:
- Render Starter tier ($7/month) — 2GB RAM, works immediately
- Any VPS with 1GB+ RAM — no modification needed
- Split into two microservices — detection API and query API,
  each fits within Render free tier 512MB limit

---

## Interview Talking Points

**On the CV + LLM integration:**
"The key insight was that CV models produce excellent detection data but poor user
interfaces. Raw bounding boxes and confidence scores are meaningless to safety managers.
The LangChain agent translates detection outputs into actionable insights grounded in
real database records from actual detections."

**On the tiered verification approach:**
"I analyzed per-class performance and found explicit violation detection was weak
(0.121 mAP50 on NO-Hardhat) due to class imbalance. Rather than accepting this, I
designed a tiered system where YOLO handles confident detections and GPT-4o verifies
uncertain cases — compensating for the model's weakness without requiring more data."

**On spatial association:**
"Instead of training a binary compliant/non-compliant classifier, I implemented
distance-based spatial association pairing each person detection with nearby PPE
detections. Compliance rules are configurable in code — the same trained model works
for any PPE policy without retraining."

**On the nano vs medium decision:**
"Counterintuitively, YOLOv11n outperformed YOLOv11m on violation classes — 0.570 vs
0.508 mAP50 on NO-Safety Vest. My hypothesis is that the smaller capacity forced more
generalizable features rather than overfitting. MLflow's parallel coordinates plot made
this comparison immediately visible and justified the decision with data."

**On the MLOps stack:**
"I built a complete MLOps stack: MLflow tracks every training experiment with per-epoch
learning curves logged automatically. DVC manages a 5-stage reproducible pipeline with
dvc.lock guaranteeing exact reproducibility of any historical model. Evidently detects
data drift using KS tests across 11 image features — I found brightness drift between
training and test distributions, indicating lighting variation across camera positions.
Prometheus scrapes 7 custom metrics every 15 seconds and Grafana visualizes them as
live dashboards tracking request latency, violation rate, and compliance trends."

---

## Version 2 Roadmap

- Industrial defect detection — extend architecture to PCB and electronics inspection
- TensorRT export — optimize for NVIDIA Jetson edge devices on production lines
- Video stream processing — real-time analysis of continuous CCTV footage
- Multi-camera zone aggregation — compliance across multiple camera positions
- Automated retraining pipeline — triggered when drift threshold is exceeded
- Kubernetes deployment — horizontal scaling for multi-site monitoring

---

## Author

Built by [Codextivity](https://github.com/codextivity) combining
Computer Vision expertise with LLM engineering and production MLOps.

**Related projects:**
- [LangChain Research Copilot](https://github.com/codextivity/langchain-copilot)
- [Multimodal Document Intelligence](https://github.com/codextivity/multimodal-doc-intelligence)
