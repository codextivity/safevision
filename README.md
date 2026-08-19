# 🦺 SafeVision — PPE Compliance Intelligence System

AI-powered construction site safety monitoring that combines
YOLOv11 object detection with GPT-4o vision verification and
a LangChain agent for natural language safety queries.

**Live Demo:** https://safevision.onrender.com/docs
**GitHub:** https://github.com/codextivity/safevision

---

## What It Does

Upload a construction site image and get instant PPE compliance analysis:

```
POST /detect → image upload → per-worker compliance report
POST /query  → natural language → safety insights from database
GET  /violations/summary → aggregate compliance statistics
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
┌───────────────────┐
│   YOLOv11m        │  Detects: Hardhat, NO-Hardhat,
│   PPE Detector    │           Safety Vest, NO-Safety Vest,
│   0.742 mAP50     │           Person
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Spatial           │  Associates each Person detection
│ Association       │  with nearby PPE detections
│ Algorithm         │
└────────┬──────────┘
         │
         ├── High confidence ──► Store directly in SQLite
         │
         └── Uncertain ─────────► GPT-4o Vision Verification
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │   SQLite    │
                                  │  Database   │
                                  └──────┬──────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │  LangChain  │
                                  │   Agent     │
                                  │  6 tools    │
                                  └─────────────┘
                                         │
                                         ▼
                                  Natural Language
                                  Safety Reports
```

---

## Key Engineering Decisions

**Why YOLO + GPT-4o instead of GPT-4o alone?**
YOLO processes images in 4ms at no API cost.
GPT-4o takes 2-3 seconds and costs per call.
The tiered system uses YOLO for all frames and GPT-4o
only for uncertain detections (typically 5-15% of cases).
This reduces cost by 85-95% while maintaining accuracy.

**Why spatial association instead of direct violation classification?**
Training a binary compliant/non-compliant classifier requires
worker-level labels that most datasets lack. Spatial association
pairs person detections with nearby PPE detections — making
the compliance logic configurable in code rather than baked
into the model. PPE requirements can be changed per site
without retraining.

**Why LangChain agent over a fixed dashboard?**
Fixed dashboards answer predefined questions.
The LangChain agent answers any question grounded in real
detection data. Safety managers can ask domain-specific
questions without waiting for a developer to add a new chart.

---

## Model Performance

Trained on 5646 construction site images — YOLOv11m, 50 epochs:

| Class | mAP50 | Assessment |
|---|---|---|
| Safety Vest | 0.893 | ✅ Production ready |
| Hardhat | 0.702 | ✅ Strong |
| Person | 0.693 | ✅ Strong |
| NO-Safety Vest | 0.335 | ⚠ GPT-4o verification |
| NO-Hardhat | 0.121 | ⚠ GPT-4o verification |
| **Overall** | **0.549** | **Demo ready** |

Violation classes score lower due to class imbalance
(9-14% of training instances). Low-confidence detections
are automatically routed to GPT-4o vision for verification.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Object detection | YOLOv11m | PPE and person detection |
| Vision verification | GPT-4o | Uncertain detection verification |
| LLM framework | LangChain + LangGraph | Agent with 6 database tools |
| Database | SQLite | Violation history storage |
| API | FastAPI | HTTP endpoints |
| Observability | LangSmith | Full chain tracing |
| Deployment | Render + Docker | Cloud hosting |

---

## Quick Start

```bash
git clone https://github.com/codextivity/safevision
cd safevision
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service status |
| POST | /detect | Upload image for PPE analysis |
| POST | /query | Natural language safety query |
| GET | /violations | List violation records |
| GET | /violations/summary | Compliance statistics |

---

## Project Structure

```
safevision/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings via pydantic-settings
│   ├── api/routes/
│   │   ├── health.py        # GET /health
│   │   ├── detect.py        # POST /detect
│   │   ├── query.py         # POST /query
│   │   └── violations.py    # GET /violations
│   └── core/
│       ├── detector.py      # YOLO inference + spatial association
│       ├── verifier.py      # GPT-4o vision verification
│       ├── bridge.py        # CV outputs → LLM-readable text
│       ├── agent.py         # LangChain agent with 6 tools
│       └── database.py      # SQLite operations
├── models/trained/best.pt   # Trained YOLOv11m weights
├── train.py                 # Training script
├── evaluate.py              # Test set evaluation
├── Dockerfile
└── docker-compose.yml
```

---

## Interview Talking Points

**On the CV + LLM integration:**
"The key insight was that CV models produce excellent detection
data but poor user interfaces. Raw bounding boxes and confidence
scores are meaningless to safety managers. The LangChain agent
layer translates detection outputs into actionable insights —
trend analysis, violation summaries, and recommendations —
grounded in real database records."

**On the tiered verification approach:**
"I analyzed the model's per-class performance and found that
explicit violation class detection was weak (0.121 mAP50 on
NO-Hardhat) due to class imbalance in the training data.
Rather than accepting this limitation, I designed a tiered system
where YOLO handles confident detections and GPT-4o verifies
uncertain cases. This compensates for the model's weakness
without requiring more training data."

**On spatial association:**
"Instead of training a binary compliant/non-compliant classifier,
I implemented a distance-based spatial association algorithm that
pairs each person detection with nearby PPE detections. This means
compliance rules are configurable in code rather than baked into
the model — the same trained model works for any PPE policy."

---

## Author

Built by [Codextivity](https://github.com/codextivity)
combining Computer Vision and LLM engineering expertise.