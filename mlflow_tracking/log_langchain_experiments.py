# mlflow_tracking/log_langchain_experiments.py
# Logs LangChain Copilot evaluation results into MLflow.
#
# Why log LLM experiments in MLflow?
# The same principle as CV experiments — every configuration
# change should be tracked so you can explain why you made
# each decision and compare results objectively.
#
# Run from project root:
#   python mlflow_tracking/log_langchain_experiments.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow

DB_PATH = Path(__file__).parent.parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# ── Experiment: LangChain Research Copilot ────────────────────────────────────
mlflow.set_experiment("langchain-research-copilot")

# ── Run 1: Initial RAG configuration ─────────────────────────────────────────
# First working version — chunk_size=1000 caused table splitting issues

print("Logging RAG v1 baseline...")

with mlflow.start_run(run_name="rag-v1-baseline"):

    mlflow.log_params({
        # LLM configuration
        "llm":                  "gpt-4o-mini",
        "temperature":          0,
        "embedding_model":      "text-embedding-3-small",

        # Vector store
        "vector_store":         "ChromaDB",
        "search_type":          "similarity",
        "retrieval_k":          4,

        # Chunking — first attempt
        "chunk_size":           1000,
        "chunk_overlap":        200,
        "text_splitter":        "RecursiveCharacterTextSplitter",

        # Advanced features
        "history_aware":        True,
        "reranking":            False,
        "hybrid_search":        False,

        # Evaluation
        "eval_questions":       20,
        "evaluator":            "LangSmith LLM-as-judge",
        "dataset":              "Cambodia economics PDF",
    })

    mlflow.log_metrics({
        "correctness":          1.00,
        "faithfulness":         0.90,
        "eval_question_count":  20,
    })

    mlflow.set_tags({
        "known_issue": (
            "chunk_size=1000 splits table headers from data rows — "
            "LLM receives numbers without column labels"
        ),
        "deployment":           "Render",
        "version":              "v1",
    })

print("RAG v1 logged ✅")

# ── Run 2: Improved chunking ──────────────────────────────────────────────────
# Fixed table splitting by increasing chunk size and overlap

print("Logging RAG v2 improved chunking...")

with mlflow.start_run(run_name="rag-v2-improved-chunking"):

    mlflow.log_params({
        "llm":                  "gpt-4o-mini",
        "temperature":          0,
        "embedding_model":      "text-embedding-3-small",
        "vector_store":         "ChromaDB",
        "search_type":          "similarity",
        "retrieval_k":          4,

        # Chunking — improved
        "chunk_size":           1500,   # increased from 1000
        "chunk_overlap":        400,    # increased from 200
        "text_splitter":        "RecursiveCharacterTextSplitter",

        "history_aware":        True,
        "reranking":            False,
        "hybrid_search":        False,
        "eval_questions":       20,
        "evaluator":            "LangSmith LLM-as-judge",
        "dataset":              "Cambodia economics PDF",
    })

    mlflow.log_metrics({
        "correctness":          1.00,
        "faithfulness":         0.92,   # improved from 0.90
        "eval_question_count":  20,
    })

    mlflow.set_tags({
        "change_from_v1": (
            "chunk_size 1000→1500, chunk_overlap 200→400"
        ),
        "reason": (
            "larger chunks keep table headers with data rows — "
            "LLM receives complete context for numerical tables"
        ),
        "result":               "faithfulness improved 0.90 → 0.92",
        "deployment":           "Render",
        "version":              "v2",
        "production":           "true",
    })

print("RAG v2 logged ✅\n")

# ── Experiment: Multimodal Document Intelligence ──────────────────────────────
mlflow.set_experiment("multimodal-doc-intelligence")

print("Logging multimodal experiment...")

with mlflow.start_run(run_name="multimodal-rag-v1"):

    mlflow.log_params({
        "llm":                      "gpt-4o",
        "vision_model":             "gpt-4o",
        "embedding_model":          "text-embedding-3-small",
        "vector_store":             "ChromaDB",
        "retrieval_k":              6,
        "chunk_size":               1500,
        "chunk_overlap":            400,

        # Document processing
        "text_pdf_extractor":       "PyMuPDF",
        "visual_extractor":         "GPT-4o vision",
        "document_classifier":      "custom PyMuPDF page analyzer",

        # Agent
        "agent_type":               "LangGraph 5-node supervisor",
        "routing_nodes":            "text,vision,tools,compare,web",
        "tool_calling":             True,
        "web_search":               "Tavily",
    })

    mlflow.log_metrics({
        # Content type coverage
        "supported_content_types":  5,
        "routing_nodes":            5,

        # System performance
        "text_retrieval_accuracy":  0.90,
        "visual_retrieval_accuracy": 0.85,
    })

    mlflow.set_tags({
        "key_innovation": (
            "two-pass retrieval — image chunks prioritized "
            "before mixed chunks for vision queries"
        ),
        "deployment":           "Render",
        "production":           "true",
    })

print("Multimodal experiment logged ✅\n")
print("=" * 50)
print("All experiments logged")
print("Open http://localhost:5000")
print("Switch between experiments using the left sidebar")