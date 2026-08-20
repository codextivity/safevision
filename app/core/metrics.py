# app/core/metrics.py
# Defines custom Prometheus metrics for SafeVision.
#
# Prometheus has 4 metric types:
#
# Counter:   monotonically increasing number
#            Example: total requests received, total violations detected
#            Use when: counting events that only go up
#
# Gauge:     current value that can go up or down
#            Example: current memory usage, active workers
#            Use when: measuring current state
#
# Histogram: distribution of values across buckets
#            Example: request latency — how many requests took 0-10ms?
#            Use when: measuring distributions (latency, size)
#
# Summary:   similar to histogram but calculates quantiles client-side
#            Less common — use histogram instead for most cases

from prometheus_client import Counter, Gauge, Histogram, Info

# ── Detection metrics ─────────────────────────────────────────────────────────

DETECTION_REQUESTS_TOTAL = Counter(
    name="safevision_detection_requests_total",
    documentation="Total number of PPE detection requests",
    labelnames=["status"]
    # Labels let you slice metrics by dimension
    # status="success" or status="error"
    # In Grafana: sum by status, compare success vs error rate
)

DETECTION_LATENCY_SECONDS = Histogram(
    name="safevision_detection_latency_seconds",
    documentation="Time spent processing a detection request end to end",
    # Buckets define the histogram boundaries
    # These buckets are chosen for API latency in seconds:
    # 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

YOLO_INFERENCE_SECONDS = Histogram(
    name="safevision_yolo_inference_seconds",
    documentation="Time spent on YOLO model inference only (excludes pre/post processing)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

WORKERS_DETECTED_TOTAL = Counter(
    name="safevision_workers_detected_total",
    documentation="Total number of workers detected across all frames"
)

VIOLATIONS_DETECTED_TOTAL = Counter(
    name="safevision_violations_detected_total",
    documentation="Total number of PPE violations detected",
    labelnames=["violation_type"]
    # violation_type="NO-Hardhat" or "NO-Safety Vest"
    # In Grafana: compare which violation type is more common
)

VLM_VERIFICATIONS_TOTAL = Counter(
    name="safevision_vlm_verifications_total",
    documentation="Total number of uncertain detections sent to GPT-4o for verification",
    labelnames=["result"]
    # result="confirmed_violation" or "cleared"
)

COMPLIANCE_RATE_GAUGE = Gauge(
    name="safevision_compliance_rate_current",
    documentation="Current compliance rate from most recently analyzed frame (0.0 to 1.0)"
)

# ── Query/agent metrics ───────────────────────────────────────────────────────

QUERY_REQUESTS_TOTAL = Counter(
    name="safevision_query_requests_total",
    documentation="Total number of natural language query requests",
    labelnames=["status"]
)

QUERY_LATENCY_SECONDS = Histogram(
    name="safevision_query_latency_seconds",
    documentation="Time spent processing a natural language query including LLM call",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

LLM_TOKENS_TOTAL = Counter(
    name="safevision_llm_tokens_total",
    documentation="Total LLM tokens used for query answering",
    labelnames=["token_type"]
    # token_type="input" or "output"
)

# ── System metrics ────────────────────────────────────────────────────────────

MODEL_LOADED_GAUGE = Gauge(
    name="safevision_model_loaded",
    documentation="Whether the YOLO model is currently loaded (1=yes, 0=no)"
)

MEMORY_USAGE_BYTES = Gauge(
    name="safevision_memory_usage_bytes",
    documentation="Current process memory usage in bytes"
)

# ── Build info ────────────────────────────────────────────────────────────────

BUILD_INFO = Info(
    name="safevision_build",
    documentation="SafeVision build information"
)

# Set build info once at startup
BUILD_INFO.info({
    "version":      "1.0.0",
    "model":        "yolo11n",
    "framework":    "ultralytics",
    "deployment":   "local",
})