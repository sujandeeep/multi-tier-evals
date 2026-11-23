# Deployment, Infrastructure & Operations Runbook

> **Target Audience:** DevOps Engineers, Site Reliability Engineers (SREs), Infrastructure Teams, and System Developers.

---

## 📌 Table of Contents
1. [Infrastructure Topology & Architecture](#1-infrastructure-topology--architecture)
2. [Prerequisites & System Requirements](#2-prerequisites--system-requirements)
3. [Environment Configuration (`.env`)](#3-environment-configuration-env)
4. [Docker Compose Orchestration](#4-docker-compose-orchestration)
5. [Local Development Setup](#5-local-development-setup)
6. [Prometheus Metrics & Scraping Configuration](#6-prometheus-metrics--scraping-configuration)
7. [Grafana Dashboards & Alerting](#7-grafana-dashboards--alerting)
8. [Automated Benchmark Execution](#8-automated-benchmark-execution)
9. [Operational Troubleshooting & Runbooks](#9-operational-troubleshooting--runbooks)

---

## 1. Infrastructure Topology & Architecture

The entire platform is orchestrated via Docker Compose:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER NETWORK (hello-mcp)                         │
│                                                                             │
│  ┌──────────────────────┐   ┌──────────────────────┐   ┌─────────────────┐  │
│  │ PostgreSQL 16        │   │ Prometheus v2.54     │   │ Grafana 11.1    │  │
│  │ (pgvector)           │   │ (Metrics Scraper)    │   │ (Dashboards)    │  │
│  │ Host Port: 5434      │   │ Host Port: 9090      │   │ Host Port: 3000 │  │
│  └──────────┬───────────┘   └──────────▲───────────┘   └────────▲────────┘  │
│             │                          │                        │           │
│             │ Internal Network         │ Scrapes /metrics       │ Datasource│
│             ▼                          │                        │           │
│  ┌─────────────────────────────────────┴────────────────────────┴────────┐  │
│  │ FastAPI Backend Service (mathrouter-backend)                          │  │
│  │ Host Port: 8000 | Container Port: 8000                                │  │
│  └─────────────────────────────────────▲─────────────────────────────────┘  │
│                                        │ Proxies API calls                  │
│  ┌─────────────────────────────────────┴─────────────────────────────────┐  │
│  │ React 19 Frontend Console (mathrouter-frontend)                       │  │
│  │ Host Port: 3001 | Container Port: 3001                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & System Requirements

### Hardware Recommendations
* **CPU:** 4 Cores or higher.
* **RAM:** 8 GB minimum (16 GB recommended for concurrent batch evaluations).
* **Disk:** 20 GB free SSD storage (for Docker images, PostgreSQL vector data, and Prometheus WAL).

### Software Requirements
* **Docker Engine:** v24.0.0 or higher.
* **Docker Compose:** v2.20.0 or higher.
* **Node.js:** v18.0.0+ and `npm` v9.0.0+.
* **Python:** v3.10 or v3.11.

---

## 3. Environment Configuration (`.env`)

Create a root `.env` file (and copy it to `hello-mcp/.env`):

```bash
# ==============================================================================
# Model API Credentials
# ==============================================================================
OPENAI_API_KEY=sk-proj-your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here

# ==============================================================================
# Database Configuration (PostgreSQL 16 + pgvector)
# ==============================================================================
# When running backend locally:
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/mathrouter

# When running backend inside Docker:
# DATABASE_URL=postgresql://postgres:postgres@postgres:5432/mathrouter

# ==============================================================================
# API & Network Security
# ==============================================================================
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:5173,http://127.0.0.1:5173

# Frontend Environment Variables (for Vite build)
VITE_BACKEND_URL=http://localhost:8000
VITE_PROMETHEUS_URL=http://localhost:9090
```

---

## 4. Docker Compose Orchestration

The platform infrastructure is defined in `hello-mcp/docker-compose.yml`.

### Launching the Full Container Stack
```bash
cd hello-mcp
docker compose up -d --build
```

### Verifying Service Health
```bash
docker compose ps
```
You should see all 5 containers active:
* `mathrouter-postgres`: Healthy on `0.0.0.0:5434->5432/tcp`
* `mathrouter-prometheus`: Up on `0.0.0.0:9090->9090/tcp`
* `mathrouter-grafana`: Up on `0.0.0.0:3000->3000/tcp`
* `mathrouter-backend`: Up on `0.0.0.0:8000->8000/tcp`
* `mathrouter-frontend`: Up on `0.0.0.0:3001->3001/tcp`

### Viewing Container Logs
```bash
# Follow logs for backend
docker compose logs -f backend

# Follow logs for postgres
docker compose logs -f postgres
```

### Stopping Services
```bash
docker compose down
```

---

## 5. Local Development Setup

For fast day-to-day development without rebuilding Docker containers:

### Step 1: Start Supporting Infrastructure
Run only PostgreSQL, Prometheus, and Grafana in Docker:
```bash
cd hello-mcp
docker compose up -d postgres prometheus grafana
```

### Step 2: Run FastAPI Backend
```bash
# In the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run backend with auto-reload
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Run React Frontend
```bash
# In another terminal in repository root
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 6. Prometheus Metrics & Scraping Configuration

Prometheus scrapes the FastAPI `/metrics` endpoint every 5 seconds, as configured in `hello-mcp/prometheus.yml`:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  - job_name: "mathrouter"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["host.docker.internal:8000", "backend:8000"]
```

### Key Metrics Dictionary
| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `rag_eval_queries_total` | Counter | Total RAG queries evaluated, labeled by `model_id` and `verdict`. |
| `rag_eval_confidence_score` | Histogram | Distribution of composite confidence scores ($0.0 - 1.0$). |
| `rag_eval_faithfulness_ratio` | Gauge | Groundedness ratio over time. |
| `rag_eval_latency_seconds` | Histogram | End-to-end evaluation latency. |
| `mathrouter_tokens_total` | Counter | Total token spend, labeled by `model` and `type` (`prompt` vs `completion`). |

---

## 7. Grafana Dashboards & Alerting

Grafana is pre-configured with automatic provisioning:
* **URL:** `http://localhost:3000`
* **Default Credentials:** User: `admin` / Password: `admin`
* **Pre-provisioned Datasource:** Connected to Prometheus at `http://prometheus:9090`.

### Pre-Configured Alert Thresholds
1. **Hallucination Alert:** Fires when `rag_eval_faithfulness_ratio < 0.80` over a 5-minute rolling window.
2. **Latency Degradation Alert:** Fires when evaluation p95 latency exceeds 5 seconds.
3. **Low Confidence Spike:** Fires when the rate of `verdict="LOW_QUALITY"` exceeds 15% of total queries.

---

## 8. Automated Benchmark Execution

To validate model accuracy or run regression benchmarks from the CLI:

```bash
source .venv/bin/activate

# Run 100-question comprehensive benchmark
python3 run_100_questions_benchmark.py

# Run philosophical complex reasoning benchmark
python3 run_philosophical_100_benchmark.py

# Re-run baseline comparisons
python3 hello-mcp/rerun_gpt5.py
python3 hello-mcp/rerun_gemini.py
```

Generated benchmark reports will be output to `rag_benchmark_comparison_100.pdf` and `benchmark_100_results.json`.

---

## 9. Operational Troubleshooting & Runbooks

### Issue 1: `pgvector` Extension Error or Vector Dimension Mismatch
* **Symptom:** `ERROR: type "vector" does not exist` or `dimensions do not match (1536 expected)`.
* **Remedy:** Ensure the PostgreSQL image is `pgvector/pgvector:pg16`. Connect to PostgreSQL and run:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

### Issue 2: CORS Header Blocking in Frontend
* **Symptom:** Browser console reports: `Access to fetch at ... has been blocked by CORS policy`.
* **Remedy:** Verify your origin is included in `ALLOWED_ORIGINS` in `.env`. The backend allows `http://localhost:5173`, `http://localhost:3000`, and `http://localhost:3001` by default.

### Issue 3: OpenAI API Rate Limits / Timeouts
* **Symptom:** `429 Too Many Requests` or early access errors for `gpt-6-astra`.
* **Remedy:** The platform includes automatic fallback logic in `src/generator.py` and `src/evaluator.py`, downgrading to `gpt-4o` if frontier models are rate-limited or unavailable.

### Issue 4: Disk Exhaustion from System Logs
* **Symptom:** High disk usage in `system.log`.
* **Remedy:** The system employs a `RotatingFileHandler` configured with a 20MB ceiling and 5 backup rotations. To manually purge:
  ```bash
  > hello-mcp/system.log
  ```
