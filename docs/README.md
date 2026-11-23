# Evaluation & Observability Platform — Documentation Portal

Welcome to the comprehensive documentation suite for the **AI Evaluation, RAG Triad, & Observability Platform**.

This platform provides an end-to-end framework for evaluating, benchmarking, and observing Large Language Model (LLM) applications. It features the **Elephant-Goldfish Model (EGM)** for strictly grounded Retrieval-Augmented Generation (RAG) evaluation, a **Prompt & Multi-Model Evaluation Studio**, a **React 19 Universal UI V2 Console**, and an enterprise telemetry stack backed by **PostgreSQL (pgvector)**, **Prometheus**, and **Grafana**.

---

## 📚 Documentation Index

The documentation is organized into modular guides catering to evaluators, prompt engineers, system architects, developers, and DevOps operators:

| Document | Primary Audience | Key Coverage |
| :--- | :--- | :--- |
| [**1. Evaluation Console User Guide**](USER_GUIDE.md) | Evaluators, PMs, End-Users | Universal UI V2 three-panel layout, RAG Triad studio, Prompt Evaluation studio, All Experiments Data Hub, and PDF/XLSX/JSON exports. |
| [**2. RAG Triad & Hallucination Analysis Guide**](RAG_EVALUATION_GUIDE.md) | AI Engineers, Evaluators | Elephant-Goldfish Model (EGM), document ingestion (`pgvector`), Faithfulness/Hallucination scoring, Context Relevance, Answer Relevancy, and confidence verdicts. |
| [**3. Prompt Evaluation & Multi-Model Guide**](PROMPT_EVALUATION_GUIDE.md) | Prompt Engineers, Researchers | Single/Multi-prompt variant testing, comparative benchmarking across GPT-4o, GPT-5.5, GPT-6, and Gemini, latency/token spend analysis, and automated judging. |
| [**4. System Architecture & Design Spec**](ARCHITECTURE.md) | Architects, Tech Leads | System sequence diagrams, EGM paradigm, HNSW vector indexing, stateless Goldfish LLM judges, UI geometry specifications, and optional MCP agent tools. |
| [**5. REST API & Database Schema Reference**](API_AND_DATABASE_REFERENCE.md) | Backend & Full-Stack Developers | Full FastAPI endpoint catalog, request/response JSON schemas, curl examples, PostgreSQL relational schema, and `pgvector` index design. |
| [**6. Deployment & Operations Runbook**](DEPLOYMENT_AND_OPERATIONS.md) | DevOps, SREs, Developers | Docker Compose stack setup (Postgres, Prometheus, Grafana, Backend, Frontend), environment configuration, Prometheus metrics dictionary, and benchmark execution. |

---

## 🌟 Core System Pillars

```
                     ┌─────────────────────────────────────────────────────────┐
                     │          Universal UI V2 Evaluation Console             │
                     │   (React 19 + Vite 8 + Multi-Panel Geometry + 900px Gate)│
                     └─────────────┬─────────────────────────────┬─────────────┘
                                   │                             │
                                   ▼                             ▼
         ┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
         │     RAG Triad & EGM Evaluation        │ │     Prompt & Multi-Model Studio       │
         │ • Document Ingestion & Chunking       │ │ • Prompt Variant A/B/C Testing        │
         │ • pgvector Cosine Search (HNSW)       │ │ • Multi-Model Side-by-Side (GPT/Gemini)│
         │ • Goldfish Stateless Triad Evaluator  │ │ • Latency & Token Spend Accounting    │
         │ • Hallucination Breakdown & Scoring   │ │ • Automated LLM-as-a-Judge Grading    │
         └───────────────────┬───────────────────┘ └───────────────────┬───────────────────┘
                             │                                         │
                             ▼                                         ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │                 FastAPI Orchestrator                    │
                     │          (REST API + Background Task Workers)           │
                     └─────────────┬─────────────────────────────┬─────────────┘
                                   │                             │
                                   ▼                             ▼
         ┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
         │       The Elephant (Storage)          │ │       Telemetry & Observability       │
         │ • PostgreSQL 16 + pgvector (port 5434)│ │ • Prometheus Metrics Scraper (9090)   │
         │ • Document Chunks & 1536-dim Vectors  │ │ • Grafana Real-Time Dashboards (3000) │
         │ • Evaluation Audits & Session History │ │ • System Event & Audit Logs           │
         └───────────────────────────────────────┘ └───────────────────────────────────────┘
```

### 1. The Elephant-Goldfish Model (EGM)
* **The Elephant (Persistent Memory):** PostgreSQL with `pgvector` stores document chunks, embeddings, historical queries, and full audit logs. It never forgets and guarantees complete provenance.
* **The Goldfish (Stateless Evaluation):** Evaluator models operate with bounded zero memory bleed. The judge inspects solely the exact query, the retrieved context chunks, and the generated response, preventing prior knowledge contamination.

### 2. The RAG Triad Scoring Framework
Every RAG response is audited across three orthogonal dimensions:
1. **Faithfulness / Groundedness ($0.0 - 1.0$):** Response decomposed into atomic factual claims. Every claim is checked against retrieved chunks. Unsupported statements are flagged as hallucinations.
2. **Context Relevance ($0.0 - 1.0$):** Measures the signal-to-noise ratio in retrieved context chunks.
3. **Answer Relevancy ($0.0 - 1.0$):** Evaluates how directly and completely the answer addresses the user query.
4. **Composite Confidence Score:**
   $$\text{Confidence Score} = (0.50 \times \text{Faithfulness}) + (0.30 \times \text{Answer Relevancy}) + (0.20 \times \text{Context Relevance})$$

### 3. Universal UI V2 Console
A three-panel enterprise interface adhering to the Universal UI Guide V2:
* **Panel 1 (Navigation Rail):** 56px collapsed rail expandable up to 320px.
* **Panel 2 (Session & History Explorer):** Retained session explorer with search, status filters, and instant selection.
* **Panel 3 (Task & Detail Inspector):** Detailed inspection with turn traces, raw JSON inspect, side-by-side judging breakdown, and PDF/XLSX export.
* **Supported Viewport Gate:** Strict 900px minimum viewport floor preventing broken horizontal layouts.

---

## ⚡ Quickstart Overview

To spin up the entire platform locally:

```bash
# 1. Clone the repository and configure environment variables
cp hello-mcp/.env.example .env

# 2. Launch the Docker Observability Stack (PostgreSQL 16, Prometheus, Grafana)
cd hello-mcp
docker compose up -d

# 3. Start the FastAPI Backend
source .venv/bin/activate
python3 src/api.py

# 4. In a separate terminal, start the React Evaluation Console
npm install
npm run dev
```

Visit `http://localhost:5173` to open the console. For detailed installation and deployment steps, see the [Deployment Runbook](DEPLOYMENT_AND_OPERATIONS.md).
