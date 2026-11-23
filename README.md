# RAG Triad & Evaluation Console V2

> **Enterprise Retrieval-Augmented Generation (RAG) Evaluation, Multi-Model Benchmarking, and Observability Platform powered by the Elephant-Goldfish Model (EGM) and Universal UI V2.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20with%20pgvector-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![React](https://img.shields.io/badge/React-19.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![Prometheus](https://img.shields.io/badge/Prometheus-v2.54-E6522C.svg?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-v11.1-F46800.svg?logo=grafana&logoColor=white)](https://grafana.com)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-v2-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)

---

## 📚 Comprehensive Documentation Suite

Complete technical and operational documentation is available in the dedicated [`docs/`](docs/) directory:

| Document | Description |
| :--- | :--- |
| [**📖 Documentation Index**](docs/README.md) | Central hub, core pillars, and high-level architectural map. |
| [**📘 Web Console User Guide**](docs/USER_GUIDE.md) | Universal UI V2 walkthrough, three-panel navigation, RAG Studio, Prompt Evaluator, and PDF/XLSX exports. |
| [**📗 RAG Triad & Hallucination Guide**](docs/RAG_EVALUATION_GUIDE.md) | The Elephant-Goldfish Model (EGM), `pgvector` ingestion, Faithfulness/Groundedness atomic claims, and confidence scoring. |
| [**📙 Prompt Evaluation & Multi-Model Guide**](docs/PROMPT_EVALUATION_GUIDE.md) | Prompt variant testing (A/B/C), cross-model benchmarking (`GPT-4o`, `GPT-5.5`, `GPT-6 Astra`, `Gemini`), token accounting, and LLM-as-a-Judge rubrics. |
| [**📐 Architecture & Design Specification**](docs/ARCHITECTURE.md) | Detailed technical architecture, HNSW cosine vector index, stateless Goldfish auditors, and sequence diagrams. |
| [**📡 REST API & Database Reference**](docs/API_AND_DATABASE_REFERENCE.md) | Full FastAPI endpoint documentation, JSON request/response payloads, and PostgreSQL relational DDL. |
| [**🚀 Deployment & Operations Runbook**](docs/DEPLOYMENT_AND_OPERATIONS.md) | Docker Compose orchestration, Prometheus metrics dictionary, Grafana dashboard setup, and troubleshooting runbooks. |

---

## 🚀 Executive Overview

**`rag-evals`** is a production-grade evaluation and observability platform designed to eliminate hallucinations, audit context relevance, and benchmark LLMs across complex question-answering workflows.

### Core Architectural Pillars
1. **The Elephant-Goldfish Model (EGM):**
   * **The Elephant (State Retention):** PostgreSQL 16 with `pgvector` stores all document chunks, 1536-dimensional embeddings, historical sessions, and full audit logs. Prometheus and Grafana preserve long-term metric distributions.
   * **The Goldfish (Stateless Auditing):** Evaluator LLMs operate with zero conversational memory bleed. Claims are strictly audited against retrieved context chunks without external world-knowledge assumptions.
2. **The RAG Triad Metric Engine:**
   * **Faithfulness / Groundedness ($0.0 - 1.0$):** Response decomposed into atomic factual assertions and verified against retrieved chunks.
   * **Context Relevance ($0.0 - 1.0$):** Evaluates signal-to-noise ratio in retrieved context chunks.
   * **Answer Relevancy ($0.0 - 1.0$):** Measures how completely and concisely the answer addresses user intent.
   * **Composite Confidence Score:** $(0.50 \times \text{Faithfulness}) + (0.30 \times \text{Answer Relevancy}) + (0.20 \times \text{Context Relevance})$.
3. **Universal UI V2 Evaluation Console:**
   * Built with **React 19** and **Vite 8**.
   * Implements strict three-panel geometry (56px Navigation Rail, 360px Session Explorer, 480px+ Detail Inspector).
   * Enforces a 900px minimum desktop viewport floor via `SupportedViewportGate`.
   * Real-time PDF, XLSX, and JSON export capabilities.

---

## 📐 Architecture & System Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Eval Runner
    participant Web as Evaluation Console (React 19)
    participant API as FastAPI Backend (src/api.py)
    participant PG as PostgreSQL 16 (pgvector: 5434)
    participant Gen as Generation LLM (GPT-6 / GPT-4o)
    participant Judge as Goldfish Triad Evaluator
    participant Prom as Prometheus (9090) / Grafana (3000)

    User->>Web: Submit Question / Select Dataset
    Web->>API: POST /api/query { query, model_id, top_k }
    
    rect rgb(30, 41, 59)
    note over API,PG: Cosine Vector Retrieval (HNSW)
    API->>API: Compute 1536-dim Embedding
    API->>PG: Cosine Search (embedding <=> query_vector)
    PG-->>API: Return Top-k Chunks & Document Metadata
    end

    rect rgb(15, 23, 42)
    note over API,Gen: Strictly Grounded Synthesis
    API->>Gen: Query + Retrieved Chunks + Strict Grounding Rules
    Gen-->>API: Return Grounded Answer & Latency
    end

    API-->>Web: Stream Answer & Chunks to Console
    Web-->>User: Visual Render in Panel 3 Inspector

    par Async Non-Blocking Evaluation
        API->>Judge: evaluate_rag_triad(query, chunks, answer)
        Judge->>Judge: Atomic Claim Decomposition & Verification
        Judge-->>API: Triad Output (Faithfulness, Relevance, Verdict)
        API->>PG: INSERT INTO rag_evaluations
        API->>Prom: Update Metric Counters & Histograms
        API-->>Web: Push Audit Breakdown
        Web-->>User: Display RAG Triad Score Cards & Hallucination Flags
    end
```

---

## 🛠️ Quickstart

### Prerequisites
* **Docker & Docker Compose** (v2.20+)
* **Python 3.10+**
* **Node.js 18+**
* **OpenAI API Key** (and optionally Google Gemini API Key)

### 1. Environment Configuration
Create a `.env` file in the repository root:
```bash
cp hello-mcp/.env .env
# Ensure OPENAI_API_KEY and DATABASE_URL are set:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5434/mathrouter
```

### 2. Launch Docker Services
Start PostgreSQL 16 (with `pgvector`), Prometheus, and Grafana:
```bash
cd hello-mcp
docker compose up -d postgres prometheus grafana
cd ..
```

* **PostgreSQL:** `localhost:5434`
* **Prometheus:** `http://localhost:9090`
* **Grafana:** `http://localhost:3000` (User: `admin` / Password: `admin`)

### 3. Start Backend API
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch Evaluation Console (Frontend)
```bash
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📂 Repository Structure

```text
rag-evals/
├── docs/                           # Comprehensive documentation suite
│   ├── README.md                   # Documentation index & architecture map
│   ├── USER_GUIDE.md               # Evaluation Console V2 user guide
│   ├── RAG_EVALUATION_GUIDE.md     # EGM architecture & RAG Triad guide
│   ├── PROMPT_EVALUATION_GUIDE.md  # Prompt testing & multi-model guide
│   ├── ARCHITECTURE.md             # Technical design & sequence flows
│   ├── API_AND_DATABASE_REFERENCE.md # REST API & SQL DDL reference
│   └── DEPLOYMENT_AND_OPERATIONS.md# Docker, Prometheus & operations runbook
├── src/
│   ├── api.py                      # FastAPI orchestrator & RAG endpoints
│   ├── server.py                   # RAG retrieval & ingestion server
│   ├── generator.py                # LLM answer synthesis & grounding
│   ├── evaluator.py                # Goldfish stateless RAG Triad judge
│   ├── db.py                       # PostgreSQL & pgvector connection layer
│   ├── metrics.py                  # Prometheus metrics instrumentation
│   ├── App.jsx                     # Universal UI V2 root application
│   └── components/
│       ├── RagEvaluationView.jsx   # RAG Triad interactive studio
│       ├── PromptEvaluationView.jsx# Prompt variant & multi-prompt studio
│       ├── AllExperimentsDataHub.jsx # Central experiments table & bulk export
│       ├── MultiModelEvaluationModal.jsx # Cross-model side-by-side comparator
│       └── SupportedViewportGate.jsx # Enforced 900px minimum desktop gate
├── schema.sql                      # PostgreSQL DDL with vector & HNSW indexes
├── hello-mcp/                      # Infrastructure & micro-servers
│   ├── docker-compose.yml          # Postgres, Prometheus, Grafana, App
│   ├── prometheus.yml              # Prometheus scrape configuration
│   └── grafana/                    # Grafana pre-provisioned datasources
├── run_100_questions_benchmark.py  # 100-question automated benchmark runner
├── run_philosophical_100_benchmark.py # Complex reasoning benchmark runner
└── package.json                    # React 19, Lucide, Vite dependencies
```

---

## 📄 License & Author

* **Author:** Sujan
* **Repository:** [sujandeeep/rag-evals](https://github.com/sujandeeep/rag-evals)
* **License:** MIT License
