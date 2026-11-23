
# Elephant-Goldfish Model (EGM) RAG Triad Evaluation Engine
## Implementation Specification & Architectural Agreement

---

## 1. Agreed Architectural Principles

This document serves as the binding specification for the new RAG and Evaluation system, formalizing all decisions agreed upon:

### A. The Elephant-Goldfish Model (EGM)
* **The Elephant (Persistent Memory & State):**
  * **PostgreSQL (port 5434)** running via Docker Compose with `pgvector` enabled.
  * Stores all document chunks, vector embeddings, historical queries, and full audit logs.
  * Never forgets; guarantees complete traceability for all chat interactions and evaluation scores.
  * **Prometheus (port 9090)** + **Grafana (port 3000)** retain long-term performance telemetry, hallucination trends, and metric drift.
* **The Goldfish (Stateless Bounded Evaluators):**
  * Every evaluation run operates with **zero memory bleed or conversational baggage**.
  * The judge receives only the exact query, the isolated retrieved chunks, and the generated response.
  * Ensures zero bias: the model cannot assume or "remember" facts not present in the provided chunks.

### B. Model Stack
* **Embeddings:** `text-embedding-3-small` (1536-dimensional vectors) via `OPENAI_API_KEY`.
* **Chatbot Generation:** Dynamic model selection via dropdown (`model_id` parameter):
  * `gpt-6-astra` (Flagship generation model)
  * `gpt-6.1-sol` (High-end coding & reasoning)
  * `gpt-5.5` (Standard fast reasoning)
  * `gpt-4o` (Omni multi-purpose)
  * `gpt-4o-mini` (Budget & high-speed)
  * Synthesizes responses strictly grounded in retrieved context chunks.
* **Evaluation Judge (The Goldfish):** `gpt-6-astra` (Flagship reasoning model) executing atomic claim decomposition, groundedness checks, and confidence scoring.

### C. Evaluation Framework (RAG Triad)
* **Faithfulness / Groundedness ($0.0 - 1.0$):** Response broken into atomic claims. Each claim verified against retrieved chunks. Any unsupported statement is flagged as an explicit hallucination.
* **Context Relevance ($0.0 - 1.0$):** Evaluates the retrieval signal-to-noise ratio and identifies extraneous chunks.
* **Answer Relevancy ($0.0 - 1.0$):** Measures directness and completeness in answering the user's intent.
* **Composite Confidence Score ($0\% - 100\%$):**
  $$\text{Confidence Score} = (0.50 \times \text{Faithfulness}) + (0.30 \times \text{Answer Relevancy}) + (0.20 \times \text{Context Relevance})$$
* **Verdicts:** `HIGH_CONFIDENCE` ($\ge 85\%$, Faithfulness $= 1.0$), `MEDIUM_CONFIDENCE` ($70-84\%$), `HALLUCINATION_DETECTED` (Faithfulness $< 0.80$), `LOW_QUALITY` ($< 70\%$).

### D. Execution Strategy: Asynchronous Background Worker (Non-Blocking)
* **Step 1:** User submits a question $\rightarrow$ Retrieve top-$k$ chunks from `pgvector`.
* **Step 2:** Stream/return answer to user immediately (zero perceived delay, normal responsive chatbot experience).
* **Step 3 (Simultaneous Background Task):** Using Python's `asyncio.create_task` (or Celery/background worker), dispatch the event `(query, chunks, response, model_used)` to the evaluator.
* **Step 4:** The Goldfish evaluator (`gpt-6-astra`) grades Faithfulness, Relevancy, and Confidence.
* **Step 5:** The worker writes the scores to PostgreSQL (`rag_evaluations`) and increments Prometheus counters.
* **Step 6:** If the confidence is low ($< 70\%$) or a hallucination is detected, an alert is triggered in Prometheus/Grafana, and a warning flag is updated in the chat session.

---

## 2. System Architecture Diagram

```
[ Documents (.pdf, .docx, tables, .txt) ]
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │                 RAGFlow Engine (Cloud/API)             │
  │  • DeepDoc Layout Analysis & OCR (extracts tables)     │
  │  • Domain-Aware Chunking Templates                     │
  │  • Multi-Model Embedding Generation                    │
  │  • Hybrid Search (Dense Vectors + Sparse BM25)         │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼ Top-K Reranked Context Chunks
User Query ─────────> [ RAG Retrieval API ]
    │                        │
    ├────────────────────────┤
    ▼                        ▼
[ Query Logged ]   [ Chatbot Generator (Dropdown: gpt-6-astra, gpt-4o...) ]
                             │
                     Response Streamed to User
                             │
    ┌────────────────────────┘
    │ (Query + RAGFlow Chunks + Response)
    ▼
[ THE GOLDFISH: gpt-6-astra Evaluator ]
    │
    ├──────────────────────────────────────────────────────────────┐
    ▼                                                              ▼
[ THE ELEPHANT: PostgreSQL ]                             [ Prometheus Exporter (:8000) ]
  Table: rag_evaluations                                   • rag_confidence_score (Histogram)
  • query & response                                       • rag_faithfulness_score (Histogram)
  • retrieved_contexts (JSONB)                             • rag_hallucinations_total (Counter)
  • unsupported_claims (JSONB)                             • Scraped by Prometheus (:9090)
  • confidence_score & verdict                                     │
    │                                                              │
    └──────────────────────────────┬───────────────────────────────┘
                                   ▼
                      [ Grafana Viewport (:3000) ]
                      • Stat Gauge: Overall Confidence %
                      • Timeseries: Groundedness vs Retrieval Noise
                      • SQL Table: Real-Time Hallucination Audit
```

---

## 3. Database Schema Specification (PostgreSQL / `pgvector`)

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Document Embeddings (Elephant Knowledge Base)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_embedding_hnsw 
ON document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 2. RAG Evaluation Audit Log (Elephant Audit Trail)
CREATE TABLE IF NOT EXISTS rag_evaluations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    query TEXT NOT NULL,
    retrieved_contexts JSONB NOT NULL,
    response TEXT NOT NULL,
    faithfulness_score FLOAT NOT NULL,
    context_relevance_score FLOAT NOT NULL,
    answer_relevance_score FLOAT NOT NULL,
    confidence_score FLOAT NOT NULL,
    verdict VARCHAR(50) NOT NULL,
    unsupported_claims JSONB DEFAULT '[]'::jsonb,
    explanations JSONB DEFAULT '{}'::jsonb,
    latency_ms INT
);

CREATE INDEX IF NOT EXISTS idx_rag_evals_verdict ON rag_evaluations(verdict);
CREATE INDEX IF NOT EXISTS idx_rag_evals_created_at ON rag_evaluations(created_at DESC);
```

---

## 4. MCP Tools Specification

The server exposes the following tools via FastMCP:

| Tool | Parameters | Output | Description |
| :--- | :--- | :--- | :--- |
| `ingest_document` | `file_path`, `document_name` | `{ status: "ok", chunks_count: int }` | Reads document, splits into chunks, computes embeddings, and stores in `document_embeddings`. |
| `ask_rag` | `query`, `top_k=4`, `async_eval=true` | `{ answer: string, sources: list, confidence_score?: float }` | Retrieves chunks via cosine search, generates answer with `gpt-6-astra`, triggers evaluation. |
| `evaluate_rag_triad` | `query`, `contexts`, `response` | `{ confidence_score, metrics, verdict, unsupported_claims, explanations }` | Pure Goldfish evaluator assessing Faithfulness, Relevance, and Confidence. |

---

## 5. Execution & Rollout Phases

### Phase 1: Storage & Infrastructure Foundation
* Connect to PostgreSQL on port `5434` (container `mathrouter-postgres`).
* Run the migration script to enable `pgvector` and create `document_embeddings` and `rag_evaluations`.

### Phase 2: Ingestion & Vector Retrieval Pipeline
* Implement `ingestion.py` using `text-embedding-3-small`.
* Build vector similarity search with Cosine Distance (`<=>`) and HNSW index.

### Phase 3: `gpt-6-astra` Generation & RAG Triad Evaluator
* Implement `generator.py` using `gpt-6-astra`.
* Implement `evaluator.py` using `gpt-6-astra` with structured Pydantic schemas (Claim decomposition, verification, confidence computation).

### Phase 4: Telemetry & Prometheus Exporter
* Implement Prometheus metrics server exposing port `8000/metrics`.
* Verify scrape continuity with `mathrouter-prometheus` (`:9090`).

### Phase 5: FastMCP Tool Server & Grafana Dashboards
* Wire the components into FastMCP (`server.py`).
* Provide the Grafana dashboard JSON panel configurations for instant visualization.
