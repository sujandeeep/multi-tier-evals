# REST API & Database Schema Reference

> **Backend:** FastAPI (Python 3.10+)  
> **Database:** PostgreSQL 16 with `pgvector` extension  
> **Target Audience:** Backend Developers, Full-Stack Engineers, and API Integrators.

---

## 📌 Table of Contents
1. [API Architecture & Standards](#1-api-architecture--standards)
2. [Health & Telemetry Endpoints](#2-health--telemetry-endpoints)
3. [RAG Ingestion & Query Endpoints](#3-rag-ingestion--query-endpoints)
4. [Evaluation & Audit Log Endpoints](#4-evaluation--audit-log-endpoints)
5. [Prompt & Multi-Model Evaluation Endpoints](#5-prompt--multi-model-evaluation-endpoints)
6. [Database Schema & Vector Index Specifications](#6-database-schema--vector-index-specifications)
7. [Relational Data Models (SQL DDL)](#7-relational-data-models-sql-ddl)

---

## 1. API Architecture & Standards

The backend runs on **FastAPI** (`src/api.py`), providing:
* Strict Pydantic model validation on all request bodies.
* Standardized JSON response formatting.
* Configurable Cross-Origin Resource Sharing (CORS) via `ALLOWED_ORIGINS`.
* Native Prometheus metrics export on `/metrics`.

**Base URL:** `http://localhost:8000`

---

## 2. Health & Telemetry Endpoints

### 2.1 Health Check
* **Endpoint:** `GET /api/health`
* **Description:** Returns service operational status.
* **Response:**
  ```json
  {
    "status": "ok",
    "service": "rag-evaluation-v2"
  }
  ```

### 2.2 Prometheus Metrics Export
* **Endpoint:** `GET /metrics`
* **Description:** Standard Prometheus metrics scrape target.
* **Content-Type:** `text/plain; version=0.0.4; charset=utf-8`

### 2.3 Synchronize Metrics from Database
* **Endpoint:** `POST /api/metrics/sync`
* **Description:** Manually reconciles and populates Prometheus counters with all historical evaluation records from PostgreSQL.
* **Response:**
  ```json
  {
    "status": "ok",
    "message": "Prometheus registry synchronized with PostgreSQL"
  }
  ```

---

## 3. RAG Ingestion & Query Endpoints

### 3.1 Run Grounded RAG Query
* **Endpoint:** `POST /api/query`
* **Description:** Performs vector retrieval against PostgreSQL, generates a grounded response using the specified LLM, and triggers an asynchronous evaluation audit.
* **Request Body:**
  ```json
  {
    "query": "What are the continuous replication lag requirements for Tier-1 DR?",
    "model_id": "gpt-6-astra",
    "top_k": 8,
    "dataset_name": "default_kb",
    "async_eval": true,
    "document_name": "cloud_disaster_recovery.txt"
  }
  ```
* **Response:**
  ```json
  {
    "query": "What are the continuous replication lag requirements for Tier-1 DR?",
    "model_id": "gpt-6-astra",
    "answer": "Under Tier-1 policy, replication lag must remain strictly below 30 seconds...",
    "retrieved_contexts": [
      {
        "document_name": "cloud_disaster_recovery.txt",
        "chunk_index": 2,
        "chunk_text": "Replication lag must remain below 30 seconds under normal operation.",
        "similarity": 0.892
      }
    ],
    "latency_ms": 1150,
    "evaluation": {
      "faithfulness_score": 1.0,
      "context_relevance_score": 0.95,
      "answer_relevance_score": 0.98,
      "confidence_score": 0.981,
      "verdict": "HIGH_CONFIDENCE",
      "unsupported_claims": [],
      "claims_analysis": [
        {
          "claim": "Replication lag must remain strictly below 30 seconds",
          "is_supported": true,
          "evidence_chunk_index": 0,
          "reasoning": "Directly confirmed in chunk 2."
        }
      ]
    }
  }
  ```

### 3.2 Dynamic Document Ingestion
* **Endpoint:** `POST /api/ingest`
* **Description:** Ingests raw text or file paths, chunks the content, computes vector embeddings, and stores them in `pgvector`.
* **Request Body:**
  ```json
  {
    "filename": "security_policy_2026.txt",
    "content": "All passwords must be at least 16 characters in length...",
    "dataset_name": "default_kb"
  }
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "document_name": "security_policy_2026.txt",
    "chunks_stored": 12
  }
  ```

### 3.3 Dynamic File Upload
* **Endpoint:** `POST /api/upload`
* **Description:** Multipart form upload accepting `.pdf`, `.txt`, and `.md` files.

### 3.4 Available Knowledge Documents
* **Endpoint:** `GET /api/documents/available`
* **Description:** Lists all documents indexed in the `document_embeddings` table.

---

## 4. Evaluation & Audit Log Endpoints

### 4.1 Fetch Evaluation History
* **Endpoint:** `GET /api/evaluations`
* **Query Parameters:**
  * `limit` (int, default: 50): Number of records to return.
  * `low_confidence_only` (bool, default: false): Filter strictly for `HALLUCINATION_DETECTED` or `LOW_QUALITY`.
* **Response:** Array of evaluation audit records with full claim decompositions.

### 4.2 Fetch Single Evaluation Detail
* **Endpoint:** `GET /api/evaluations/{eval_id}`
* **Description:** Returns full audit record including all raw JSONB chunks, claims, explanations, and model metadata.

### 4.3 Deduplicate Evaluation Records
* **Endpoint:** `DELETE /api/evaluations/duplicates`
* **Description:** Cleans up redundant evaluation runs with identical query and response payloads.

---

## 5. Prompt & Multi-Model Evaluation Endpoints

### 5.1 Multi-Model Head-to-Head Evaluation
* **Endpoint:** `POST /api/multi_model_eval`
* **Description:** Executes the same prompt across multiple models concurrently and compares outputs.
* **Request Body:**
  ```json
  {
    "prompt": "Explain the difference between optimistic and pessimistic locking.",
    "models": ["gpt-4o", "gpt-5.5", "gpt-6-astra"]
  }
  ```
* **Response:**
  ```json
  {
    "results": [
      {
        "model": "gpt-4o",
        "output": "...",
        "latency_ms": 780,
        "prompt_tokens": 45,
        "completion_tokens": 180,
        "spend_usd": 0.0022
      },
      {
        "model": "gpt-5.5",
        "output": "...",
        "latency_ms": 1250,
        "prompt_tokens": 45,
        "completion_tokens": 240,
        "spend_usd": 0.0035
      }
    ]
  }
  ```

### 5.2 Prompt Variant Evaluation
* **Endpoint:** `POST /api/prompt_eval`
* **Description:** Tests prompt variations against a test input and returns side-by-side completions and LLM-as-a-Judge scores.

---

## 6. Database Schema & Vector Index Specifications

The platform requires **PostgreSQL 16** with the **pgvector** extension installed.

### PostgreSQL Connection:
* **Host:** `localhost` (or `postgres` inside Docker)
* **Port:** `5434` (mapped from 5432)
* **User / Password:** `postgres` / `postgres`
* **Database Name:** `mathrouter`

---

## 7. Relational Data Models (SQL DDL)

```sql
-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Document Embeddings Table (The Elephant Knowledge Base)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fast Approximate Nearest Neighbor Cosine Search (HNSW)
CREATE INDEX IF NOT EXISTS idx_doc_embedding_hnsw 
ON document_embeddings USING hnsw (embedding vector_cosine_ops);

-- 3. RAG Evaluations Audit Log Table (The Elephant Audit Trail)
CREATE TABLE IF NOT EXISTS rag_evaluations (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    query TEXT NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    retrieved_contexts JSONB NOT NULL,
    response TEXT NOT NULL,
    faithfulness_score FLOAT NOT NULL,
    context_relevance_score FLOAT NOT NULL,
    answer_relevance_score FLOAT NOT NULL,
    confidence_score FLOAT NOT NULL,
    verdict VARCHAR(50) NOT NULL,
    unsupported_claims JSONB DEFAULT '[]'::jsonb,
    explanations JSONB DEFAULT '{}'::jsonb,
    latency_ms INT,
    claims_analysis JSONB DEFAULT '[]'::jsonb
);

-- Indexes for time-series aggregation, verdict queries, and model filtering
CREATE INDEX IF NOT EXISTS idx_rag_evals_verdict ON rag_evaluations(verdict);
CREATE INDEX IF NOT EXISTS idx_rag_evals_model ON rag_evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_rag_evals_created_at ON rag_evaluations(created_at DESC);

-- 4. Session & Turn Tracking Tables
CREATE TABLE IF NOT EXISTS eval_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS eval_turns (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES eval_sessions(session_id) ON DELETE CASCADE,
    turn_index INT NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT,
    tool_calls JSONB DEFAULT '[]'::jsonb,
    latency_seconds FLOAT,
    tokens_prompt INT,
    tokens_completion INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
