# Evaluation Console V2 — Comprehensive User Guide

> **Applicability:** AI Evaluators, System Administrators, Prompt Engineers, and Data Analysts.  
> **Interface Standard:** Universal UI Guide V2 Compliance (Three-Panel Geometry, 900px Viewport Floor).

---

## 📌 Table of Contents
1. [Introduction & Layout Anatomy](#1-introduction--layout-anatomy)
2. [Supported Viewport Floor & Responsiveness](#2-supported-viewport-floor--responsiveness)
3. [Navigation & Workspace Views](#3-navigation--workspace-views)
4. [Using the RAG Triad Evaluation Studio](#4-using-the-rag-triad-evaluation-studio)
5. [Using the Prompt Evaluation Studio](#5-using-the-prompt-evaluation-studio)
6. [Multi-Model Comparative Evaluation Modal](#6-multi-model-comparative-evaluation-modal)
7. [The All Experiments Data Hub](#7-the-all-experiments-data-hub)
8. [Database Explorer & Schema Viewer](#8-database-explorer--schema-viewer)
9. [Telemetry & Metrics Dashboard](#9-telemetry--metrics-dashboard)
10. [Exporting Evaluation Reports (PDF, XLSX, JSON)](#10-exporting-evaluation-reports-pdf-xlsx-json)

---

## 1. Introduction & Layout Anatomy

The **Evaluation Console V2** is built with React 19 and Vite 8, implementing the strict Three-Panel Architecture defined in the Universal UI Guide V2.

```
┌──────┬──────────────────────┬────────────────────────────────────────────────────────┐
│ Rail │ Panel 2: Explorer    │ Panel 3: Inspection & Working Canvas                   │
│ 56px │ (360px - 364px)      │ (Min 480px Floor, Flex-grow)                          │
├──────┼──────────────────────┼────────────────────────────────────────────────────────┤
│ [📁] │ • Retained Sessions  │ • Active Workspace Canvas                              │
│ [🎯] │ • Filter by Model    │ • Real-time Trace & Turn Inspector                     │
│ [⚡] │ • Search Runs        │ • Side-by-Side Model Outputs                           │
│ [📊] │ • Timestamp Sorting  │ • RAG Triad Score Cards (Faithfulness, Relevance)      │
│ [🗄️] │ • Status Badges      │ • Atomic Claim Decomposition & Hallucination Flags     │
│ [📈] │ • Bulk Actions       │ • Raw JSON & Telemetry Metrics Inspector               │
│ [🌙] │                      │ • One-Click Export Toolbar (PDF, XLSX, Copy JSON)      │
└──────┴──────────────────────┴────────────────────────────────────────────────────────┘
```

### The Three Panels
* **Panel 1 (Navigation Rail):** Compact 56px left rail providing instantaneous switching between the core workspaces. Expandable to 180–320px for labeled navigation.
* **Panel 2 (Session Explorer):** 360px–364px retained column listing historical evaluation sessions, live runs, and recorded benchmarks. Includes instant search, model tags, and deletion controls.
* **Panel 3 (Task & Detail Inspector):** 480px minimum width pane that displays chronological turns, processing sequences, model comparisons, and evaluation score breakdowns.

---

## 2. Supported Viewport Floor & Responsiveness

The console enforces a **strict 900px minimum desktop viewport floor** via the `SupportedViewportGate` component:
* If the window width falls below 900px, the console automatically replaces the workspace with a clear **Supported Viewport Gate Notice**, instructing the user to expand their browser window.
* This ensures that multi-column comparison tables, side-by-side JSON payloads, and split inspector panes never clip, collapse, or cause unreadable horizontal scrollbars.
* When resizing above 900px, your current session, selected tab, and inspection scroll positions are **fully restored in memory**.

---

## 3. Navigation & Workspace Views

Using the left Navigation Rail, you can switch between the following dedicated views:

| View Name | Icon | Core Functionality |
| :--- | :--- | :--- |
| **All Experiments Data Hub** | `Layers` | Unified central command table showing all evaluation runs, aggregate metrics, and bulk exports. |
| **RAG Triad Studio** | `Sparkles` | Ingestion, semantic search, RAG execution, and atomic hallucination scoring. |
| **Prompt Evaluations** | `Sliders` | Prompt variant testing (A/B/C), system instruction tuning, and parameter optimization. |
| **Session Explorer** | `Database` | Deep dive into turn-by-turn agent and model interaction recordings. |
| **Database Explorer** | `Server` | Direct tabular inspection of PostgreSQL tables (`rag_evaluations`, `document_embeddings`). |
| **Telemetry & Metrics** | `Activity` | Real-time Prometheus metrics viewer and direct Grafana telemetry links. |

---

## 4. Using the RAG Triad Evaluation Studio

The **RAG Triad Studio** (`RagEvaluationView`) is the core workspace for auditing Retrieval-Augmented Generation workflows.

### Step 1: Upload or Ingest Knowledge Documents
1. Navigate to **RAG Evaluation** from the navigation rail.
2. In the top toolbar, click **Upload Document** or choose from available pre-indexed documents in the dropdown:
   * Supported formats: `.pdf`, `.txt`, `.md`.
   * The platform automatically chunks the text and computes 1536-dimensional embeddings using `text-embedding-3-small`.
   * Vectors and chunk metadata are committed directly to PostgreSQL (`document_embeddings`).

### Step 2: Formulate and Submit a Query
1. Enter your test prompt or question in the **Query Input** box.
2. Select the **Generation Model** from the dropdown:
   * `gpt-6-astra` (Flagship reasoning & synthesis)
   * `gpt-6.1-sol` (High-end coding and logical deduction)
   * `gpt-5.5` (Fast, high-precision generation)
   * `gpt-4o` (Multi-purpose default)
   * `gpt-4o-mini` (High-speed budget tier)
3. Set `Top-K Chunks` (default: 8) to control retrieval depth.
4. Click **Run RAG Query**.

### Step 3: Inspect Retrieval & RAG Triad Scores
Once the query completes, Panel 3 displays the evaluation breakdown:
* **Retrieved Chunks Panel:** View each chunk used in the context, along with its source document name, chunk index, and cosine similarity distance.
* **RAG Triad Score Cards:**
  * **Faithfulness / Groundedness ($0.0 - 1.0$):** The proportion of claims in the generated response that are directly backed by the retrieved excerpts.
  * **Context Relevance ($0.0 - 1.0$):** How cleanly the retrieved chunks match the query without irrelevant noise.
  * **Answer Relevancy ($0.0 - 1.0$):** How completely and directly the response answers the user's intent.
  * **Composite Confidence Score ($0\% - 100\%$):** Weighted aggregate score combining all three triad metrics.
* **Verdict Badge:** Instantly see if the response was graded `HIGH_CONFIDENCE`, `MEDIUM_CONFIDENCE`, `HALLUCINATION_DETECTED`, or `LOW_QUALITY`.

### Step 4: Atomic Claim & Hallucination Breakdown
Under the **Claims Verification** tab:
* Every sentence of the generated answer is decomposed into an individual atomic claim.
* Claims supported by context are marked with a green checkmark `[Supported]` and link to the exact context chunk index.
* Any unsupported or extrapolated claim is highlighted with a red warning badge `[Hallucination Detected]` along with the auditor's reasoning.

---

## 5. Using the Prompt Evaluation Studio

The **Prompt Evaluation Studio** (`PromptEvaluationView`) enables systematic testing of prompt templates, system instructions, and generation parameters.

### Workflow:
1. **Define Test Cases:** Enter one or more prompt inputs or select from standard benchmark questions.
2. **Configure Prompt Variations:**
   * **Variant A (Baseline):** Standard system prompt.
   * **Variant B (Structured / Strict Grounding):** Step-by-step instructions with negative constraints.
   * **Variant C (Chain-of-Thought / XML Tags):** Structured reasoning tags.
3. **Run Batch Evaluation:**
   * The backend dispatches test cases asynchronously across variants.
   * Outputs are rendered in a clean side-by-side comparison view.
4. **Automated LLM-as-a-Judge Scoring:**
   * The Goldfish judge rates each variant on:
     * Accuracy & Task Completion ($1 - 5$)
     * Instruction Following ($1 - 5$)
     * Conciseness & Formatting ($1 - 5$)
5. **Cost & Latency Comparison:**
   * Review prompt tokens, completion tokens, calculated monetary spend, and end-to-end latency per variant.

---

## 6. Multi-Model Comparative Evaluation Modal

Clicking the **Compare Models** button launches the `MultiModelEvaluationModal`:
* **Side-by-Side Execution:** Run the exact same query simultaneously across multiple target models (e.g. `GPT-4o` vs. `GPT-5.5` vs. `Gemini 1.5 Pro`).
* **Synchronized Scrolling:** Compare answers side-by-side with synchronized reading views.
* **Comparative Metrics Table:**
  * Roundtrip latency comparison (ms)
  * Token consumption deltas (Prompt vs Completion)
  * Calculated financial spend ($USD)
  * LLM Judge head-to-head preference winner

---

## 7. The All Experiments Data Hub

The **All Experiments Data Hub** (`AllExperimentsDataHub`) aggregates every evaluation ever run in the platform:
* **Interactive Filtering:** Filter records by generation model, verdict (`HIGH_CONFIDENCE`, `HALLUCINATION_DETECTED`), confidence score range, or date.
* **Instant Search:** Search across query text, response content, and source document filenames.
* **Quick Metrics Summary:** High-level cards display:
  * Total Experiments Run
  * Average Confidence Score
  * Hallucination Rate Percentage
  * Total Estimated Spend ($USD)
* **Bulk Export:** Export the filtered dataset to XLSX or structured JSON with a single click.

---

## 8. Database Explorer & Schema Viewer

Navigate to **Database** in the navigation rail to access direct tabular inspections:
* **Active Tables Available:**
  * `rag_evaluations`: Complete audit logs including atomic claims, scores, latencies, and verdicts.
  * `document_embeddings`: Vectorized document chunks, embedding metadata, and document identifiers.
  * `eval_sessions` & `eval_turns`: Historical conversation sessions and turn traces.
* **Search & Inspect:** Search within raw database rows, copy record IDs, and view raw JSON payloads.

---

## 9. Telemetry & Metrics Dashboard

Navigate to **Metrics** in the navigation rail:
* Displays live Prometheus scrape counters directly from `/metrics`.
* Monitored counters include:
  * `rag_eval_queries_total`: Total queries grouped by model and verdict.
  * `rag_eval_confidence_score`: Average and p95 confidence distribution.
  * `rag_eval_faithfulness_ratio`: Continuous groundedness metric tracking.
  * `rag_eval_latency_seconds`: Latency histograms.
* **Grafana Integration:** Click the **Open in Grafana** button in the header to jump directly to pre-configured Grafana dashboards at `http://localhost:3000`.

---

## 10. Exporting Evaluation Reports (PDF, XLSX, JSON)

The console provides built-in export mechanisms:
* **Executive PDF Report:** Click **Export PDF** in the top action bar to generate a styled, multi-page PDF evaluation summary containing executive metrics, RAG Triad score distribution charts, and full audit logs.
* **Spreadsheet (XLSX) Export:** Click **Export XLSX** to generate an Excel spreadsheet matching the exact visible columns, filters, and values of the active view.
* **Raw JSON Clipboard Copy:** Click the **Copy JSON** icon on any evaluation turn or session card to copy the raw payload and provenance data directly to your clipboard.
