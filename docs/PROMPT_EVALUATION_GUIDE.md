# Prompt Evaluation & Multi-Model Benchmarking Guide

> **Applicability:** Prompt Engineers, AI Researchers, Product Managers, and Quality Assurance Specialists.  
> **Key Capabilities:** Multi-Prompt Variant Testing, Cross-Model Head-to-Head Comparisons, Token & Cost Accounting, and Automated LLM-as-a-Judge Rubrics.

---

## 📌 Table of Contents
1. [Overview & Objectives](#1-overview--objectives)
2. [Single-Prompt vs. Multi-Prompt Testing](#2-single-prompt-vs-multi-prompt-testing)
3. [Prompt Engineering & Variant Workflows](#3-prompt-engineering--variant-workflows)
4. [Multi-Model Comparative Benchmarking](#4-multi-model-comparative-benchmarking)
5. [Automated LLM-as-a-Judge Rubrics](#5-automated-llm-as-a-judge-rubrics)
6. [Token Accounting & Cost Modeling](#6-token-accounting--cost-modeling)
7. [Running Batch Benchmark Suites](#7-running-batch-benchmark-suites)
8. [Exporting Evaluation Reports](#8-exporting-evaluation-reports)

---

## 1. Overview & Objectives

Small changes in system instructions, prompt framing, or temperature parameters can dramatically alter model output accuracy, token consumption, and tone.

The **Prompt Evaluation & Multi-Model Studio** provides a structured testbed to:
* Empirically measure the impact of prompt changes across identical test questions.
* Compare frontier LLMs side-by-side (`GPT-4o`, `GPT-5.5`, `GPT-6 Astra`, `GPT-6.1 Sol`, `Gemini 1.5 Pro`).
* Quantify latency deltas, prompt/completion token usage, and financial spend per turn.
* Automatically grade outputs against ground-truth benchmarks using a strict LLM-as-a-Judge harness.

---

## 2. Single-Prompt vs. Multi-Prompt Testing

The studio supports two primary testing modes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROMPT EVALUATION MODES                               │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ 1. Single-Prompt Testing             │ 2. Multi-Prompt Variant Testing      │
│                                      │                                      │
│ • Rapid interactive experimentation  │ • Systematic A/B/C prompt testing    │
│ • Test one query against one prompt  │ • Multiple prompt templates run      │
│ • Live inspector with JSON breakdown │   concurrently across a test matrix  │
│ • Quick iterative prompt tuning      │ • Automated head-to-head score cards │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

* **Single-Prompt Evaluation:** Ideal for interactive development. Enter a system prompt, user prompt, and generation parameters, and inspect the immediate response, token count, and latency.
* **Multi-Prompt Evaluation:** Ideal for regression testing and optimization. Define 2 to 5 distinct prompt variants (e.g. Baseline, Zero-Shot CoT, XML Grounded) and execute them against the same test set to determine the optimal configuration.

---

## 3. Prompt Engineering & Variant Workflows

### Standard Variant Architectures

1. **Variant A: Baseline Zero-Shot**
   * *Purpose:* Establish standard model baseline with minimal instruction constraints.
   * *System Prompt:* `"You are a helpful assistant. Answer the user prompt directly."`
2. **Variant B: Structured Reasoning & Grounded Rules**
   * *Purpose:* Enforce strict output structure and negative constraints.
   * *System Prompt:*
     ```markdown
     You are an accurate, strictly grounded AI assistant.
     Rules:
     1. Ground every statement strictly in provided data.
     2. Organize the answer cleanly using numbered markdown section headers.
     3. Use bullet points with bold highlights for key points.
     4. Always conclude with a succinct synthesis under a dedicated '### Conclusion' heading.
     ```
3. **Variant C: Step-by-Step Chain-of-Thought with Verification**
   * *Purpose:* Prevent logical leaps and enforce explicit intermediate verification.
   * *System Prompt:*
     ```markdown
     Think through the problem step-by-step inside <thinking> tags before answering.
     Verify each factual assertion against the provided reference before formulating your final response.
     ```

### Executing Prompt Variations in the UI
1. Navigate to **Prompt Evaluations** in the navigation rail.
2. In the configuration card, click **Add Prompt Variant** to create Variant A, B, and C tabs.
3. Paste your system instructions into each respective tab.
4. Select target temperature (`0.0` for deterministic tasks, `0.7` for creative tasks).
5. Click **Run Variant Evaluation**.
6. Inspect the side-by-side output columns and judge scoreboards.

---

## 4. Multi-Model Comparative Benchmarking

Different models exhibit varying trade-offs between reasoning capability, speed, and cost:

| Model Tier | Identifier | Typical Use Case | Trade-offs |
| :--- | :--- | :--- | :--- |
| **Frontier Reasoning** | `gpt-6-astra` | Deep logic, complex multi-step synthesis, Goldfish judging | Higher latency, maximum reasoning depth |
| **Coding & Logic** | `gpt-6.1-sol` | Code generation, schema mapping, algorithmic reasoning | High reasoning tokens |
| **Fast Reasoning** | `gpt-5.5` | Production RAG synthesis, enterprise Q&A | Balanced latency and high groundedness |
| **Omni Flagship** | `gpt-4o` | Multi-purpose conversational assistant | Fast, broad capabilities, moderate cost |
| **High-Speed Budget**| `gpt-4o-mini` | Classification, chunk filtering, lightweight tasks | Lowest latency and lowest token cost |

### Running Cross-Model Comparisons
1. Click **Compare Models** from the header or Prompt Evaluation view.
2. Select your **Model A** (e.g. `gpt-4o`) and **Model B** (e.g. `gpt-5.5` or `gemini-1.5-pro`).
3. Enter your test input or select from the pre-populated benchmark question sets.
4. Click **Run Comparative Evaluation**.
5. The UI executes both models concurrently and displays:
   * Side-by-side markdown answers with synchronized reading controls.
   * Latency delta badge (e.g. `+420ms`).
   * Token difference (Prompt vs Completion).
   * Estimated cost comparison ($USD).

---

## 5. Automated LLM-as-a-Judge Rubrics

To eliminate subjective manual grading, the platform incorporates an automated **LLM-as-a-Judge** scoring harness.

### Evaluation Criteria (1 - 5 Scale)
* **Accuracy & Factuality:** Does the response contain any incorrect statements or false assertions?
* **Completeness & Coverage:** Did the model answer all sub-parts and criteria requested in the user query?
* **Instruction Following:** Did the model adhere strictly to formatting rules, length constraints, and negative boundaries?
* **Conciseness & Clarity:** Is the answer free of unnecessary filler, disclaimers, and rambling prose?

### Judge Output Schema
The judge generates structured JSON containing:
```json
{
  "scores": {
    "accuracy": 5,
    "completeness": 4,
    "instruction_following": 5,
    "clarity": 5
  },
  "overall_rating": 4.75,
  "rationale": "Variant B adhered strictly to all numbered section headers and provided exact citations for each claim.",
  "recommended_variant": "Variant B"
}
```

---

## 6. Token Accounting & Cost Modeling

Real-time token tracking runs on every evaluation turn to track inference expenditure:

### Cost Calculation Matrix (Indicative)
$$\text{Cost} = \left(\frac{\text{Prompt Tokens}}{1,000,000} \times \text{Rate}_{\text{input}}\right) + \left(\frac{\text{Completion Tokens}}{1,000,000} \times \text{Rate}_{\text{output}}\right)$$

### Console Accounting Features
* **Live Spend Counter:** Displayed in the top application header, tracking total accumulated cost across all active evaluation runs.
* **Per-Turn Cost Breakdown:** View exact input token cost, output token cost, and total turn expense in the Turn Inspector.
* **Historical Database Spend:** Direct integration with PostgreSQL to aggregate long-term expenditure across teams and models.

---

## 7. Running Batch Benchmark Suites

The repository includes automated batch evaluation scripts to evaluate models over 100 benchmark questions:

```bash
# Activate your virtual environment
source .venv/bin/activate

# 1. Run the standard 100-questions benchmark suite
python3 run_100_questions_benchmark.py

# 2. Run the philosophical / complex reasoning 100-questions benchmark
python3 run_philosophical_100_benchmark.py

# 3. Run a quick 5-question sample benchmark
python3 run_benchmark_sample.py
```

### Benchmark Artifacts
The benchmark scripts automatically export:
* `benchmark_100_results.json`: Complete raw JSON traces for all 100 queries.
* `rag_benchmark_comparison_100.pdf`: Formatted executive PDF report with score distributions and win/loss rates.

---

## 8. Exporting Evaluation Reports

From the Evaluation Console UI, you can export your prompt evaluation results:
* **Export PDF:** Click **Export PDF** to generate an executive report including the prompt configuration, side-by-side model outputs, token spend, and judge rationale.
* **Export XLSX:** Download a clean Excel spreadsheet of all evaluated test cases with column-by-column score breakdowns.
* **Copy JSON:** Copy the full raw evaluation telemetry to your clipboard for external CI/CD pipelines.
