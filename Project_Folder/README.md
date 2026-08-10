# Enterprise LangChain & LangGraph RAG Starter Kit

A production-grade, battle-tested starter kit for building modular Retrieval-Augmented Generation (RAG) applications with **LangChain v0.3**, **LangGraph**, **PostgreSQL (pgvector)**, **Redis**, and **FastAPI**.

Designed using **Hexagonal (Clean) Architecture**, separating the headless domain engine (`src/`) from infrastructure adapters and swappable presentation layers (`ui/`).

---

## 🚀 How to Copy and Customize This Starter Kit

To create your own standalone project from this starter kit:

```bash
# 1. Copy the entire Project_Folder to your new project location
cp -r Project_Folder my_enterprise_rag

# 2. Navigate to your new directory
cd my_enterprise_rag

# 3. Create your environment configuration
cp .env.example .env

# 4. Fill in your LLM API keys and credentials in .env
```

---

## 🏗️ Architecture Overview

The starter kit enforces a **Two-Path Separation**:

### Path A: Data Ingestion Pipeline (`src/ingestion/`)
1. **Layout-Aware Visual Extraction**: Extract tables cleanly into Markdown format, preserve document hierarchy, and run OCR on images.
2. **Hierarchical (Small-to-Big) Chunking**: Embeds small child chunks (~200 tokens) in `pgvector` for vector search precision, linked via `parent_id` to large parent chunks (~1,200 tokens) stored in PostgreSQL for synthesis context.
3. **Rich Metadata Schema & Deduplication**: Chunks store SHA-256 checksums, page numbers, breadcrumbs, and token counts. Existing files are deduplicated automatically.
4. **Dual Indexing**: Dense vector search (HNSW Cosine) + Sparse keyword search (PostgreSQL `tsvector` with GIN inverted index).

### Path B: 12-Step Agentic Query Pipeline (`src/agent/`)
Built as a deterministic **LangGraph State Machine**:
- **Step 1**: Input Guardrail & Moderation.
- **Step 1.5**: Redis Semantic Cache (instant $<40\text{ms}$ return on cosine similarity $> 0.96$).
- **Step 2**: Intent Classification & Routing.
- **Step 3**: Query Cleaning & Sanitization.
- **Step 4**: Contextual Query Rewriting with PostgreSQL chat history.
- **Step 5**: Query Decomposition into atomic sub-queries.
- **Step 6**: Parallel concurrent retrieval across all sub-queries.
- **Step 7**: Hybrid Search via Reciprocal Rank Fusion (RRF) + Cross-Query Deduplication.
- **Step 8**: Cross-Encoder Reranking (FlashRank / Cohere).
- **Step 8.5**: Adaptive Confidence Gate (refusal or fallback if relevance is below threshold).
- **Step 9**: Processing with LLM using **Strict Remote Prompt Hub** (**FAIL-FAST, NO local fallback**).
- **Step 10**: Synthesizer with citation grounding `[source.pdf, p. 4]` and attribution mapping.
- **Step 11**: Output Guardrails (Self-RAG hallucination check with a self-correction reflection loop + PII redaction).
- **Step 12**: Bidirectional WebSocket Streaming with TTFT tracking, token & latency breakdown, and client-initiated kill switch.

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose
- API Key for at least one LLM provider (OpenAI, Anthropic, Google Gemini, or local Ollama)

### 2. Infrastructure Setup (Postgres + pgvector & Redis)
```bash
docker compose up -d postgres redis
```
*(Optional: Run `docker compose --profile observability up -d` to launch local Arize Phoenix on port 6006)*

### 3. Install Dependencies & Run Database Migrations
```bash
# Install dependencies in editable mode
pip install -e ".[dev]"

# Run Alembic database migrations to head revision
make db-migrate
```

To auto-generate a new migration after modifying Python ORM models in `src/models/schema.py`:
```bash
make db-revision msg="add_new_column_or_table"
```

### 4. Start the Services
- **Backend API (FastAPI + WebSockets)**:
  ```bash
  make run-api
  # API docs available at: http://localhost:8000/docs
  ```

- **Frontend UI (Streamlit Decoupled UI)**:
  ```bash
  make run-ui
  # UI available at: http://localhost:8501
  ```

---

## 📊 Observability & Strict Remote Prompt Hub

This starter kit enforces a **Strict Remote Prompt Hub Policy (No Local Fallback)**:
- Prompts are retrieved dynamically from **Langfuse** or **LangSmith** by prompt name and version tag.
- In-memory caching with TTL (default: 5 minutes) ensures rapid query execution.
- If a prompt is missing or the remote hub is unreachable, the system raises `PromptHubError` immediately. This guarantees prompt governance and prevents untracked local prompt drift.

---

## 🧹 Clean Code & Logging Standards

- **Log Helper (`src/core/logger.py`)**:
  - Contextual metadata binding: `trace_id`, `session_id`, `step_name`.
  - Zero-leak secret and PII redactor: automatically masks API keys and credentials.
  - Formats: Colorized console in dev; JSON lines in production.
  - `@timed_block("step_name")` decorator for exact millisecond latency measurement.
- **Decoupled Presentation Layer (`ui/`)**:
  - The UI directory is 100% decoupled from the backend. You can replace or rewrite `ui/` using Next.js, React, or Vue without modifying a single line in `src/`.
