# Project Guidelines & Development Protocol

## Core Operating Protocol: Plan -> Review -> Implement After Approval

All development, architectural modifications, and code generation in this project must strictly adhere to the following 3-stage lifecycle:

```
┌─────────────────┐       ┌─────────────────┐       ┌────────────────────────┐
│     1. PLAN     │  ───> │    2. REVIEW    │  ───> │ 3. IMPLEMENT AFTER     │
│ Brainstorm &    │       │ Present to user │       │    APPROVAL ONLY       │
│ Design Specs    │       │ for alignment   │       │ Execute code & verify  │
└─────────────────┘       └─────────────────┘       └────────────────────────┘
```

### Stage 1: Plan
- Analyze the user request, dependencies, and integration touchpoints.
- Design modular solutions adhering to Hexagonal / Clean Architecture.
- Detail folder structures, configuration keys, data schemas, and edge case handling.
- Produce a clear, itemized proposal before making any code or environment changes.

### Stage 2: Review
- Present the plan explicitly to the user.
- Highlight architectural choices, trade-offs, and external dependencies.
- Await the user's feedback, refinements, or explicit approval.
- **Never proceed to implementation until the user gives explicit consent/approval.**

### Stage 3: Implement After Approval
- Once approved, implement the planned changes incrementally.
- Write clean, type-annotated, production-ready Python / TypeScript code.
- Verify through tests, linter checks, and sanity runs.
- Document changes, configuration updates, and usage instructions.

---

## Architectural Principles for LangChain Starter Kit (`RAG_BASE`)

1. **Decoupled Adapters (Hexagonal Architecture)**
   - Core domain logic must never directly couple to specific LLM vendors or vector databases.
   - Use adapter interfaces (`BaseLLMAdapter`, `BaseVectorStoreAdapter`, `BaseEmbeddingAdapter`).

2. **Configuration-Driven (Pydantic v2)**
   - All settings, credentials, endpoints, and RAG hyperparameters are governed by `src/core/config.py` backed by `config/settings.yaml` and `.env`.
   - Support dynamic runtime overrides for model switching and prompt variants.

3. **Pluggable Observability & Prompt Desk**
   - Seamless toggling between **LangSmith**, **Langfuse**, and **Arize Phoenix**.
   - Strict remote prompt hub synchronization (Single Source of Truth).
   - **NO local fallback**: If the prompt cannot be retrieved from the observability desk/hub or is missing, fail fast with a strict error (`PromptHubError`) to guarantee governance, auditability, and production prompt integrity.
   - Built-in metrics collection (token cost, latency, error rates).

4. **Resilience & Task Supervisor**
   - Provide cancellation tokens / process kill switches for long-running or runaway generations.
   - Circuit breakers, timeout watchdogs, and token budget guards.

5. **Async-First & Production-Ready**
   - Fast, non-blocking asynchronous APIs (FastAPI + SSE streaming).
   - Async database connections (`asyncpg` / `SQLAlchemy async`) with pgvector support.
   - Comprehensive containerization (`docker-compose.yml`) and testing.
