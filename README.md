# LangChain Enterprise Starter Kit (`RAG_BASE`)

Welcome to the **Enterprise LangChain Starter Kit**.

This repository contains a modular, production-ready RAG application template located entirely inside the **[`Project_Folder/`](file:///c:/Users/harik/Documents/projects/RAG_BASE/Project_Folder)** directory.

---

## 🚀 Quick Usage

To create a new project:
```bash
# Copy Project_Folder to your target project directory
cp -r Project_Folder my_rag_app
cd my_rag_app
cp .env.example .env
```

Review the [Project_Folder/README.md](file:///c:/Users/harik/Documents/projects/RAG_BASE/Project_Folder/README.md) for complete instructions.

---

## 📖 Key Architectural Standards
- **Development Protocol**: Strict `Plan -> Review -> Implement After Approval` defined in [`gemini.md`](file:///c:/Users/harik/Documents/projects/RAG_BASE/gemini.md).
- **Two-Path Separation**: Path A (Ingestion with Visual Extraction, Hierarchical Small-to-Big chunking, BM25 GIN indexing) and Path B (12-Step LangGraph Agentic Pipeline with Semantic Caching and WebSockets).
- **Strict Remote Prompt Hub**: Single source of truth with strict fail-fast policy (NO local fallback).
- **Decoupled UI**: Presentation layer isolated in `ui/`, completely swappable with Next.js, React, or Vue.
