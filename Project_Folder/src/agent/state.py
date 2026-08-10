"""State schema definition for the 12-Step LangGraph Agent Pipeline."""

from typing import Any, TypedDict
from langchain_core.documents import Document

from src.observability.tracker import PipelineRunTracker


class Citation(TypedDict):
    """Citation metadata for a referenced source passage."""
    source_name: str
    page_number: int
    chunk_id: str
    snippet: str
    relevance_score: float


class AgentState(TypedDict):
    """State dictionary passed and mutated across LangGraph pipeline nodes."""
    # Identification & Tracing
    session_id: str
    trace_id: str
    tracker: PipelineRunTracker

    # User Query & History
    user_query: str
    chat_history: list[dict[str, str]]

    # Step Transformations
    cleaned_query: str
    rewritten_query: str
    sub_queries: list[str]

    # Intent & Routing
    intent: str  # "rag_query" | "chit_chat" | "direct_reasoning" | "refusal"

    # Retrieval & Ranking
    raw_retrieved_docs: list[Document]
    reranked_docs: list[tuple[Document, float]]
    context_text: str

    # LLM Generation & Output
    model_provider: str | None
    model_name: str | None
    temperature: float | None
    llm_response: str
    citations: list[Citation]
    confidence_score: float

    # Flags & Guards
    is_cached: bool
    is_refusal: bool
    is_hallucination: bool
    reflection_count: int
    reflection_critique: str | None

    # Final Telemetry
    telemetry: dict[str, Any]
