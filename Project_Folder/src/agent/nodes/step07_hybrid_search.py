"""Step 7: Hybrid Fusion via Reciprocal Rank Fusion (RRF) & Small-to-Big Expansion.

Merges dense and sparse candidates using RRF: RRF(d) = SUM(1 / (60 + rank)),
deduplicates identical chunks across sub-queries, and resolves child chunks
into full parent documents (Small-to-Big).
"""

from typing import Any
from langchain_core.documents import Document

from src.adapters.vectorstores.pgvector_adapter import PGVectorAdapter
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


async def step07_hybrid_search(state: AgentState) -> dict[str, Any]:
    """Execute Step 7: Reciprocal Rank Fusion and Small-to-Big parent expansion."""
    tracker = state["tracker"]
    tracker.start_span("step07_hybrid_search")

    settings = get_settings()
    k_constant = settings.agent.rrf_k
    raw_docs: list[Document] = state.get("raw_retrieved_docs", [])

    if not raw_docs:
        tracker.end_span("step07_hybrid_search", {"candidates": 0})
        return {"raw_retrieved_docs": []}

    # 1. Group documents by method and compute RRF scores
    rrf_scores: dict[str, float] = {}
    doc_registry: dict[str, Document] = {}

    dense_docs = [d for d in raw_docs if d.metadata.get("retrieval_method") == "dense"]
    sparse_docs = [d for d in raw_docs if d.metadata.get("retrieval_method") == "sparse"]

    for rank, doc in enumerate(dense_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id") or doc.page_content[:64]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_constant + rank))
        doc_registry[chunk_id] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        chunk_id = doc.metadata.get("chunk_id") or doc.page_content[:64]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_constant + rank))
        doc_registry[chunk_id] = doc

    # 2. Sort documents descending by fused RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused_docs: list[Document] = []
    for cid in sorted_ids:
        doc = doc_registry[cid]
        doc.metadata["rrf_score"] = round(rrf_scores[cid], 5)
        fused_docs.append(doc)

    logger.debug(f"RRF fusion sorted {len(fused_docs)} unique documents")

    # 3. Hierarchical (Small-to-Big) Parent Expansion
    vector_adapter = PGVectorAdapter()
    expanded_docs = await vector_adapter.resolve_parent_documents(fused_docs)

    tracker.end_span(
        "step07_hybrid_search",
        {
            "unique_candidates": len(fused_docs),
            "expanded_parents": len(expanded_docs),
        },
    )

    return {"raw_retrieved_docs": expanded_docs}
