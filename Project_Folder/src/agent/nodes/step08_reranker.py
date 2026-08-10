"""Step 8: Cross-Encoder Reranking.

Reranks candidate passages using deep cross-attention (FlashRank local CPU or Cohere API)
to produce accurate relevance scores before context construction.
"""

from typing import Any
from langchain_core.documents import Document

from src.adapters.rerankers.reranker import RerankerAdapter
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


async def step08_reranker(state: AgentState) -> dict[str, Any]:
    """Execute Step 8: Cross-encoder reranking."""
    tracker = state["tracker"]
    tracker.start_span("step08_reranker")

    settings = get_settings()
    query = state.get("rewritten_query") or state.get("user_query", "")
    candidates: list[Document] = state.get("raw_retrieved_docs", [])

    if not candidates:
        tracker.end_span("step08_reranker", {"reranked_count": 0})
        return {"reranked_docs": []}

    reranker = RerankerAdapter()
    reranked = await reranker.rerank(
        query=query,
        documents=candidates,
        top_n=settings.agent.reranker.top_n,
    )

    top_score = reranked[0][1] if reranked else 0.0
    tracker.end_span(
        "step08_reranker",
        {
            "input_candidates": len(candidates),
            "output_top_n": len(reranked),
            "top_relevance_score": round(top_score, 4),
        },
    )

    logger.info(f"Reranking selected top {len(reranked)} documents (top_score={top_score:.4f})")
    return {"reranked_docs": reranked}
