"""Step 10: Citation Grounding, Attribution Mapping & Semantic Caching.

Assembles source attribution cards with page numbers, scores, and text snippets.
Caches successful query-response pairs in Redis for instant future recall.
"""

from typing import Any

from src.adapters.databases.redis_cache import RedisCacheAdapter
from src.adapters.embeddings.factory import EmbeddingFactory
from src.agent.state import AgentState, Citation
from src.core.logger import get_logger

logger = get_logger(__name__)


async def step10_synthesizer(state: AgentState) -> dict[str, Any]:
    """Execute Step 10: Citation mapping and cache storage."""
    tracker = state["tracker"]
    tracker.start_span("step10_synthesizer")

    reranked = state.get("reranked_docs", [])
    citations: list[Citation] = []

    for doc, score in reranked:
        snippet = doc.page_content.strip()[:200] + "..." if len(doc.page_content) > 200 else doc.page_content.strip()
        citations.append(
            Citation(
                source_name=doc.metadata.get("source_name", "Document"),
                page_number=int(doc.metadata.get("page_number", 1)),
                chunk_id=doc.metadata.get("chunk_id", ""),
                snippet=snippet,
                relevance_score=round(score, 4),
            )
        )

    # Cache successful answer in Redis Semantic Cache
    query = state.get("user_query", "")
    response_text = state.get("llm_response", "")

    if query and response_text and not state.get("is_refusal"):
        try:
            embeddings = EmbeddingFactory.create_embeddings()
            query_emb = await embeddings.aembed_query(query)
            await RedisCacheAdapter.set_cache_entry(
                query=query,
                query_embedding=query_emb,
                response=response_text,
                citations=citations,
            )
        except Exception as err:
            logger.debug(f"Could not populate semantic cache: {err}")

    tracker.end_span("step10_synthesizer", {"citations_count": len(citations)})
    return {"citations": citations}
