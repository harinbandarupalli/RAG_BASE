"""Step 1.5: Redis Semantic Cache Check.

Computes query embedding and checks Redis cache for semantically identical queries
(cosine similarity > 0.96). If hit, returns in <40ms with 0 tokens.
"""

from typing import Any

from src.adapters.databases.redis_cache import RedisCacheAdapter
from src.adapters.embeddings.factory import EmbeddingFactory
from src.agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)


async def step01_cache(state: AgentState) -> dict[str, Any]:
    """Execute Step 1.5: Semantic Cache lookup."""
    tracker = state["tracker"]
    tracker.start_span("step01_cache")

    query = state.get("user_query", "")

    try:
        # Embed query
        embeddings = EmbeddingFactory.create_embeddings()
        query_emb = await embeddings.aembed_query(query)

        # Lookup in Redis Semantic Cache
        cached_entry = await RedisCacheAdapter.get_semantic_match(query_emb)
        if cached_entry:
            tracker.end_span("step01_cache", {"hit": True, "score": cached_entry.get("similarity_score")})
            return {
                "is_cached": True,
                "llm_response": cached_entry.get("response", ""),
                "citations": cached_entry.get("citations", []),
                "confidence_score": 1.0,
            }

    except Exception as err:
        logger.debug(f"Semantic cache lookup skipped: {err}")

    tracker.end_span("step01_cache", {"hit": False})
    return {"is_cached": False}
