"""Redis client and Semantic Cache adapter.

Provides:
- Async Redis connection management and ping health check
- Cosine similarity based Semantic Caching:
  Returns cached responses for semantically identical queries in <40ms with 0 tokens.
- Graceful degradation: if Redis is offline, cache simply misses and logs warning.
"""

import json
import math
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RedisCacheAdapter:
    """Adapter for Redis operations and Semantic Caching."""

    _client: Redis | None = None

    @classmethod
    def get_client(cls) -> Redis:
        """Get or initialize the async Redis client."""
        if cls._client is None:
            settings = get_settings()
            cls._client = aioredis.from_url(
                settings.storage.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3.0,
            )
        return cls._client

    @classmethod
    async def check_health(cls) -> bool:
        """Ping Redis to verify connectivity."""
        try:
            client = cls.get_client()
            return await client.ping()
        except Exception as err:
            logger.warning(f"Redis health check failed: {err}")
            return False

    @classmethod
    async def get_semantic_match(
        cls,
        query_embedding: list[float],
        threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """Search cached entries using cosine similarity on embeddings."""
        settings = get_settings()
        if not settings.agent.semantic_cache.enabled:
            return None

        sim_threshold = threshold or settings.agent.semantic_cache.similarity_threshold

        try:
            client = cls.get_client()
            # Scan all semantic cache keys
            cursor = 0
            best_match: dict[str, Any] | None = None
            highest_score = -1.0

            while True:
                cursor, keys = await client.scan(cursor=cursor, match="sem_cache:*", count=100)
                if keys:
                    values = await client.mget(keys)
                    for raw in values:
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                            cached_emb = entry.get("embedding")
                            if not cached_emb:
                                continue
                            score = cosine_similarity(query_embedding, cached_emb)
                            if score > highest_score and score >= sim_threshold:
                                highest_score = score
                                best_match = entry
                                best_match["similarity_score"] = round(score, 4)
                        except json.JSONDecodeError:
                            continue

                if cursor == 0:
                    break

            if best_match:
                logger.info(
                    f"✔ Semantic Cache Hit! Score={best_match['similarity_score']:.4f} "
                    f"(Original: '{best_match.get('query')[:40]}...')"
                )
                return best_match

            return None

        except Exception as err:
            logger.warning(f"Semantic cache lookup encountered error (degrading gracefully): {err}")
            return None

    @classmethod
    async def set_cache_entry(
        cls,
        query: str,
        query_embedding: list[float],
        response: str,
        citations: list[dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> None:
        """Store query embedding, generated response, and citations in semantic cache."""
        settings = get_settings()
        if not settings.agent.semantic_cache.enabled:
            return

        ttl = ttl_seconds or settings.agent.semantic_cache.ttl_seconds

        try:
            client = cls.get_client()
            key = f"sem_cache:{hash(query)}_{len(query)}"
            payload = {
                "query": query,
                "embedding": query_embedding,
                "response": response,
                "citations": citations,
            }
            await client.set(key, json.dumps(payload), ex=ttl)
            logger.debug(f"Cached semantic query entry with TTL={ttl}s")
        except Exception as err:
            logger.warning(f"Failed to write semantic cache entry: {err}")

    @classmethod
    async def cache_parent_document(
        cls,
        parent_id: str,
        content: str,
        metadata: dict[str, Any],
        ttl_seconds: int = 86400,
    ) -> None:
        """Cache a full parent document section in Redis RAM for sub-2ms lookup."""
        try:
            client = cls.get_client()
            key = f"parent_cache:{parent_id}"
            payload = json.dumps({"parent_id": parent_id, "content": content, "metadata": metadata})
            await client.set(key, payload, ex=ttl_seconds)
            logger.debug(f"Cached parent document in Redis: {key}")
        except Exception as err:
            logger.debug(f"Failed to cache parent document in Redis: {err}")

    @classmethod
    async def get_cached_parent_document(cls, parent_id: str) -> dict[str, Any] | None:
        """Fetch cached parent document from Redis RAM."""
        try:
            client = cls.get_client()
            key = f"parent_cache:{parent_id}"
            raw = await client.get(key)
            if raw:
                logger.debug(f"✔ Parent document RAM Cache Hit: {key}")
                return json.loads(raw)
            return None
        except Exception as err:
            logger.debug(f"Redis parent document cache lookup failed: {err}")
            return None

    @classmethod
    async def cache_chunk_embedding(cls, text_hash: str, embedding: list[float], ttl_seconds: int = 604800) -> None:
        """Cache computed embedding vector for a chunk to prevent re-billing embedding APIs."""
        try:
            client = cls.get_client()
            key = f"emb_cache:{text_hash}"
            await client.set(key, json.dumps(embedding), ex=ttl_seconds)
        except Exception as err:
            logger.debug(f"Failed to cache chunk embedding: {err}")

    @classmethod
    async def get_cached_chunk_embedding(cls, text_hash: str) -> list[float] | None:
        """Fetch cached embedding vector for a chunk."""
        try:
            client = cls.get_client()
            key = f"emb_cache:{text_hash}"
            raw = await client.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception as err:
            return None

    @classmethod
    async def close(cls) -> None:
        """Close active Redis connection."""
        if cls._client is not None:
            await cls._client.aclose()
            cls._client = None
            logger.info("Redis client closed")
