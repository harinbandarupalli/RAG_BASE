"""Health, Liveness, and Infrastructure Probes."""

from typing import Any
from fastapi import APIRouter

from src.adapters.databases.postgres import PostgresManager
from src.adapters.databases.redis_cache import RedisCacheAdapter
from src.core.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Health & Status"])


@router.get("/health", summary="System Health & Infrastructure Probes")
async def health_check() -> dict[str, Any]:
    """Check connectivity to PostgreSQL (pgvector) and Redis Semantic Cache."""
    settings = get_settings()

    postgres_ok = await PostgresManager.check_health()
    redis_ok = await RedisCacheAdapter.check_health()

    status = "healthy" if postgres_ok and redis_ok else "degraded"

    return {
        "status": status,
        "app_name": settings.app.name,
        "environment": settings.app.environment,
        "infrastructure": {
            "postgres_pgvector": "connected" if postgres_ok else "disconnected",
            "redis_semantic_cache": "connected" if redis_ok else "disconnected",
        },
        "default_model": f"{settings.llm.default_provider}/{settings.llm.default_model}",
        "observability_backend": settings.observability.backend,
    }
