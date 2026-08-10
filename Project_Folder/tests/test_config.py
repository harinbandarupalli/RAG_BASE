"""Unit tests for Pydantic v2 settings and configuration loader."""

import pytest
from src.core.config import get_settings


def test_settings_load_defaults():
    settings = get_settings()
    assert settings.app.name == "Enterprise-RAG-StarterKit"
    assert settings.ingestion.chunking_strategy == "hierarchical"
    assert settings.ingestion.parent_chunk_size == 1200
    assert settings.ingestion.child_chunk_size == 250
    assert settings.agent.semantic_cache.enabled is True
    assert settings.observability.strict_remote_hub is True


def test_postgres_urls():
    settings = get_settings()
    assert "postgresql+asyncpg://" in settings.storage.postgres_async_url
    assert "postgresql+psycopg://" in settings.storage.postgres_sync_url
