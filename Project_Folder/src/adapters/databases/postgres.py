"""PostgreSQL async connection manager with pgvector extension auto-provisioning.

Provides:
- Async engine & connection pool via SQLAlchemy + asyncpg
- Table creation for Hierarchical Small-to-Big Parent Documents
- Health check probe
- pgvector extension validation
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import get_settings
from src.core.exceptions import DatabaseConnectionError
from src.core.logger import get_logger

logger = get_logger(__name__)


class PostgresManager:
    """Singleton manager for PostgreSQL async database connections and schema setup."""

    _instance: "PostgresManager | None" = None
    _engine: AsyncEngine | None = None
    _sessionmaker: async_sessionmaker[AsyncSession] | None = None

    def __new__(cls) -> "PostgresManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_engine(cls) -> AsyncEngine:
        """Get or initialize the SQLAlchemy async engine."""
        if cls._engine is None:
            settings = get_settings()
            db_url = settings.storage.postgres_async_url
            logger.info(f"Initializing PostgreSQL async engine at {settings.storage.postgres_host}:{settings.storage.postgres_port}")
            try:
                cls._engine = create_async_engine(
                    db_url,
                    echo=False,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                )
                cls._sessionmaker = async_sessionmaker(
                    bind=cls._engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                )
            except Exception as err:
                raise DatabaseConnectionError(f"Failed to create PostgreSQL engine: {err}") from err
        return cls._engine

    @classmethod
    @asynccontextmanager
    async def get_session(cls) -> AsyncGenerator[AsyncSession, None]:
        """Provide an asynchronous transactional session scope."""
        if cls._sessionmaker is None:
            cls.get_engine()
        assert cls._sessionmaker is not None

        async with cls._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception as err:
                await session.rollback()
                logger.error(f"Database session error: {err}")
                raise
            finally:
                await session.close()

    @classmethod
    async def check_health(cls) -> bool:
        """Ping database to confirm connectivity."""
        try:
            async with cls.get_session() as session:
                res = await session.execute(text("SELECT 1;"))
                return res.scalar() == 1
        except Exception as err:
            logger.error(f"PostgreSQL health check failed: {err}")
            return False

    @classmethod
    async def initialize_schema(cls) -> None:
        """Ensure pgvector extension exists and create parent documents table for hierarchical RAG."""
        logger.info("Verifying pgvector extension and initializing RAG database schema...")
        try:
            async with cls.get_session() as session:
                # 1. Enable pgvector extension
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

                # 2. Create Parent Documents Table for Hierarchical (Small-to-Big) Chunking
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS parent_documents (
                        parent_id VARCHAR(128) PRIMARY KEY,
                        doc_id VARCHAR(128) NOT NULL,
                        doc_hash_sha256 VARCHAR(64) NOT NULL,
                        source_name VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_parent_docs_hash ON parent_documents(doc_hash_sha256);
                    CREATE INDEX IF NOT EXISTS idx_parent_docs_source ON parent_documents(source_name);
                """))

                # 3. Create Conversational Memory Table for Multi-turn Chat
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id BIGSERIAL PRIMARY KEY,
                        session_id VARCHAR(128) NOT NULL,
                        role VARCHAR(32) NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);
                """))
            logger.info("✔ PostgreSQL schema and pgvector extension initialized successfully")
        except Exception as err:
            raise DatabaseConnectionError(f"Failed to initialize PostgreSQL schema: {err}") from err

    @classmethod
    async def close(cls) -> None:
        """Dispose the engine and close active connection pool."""
        if cls._engine is not None:
            await cls._engine.dispose()
            cls._engine = None
            cls._sessionmaker = None
            logger.info("PostgreSQL connection pool disposed")
