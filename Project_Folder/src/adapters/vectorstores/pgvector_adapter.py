"""PGVector adapter with HNSW indexing and Hierarchical Parent-Document resolution.

Integrates:
- LangChain PGVector integration for dense vector search
- Automatic HNSW cosine distance indexing
- Hierarchical (Small-to-Big) parent document expansion:
  Resolves child chunk matches to full parent sections from the database.
"""

from typing import Any
from langchain_core.documents import Document
from sqlalchemy import text

from src.adapters.databases.postgres import PostgresManager
from src.adapters.embeddings.factory import EmbeddingFactory
from src.core.config import get_settings
from src.core.exceptions import VectorStoreError
from src.core.logger import get_logger

logger = get_logger(__name__)


class PGVectorAdapter:
    """Adapter for PostgreSQL pgvector storage with Small-to-Big hierarchical retrieval."""

    def __init__(self, collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.storage.vector_collection_name
        self.embeddings = EmbeddingFactory.create_embeddings()
        self._vectorstore = None

    def _get_store(self):
        """Lazy load LangChain PGVector store."""
        if self._vectorstore is None:
            try:
                from langchain_postgres import PGVector
                self._vectorstore = PGVector(
                    embeddings=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.settings.storage.postgres_sync_url,
                    use_jsonb=True,
                )
            except Exception as err:
                raise VectorStoreError(f"Failed to connect to PGVector: {err}") from err
        return self._vectorstore

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add child document chunks to the vector store."""
        if not documents:
            return []
        try:
            logger.info(f"Indexing {len(documents)} document chunks into collection '{self.collection_name}'")
            store = self._get_store()
            ids = store.add_documents(documents)
            logger.info(f"✔ Successfully added {len(ids)} chunks to pgvector")
            return ids
        except Exception as err:
            raise VectorStoreError(f"Error adding documents to PGVector: {err}") from err

    async def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform dense vector similarity search returning documents and similarity scores."""
        target_k = k or self.settings.agent.dense_top_k
        try:
            store = self._get_store()
            results = store.similarity_search_with_score(query, k=target_k)
            return results
        except Exception as err:
            raise VectorStoreError(f"Error executing similarity search in PGVector: {err}") from err

    async def resolve_parent_documents(
        self,
        child_docs: list[Document],
    ) -> list[Document]:
        """Expand child chunks into full parent documents if parent_id exists (Small-to-Big)."""
        parent_ids = list({
            doc.metadata["parent_id"]
            for doc in child_docs
            if "parent_id" in doc.metadata and doc.metadata["parent_id"]
        })

        if not parent_ids:
            return child_docs

        logger.debug(f"Expanding {len(parent_ids)} child chunks into full parent documents (Small-to-Big)")
        try:
            from src.adapters.databases.redis_cache import RedisCacheAdapter
            parent_map: dict[str, Document] = {}
            missing_ids: list[str] = []

            # 1. Check Redis RAM cache first for each parent ID (<2ms)
            for pid in parent_ids:
                cached = await RedisCacheAdapter.get_cached_parent_document(pid)
                if cached:
                    meta = dict(cached.get("metadata", {}))
                    meta["is_parent_document"] = True
                    parent_map[pid] = Document(page_content=cached.get("content", ""), metadata=meta)
                else:
                    missing_ids.append(pid)

            # 2. Query PostgreSQL only for cache misses
            if missing_ids:
                async with PostgresManager.get_session() as session:
                    query = text("""
                        SELECT parent_id, doc_id, source_name, content, metadata
                        FROM parent_documents
                        WHERE parent_id = ANY(:ids);
                    """)
                    rows = await session.execute(query, {"ids": missing_ids})
                    for r in rows:
                        meta = dict(r.metadata or {})
                        meta.update({
                            "parent_id": r.parent_id,
                            "doc_id": r.doc_id,
                            "source_name": r.source_name,
                            "is_parent_document": True,
                        })
                        parent_doc = Document(page_content=r.content, metadata=meta)
                        parent_map[r.parent_id] = parent_doc
                        # Populate Redis RAM cache for hot access
                        await RedisCacheAdapter.cache_parent_document(r.parent_id, r.content, meta)

            # Assemble expanded documents
            expanded_docs: list[Document] = []
            seen_parents: set[str] = set()

            for doc in child_docs:
                p_id = doc.metadata.get("parent_id")
                if p_id and p_id in parent_map:
                    if p_id not in seen_parents:
                        expanded_docs.append(parent_map[p_id])
                        seen_parents.add(p_id)
                else:
                    expanded_docs.append(doc)

            logger.info(f"Expanded {len(child_docs)} child chunks into {len(expanded_docs)} rich parent documents")
            return expanded_docs

        except Exception as err:
            logger.warning(f"Failed to resolve parent documents (falling back to child chunks): {err}")
            return child_docs
