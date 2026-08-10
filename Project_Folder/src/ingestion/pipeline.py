"""Master Ingestion Pipeline Orchestrator (Path A).

Coordinates:
1. Visual layout & markdown table extraction (VisualDocumentExtractor)
2. SHA-256 Checksum deduplication
3. Hierarchical (Small-to-Big) chunking (DocumentChunker)
4. PostgreSQL parent document persistence (PostgresManager)
5. Dense vector indexing in pgvector (PGVectorAdapter)
6. Sparse keyword indexing in BM25 (BM25Indexer)
"""

import json
from pathlib import Path
import time
from typing import Any
import uuid

from sqlalchemy import text

from src.adapters.databases.postgres import PostgresManager
from src.adapters.vectorstores.pgvector_adapter import PGVectorAdapter
from src.core.config import get_settings
from src.core.logger import get_logger, timed_block
from src.ingestion.bm25_indexer import BM25Indexer
from src.ingestion.chunker import DocumentChunker
from src.ingestion.metadata_builder import compute_file_sha256
from src.ingestion.visual_extractor import VisualDocumentExtractor

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end ingestion and indexing pipeline."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.extractor = VisualDocumentExtractor()
        self.chunker = DocumentChunker()
        self.vector_adapter = PGVectorAdapter()

    async def ingest_file(
        self,
        file_path: Path | str,
        collection_name: str | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """Ingest and index a document file from disk."""
        path = Path(file_path)
        start_time = time.perf_counter()
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        logger.info(f"▶ Commencing ingestion pipeline for '{path.name}' (doc_id={doc_id})")

        # 1. Compute SHA-256 hash for deduplication
        file_hash = compute_file_sha256(path)

        if not force_reindex and self.settings.ingestion.enable_sha256_deduplication:
            is_dup = await self._check_duplicate_hash(file_hash)
            if is_dup:
                logger.warning(f"Duplicate document detected (SHA-256: {file_hash[:12]}...). Skipping re-indexing.")
                return {
                    "status": "skipped",
                    "reason": "duplicate_hash",
                    "file_name": path.name,
                    "file_hash": file_hash,
                    "message": "Document already indexed with identical content.",
                }

        # 2. Visual Extraction (Layout, Markdown Tables, OCR)
        with timed_block("visual_extraction"):
            sections = self.extractor.extract_document(path)

        if not sections:
            return {
                "status": "empty",
                "file_name": path.name,
                "message": "No extractable text or tables found in document.",
            }

        tables_detected = sum(1 for s in sections if s.has_tables)

        # 3. Hierarchical (Small-to-Big) Chunking
        with timed_block("chunking"):
            chunk_res = self.chunker.chunk_sections(sections, doc_id=doc_id, file_hash=file_hash)

        # 4. Store Parent Documents in PostgreSQL
        if chunk_res.parent_documents:
            with timed_block("store_parent_docs"):
                await self._store_parents_in_postgres(chunk_res.parent_documents)

        # 5. Index Child Chunks into pgvector
        with timed_block("index_pgvector"):
            child_docs = chunk_res.child_documents
            vector_ids = await self.vector_adapter.add_documents(child_docs)

        # 6. Index into BM25 Sparse Inverted Index
        with timed_block("index_bm25"):
            BM25Indexer.index_documents(child_docs)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        summary = {
            "status": "success",
            "doc_id": doc_id,
            "file_name": path.name,
            "file_hash": file_hash,
            "pages_extracted": len(sections),
            "tables_detected": tables_detected,
            "parent_documents_stored": len(chunk_res.parent_documents),
            "child_chunks_indexed": len(child_docs),
            "vector_ids_count": len(vector_ids),
            "bm25_total_count": BM25Indexer.count(),
            "elapsed_ms": elapsed_ms,
        }

        logger.info(
            f"✔ Ingestion pipeline completed for '{path.name}' in {elapsed_ms}ms "
            f"({len(child_docs)} child chunks, {len(chunk_res.parent_documents)} parents)"
        )
        return summary

    async def _check_duplicate_hash(self, file_hash: str) -> bool:
        """Query PostgreSQL to check if a document with this hash is already present."""
        try:
            async with PostgresManager.get_session() as session:
                query = text("SELECT 1 FROM parent_documents WHERE doc_hash_sha256 = :h LIMIT 1;")
                res = await session.execute(query, {"h": file_hash})
                return res.scalar() is not None
        except Exception as err:
            logger.warning(f"Could not verify duplicate hash ({err}). Proceeding with ingestion.")
            return False

    async def _store_parents_in_postgres(self, parent_docs: list[Any]) -> None:
        """Insert parent document records into PostgreSQL parent_documents table."""
        async with PostgresManager.get_session() as session:
            insert_stmt = text("""
                INSERT INTO parent_documents (
                    parent_id, doc_id, doc_hash_sha256, source_name, content, metadata
                ) VALUES (
                    :parent_id, :doc_id, :doc_hash_sha256, :source_name, :content, :metadata
                )
                ON CONFLICT (parent_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata;
            """)

            for p in parent_docs:
                await session.execute(
                    insert_stmt,
                    {
                        "parent_id": p.metadata["chunk_id"],
                        "doc_id": p.metadata["doc_id"],
                        "doc_hash_sha256": p.metadata["doc_hash_sha256"],
                        "source_name": p.metadata["source_name"],
                        "content": p.page_content,
                        "metadata": json.dumps(p.metadata),
                    },
                )
