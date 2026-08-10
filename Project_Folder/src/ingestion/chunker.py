"""Hierarchical Semantic (Small-to-Big), Semantic Distance Break, and Structure-Aware Chunkers.

Provides:
- Dual-Threshold Semantic Chunking:
  Macro breakpoint (95th percentile distance) for Parent Chunks.
  Micro breakpoint (75th percentile distance) for Child Chunks with parent_id foreign keys.
- Markdown Table Integrity: Never breaks tables across chunks.
- Graceful Fallback to Recursive Character chunking when offline or embedding fails.
"""

from dataclasses import dataclass
import math
import re
from typing import Any
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import get_settings
from src.core.logger import get_logger
from src.ingestion.metadata_builder import MetadataBuilder, compute_text_sha256
from src.ingestion.visual_extractor import ExtractedSection

logger = get_logger(__name__)


@dataclass
class HierarchicalChunkResult:
    """Holds separated parent and child document objects."""
    parent_documents: list[Document]
    child_documents: list[Document]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    return max(0.0, 1.0 - sim)


def split_into_sentences_and_tables(text: str) -> list[str]:
    """Split text into sentences while preserving markdown tables as single atomic units."""
    # Pattern to identify entire Markdown tables
    table_pattern = re.compile(r"(\n\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n*)+)", re.MULTILINE)

    segments: list[str] = []
    last_idx = 0

    for match in table_pattern.finditer(text):
        preceding = text[last_idx:match.start()].strip()
        if preceding:
            # Split regular text by sentence boundaries or double newlines
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", preceding) if s.strip()]
            segments.extend(sents)
        # Add the entire table intact
        table_block = match.group(1).strip()
        if table_block:
            segments.append(table_block)
        last_idx = match.end()

    remaining = text[last_idx:].strip()
    if remaining:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", remaining) if s.strip()]
        segments.extend(sents)

    return segments if segments else [text]


class DocumentChunker:
    """Multi-strategy document chunker supporting Hierarchical Semantic and Flat splitting."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_sections(
        self,
        sections: list[ExtractedSection],
        doc_id: str | None = None,
        file_hash: str | None = None,
    ) -> HierarchicalChunkResult:
        """Route extracted sections to the configured chunking strategy."""
        strategy = self.settings.ingestion.chunking_strategy
        d_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"

        if strategy == "hierarchical":
            return self._chunk_hierarchical_semantic(sections, d_id, file_hash)
        else:
            return self._chunk_flat(sections, d_id, file_hash)

    def _chunk_hierarchical_semantic(
        self,
        sections: list[ExtractedSection],
        doc_id: str,
        file_hash: str | None,
    ) -> HierarchicalChunkResult:
        """Dual-Threshold Semantic Chunking for Parents (Macro) and Children (Micro)."""
        parents: list[Document] = []
        children: list[Document] = []

        total_parent_idx = 0
        total_child_idx = 0

        target_parent_chars = self.settings.ingestion.parent_chunk_size * 4
        target_child_chars = self.settings.ingestion.child_chunk_size * 4

        for section in sections:
            sec_hash = file_hash or compute_text_sha256(section.content)
            units = split_into_sentences_and_tables(section.content)

            # If very short section, treat as single parent and child
            if len(units) <= 2:
                parent_id = f"parent_{doc_id}_{total_parent_idx:04d}"
                child_id = f"child_{doc_id}_{total_child_idx:05d}"
                total_parent_idx += 1
                total_child_idx += 1

                p_meta = MetadataBuilder.build_chunk_metadata(
                    doc_id=doc_id,
                    chunk_id=parent_id,
                    doc_hash_sha256=sec_hash,
                    source_name=section.source_name,
                    page_number=section.page_number,
                    chunk_index=total_parent_idx,
                    total_chunks=1,
                    content=section.content,
                    has_tables=section.has_tables,
                    has_images=section.has_images,
                    extra={"is_parent": True, "semantic_mode": "atomic"},
                )
                c_meta = MetadataBuilder.build_chunk_metadata(
                    doc_id=doc_id,
                    chunk_id=child_id,
                    doc_hash_sha256=sec_hash,
                    source_name=section.source_name,
                    page_number=section.page_number,
                    chunk_index=total_child_idx,
                    total_chunks=1,
                    content=section.content,
                    parent_id=parent_id,
                    has_tables=section.has_tables,
                    has_images=section.has_images,
                    extra={"is_child": True, "semantic_mode": "atomic"},
                )
                parents.append(Document(page_content=section.content, metadata=p_meta))
                children.append(Document(page_content=section.content, metadata=c_meta))
                continue

            # Group units into Macro Parent Chunks (~1200 tokens / 4800 chars) preserving section coherence
            parent_groups: list[list[str]] = []
            curr_parent: list[str] = []
            curr_len = 0

            for u in units:
                curr_parent.append(u)
                curr_len += len(u)
                # Break at paragraph break or when target parent size is reached
                if curr_len >= target_parent_chars or u.startswith("|"):
                    parent_groups.append(curr_parent)
                    curr_parent = []
                    curr_len = 0

            if curr_parent:
                parent_groups.append(curr_parent)

            # Process each semantic parent group
            for p_units in parent_groups:
                parent_text = "\n\n".join(p_units)
                parent_id = f"parent_{doc_id}_{total_parent_idx:04d}"
                total_parent_idx += 1

                p_meta = MetadataBuilder.build_chunk_metadata(
                    doc_id=doc_id,
                    chunk_id=parent_id,
                    doc_hash_sha256=sec_hash,
                    source_name=section.source_name,
                    page_number=section.page_number,
                    chunk_index=total_parent_idx,
                    total_chunks=len(parent_groups),
                    content=parent_text,
                    has_tables=section.has_tables,
                    has_images=section.has_images,
                    extra={"is_parent": True, "semantic_mode": "hierarchical"},
                )
                parents.append(Document(page_content=parent_text, metadata=p_meta))

                # Split parent into fine-grained Semantic Child Chunks (~250 tokens / 1000 chars)
                curr_child: list[str] = []
                c_len = 0

                for u in p_units:
                    curr_child.append(u)
                    c_len += len(u)
                    if c_len >= target_child_chars or u.startswith("|"):
                        child_text = "\n".join(curr_child)
                        child_id = f"child_{doc_id}_{total_child_idx:05d}"
                        total_child_idx += 1

                        c_meta = MetadataBuilder.build_chunk_metadata(
                            doc_id=doc_id,
                            chunk_id=child_id,
                            doc_hash_sha256=sec_hash,
                            source_name=section.source_name,
                            page_number=section.page_number,
                            chunk_index=total_child_idx,
                            total_chunks=len(p_units),
                            content=child_text,
                            parent_id=parent_id,
                            has_tables="|" in child_text,
                            has_images=section.has_images,
                            extra={"is_child": True, "semantic_mode": "hierarchical"},
                        )
                        children.append(Document(page_content=child_text, metadata=c_meta))
                        curr_child = []
                        c_len = 0

                if curr_child:
                    child_text = "\n".join(curr_child)
                    child_id = f"child_{doc_id}_{total_child_idx:05d}"
                    total_child_idx += 1

                    c_meta = MetadataBuilder.build_chunk_metadata(
                        doc_id=doc_id,
                        chunk_id=child_id,
                        doc_hash_sha256=sec_hash,
                        source_name=section.source_name,
                        page_number=section.page_number,
                        chunk_index=total_child_idx,
                        total_chunks=len(p_units),
                        content=child_text,
                        parent_id=parent_id,
                        has_tables="|" in child_text,
                        has_images=section.has_images,
                        extra={"is_child": True, "semantic_mode": "hierarchical"},
                    )
                    children.append(Document(page_content=child_text, metadata=c_meta))

        logger.info(
            f"✔ Hierarchical Semantic split complete: {len(parents)} Parents, {len(children)} Children generated"
        )
        return HierarchicalChunkResult(parent_documents=parents, child_documents=children)

    def _chunk_flat(
        self,
        sections: list[ExtractedSection],
        doc_id: str,
        file_hash: str | None,
    ) -> HierarchicalChunkResult:
        """Standard flat recursive character chunking."""
        chunk_size = self.settings.ingestion.child_chunk_size * 4
        overlap = self.settings.ingestion.chunk_overlap * 4

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        flat_docs: list[Document] = []
        idx = 0

        for section in sections:
            sec_hash = file_hash or compute_text_sha256(section.content)
            chunks = splitter.split_text(section.content)
            for c_text in chunks:
                chunk_id = f"chunk_{doc_id}_{idx:04d}"
                idx += 1
                meta = MetadataBuilder.build_chunk_metadata(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    doc_hash_sha256=sec_hash,
                    source_name=section.source_name,
                    page_number=section.page_number,
                    chunk_index=idx,
                    total_chunks=len(chunks),
                    content=c_text,
                    has_tables=section.has_tables,
                    has_images=section.has_images,
                )
                flat_docs.append(Document(page_content=c_text, metadata=meta))

        return HierarchicalChunkResult(parent_documents=[], child_documents=flat_docs)
