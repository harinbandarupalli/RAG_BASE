"""Metadata Builder & Document Checksum Deduplication.

Enforces structured metadata schema across all chunks and prevents index pollution
via SHA-256 document hashing.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any
import uuid


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hexadecimal hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute SHA-256 hexadecimal hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count (~4 characters per token)."""
    return max(1, len(text) // 4)


class MetadataBuilder:
    """Builder for uniform chunk metadata envelopes."""

    @staticmethod
    def build_chunk_metadata(
        doc_id: str,
        chunk_id: str,
        doc_hash_sha256: str,
        source_name: str,
        page_number: int,
        chunk_index: int,
        total_chunks: int,
        content: str,
        parent_id: str | None = None,
        has_tables: bool = False,
        has_images: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct standard production metadata dictionary."""
        meta: dict[str, Any] = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "doc_hash_sha256": doc_hash_sha256,
            "source_name": source_name,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "token_count": estimate_token_count(content),
            "has_tables": has_tables,
            "has_images": has_images,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if parent_id:
            meta["parent_id"] = parent_id

        if extra:
            meta.update(extra)

        return meta
