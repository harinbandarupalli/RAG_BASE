"""Ingestion API endpoints for document upload, visual table preview, and indexing."""

from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.core.logger import get_logger
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.visual_extractor import VisualDocumentExtractor

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Data Ingestion"])


@router.post("/ingest", summary="Upload and Index Document into Vector Store")
async def ingest_document(
    file: UploadFile = File(...),
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Upload document (PDF, DOCX, TXT, MD, CSV) for hierarchical layout extraction and dual indexing."""
    suffix = Path(file.filename or "upload.tmp").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(file.file, temp_file)

    try:
        pipeline = IngestionPipeline()
        result = await pipeline.ingest_file(temp_path, force_reindex=force_reindex)
        result["original_filename"] = file.filename
        return result
    except Exception as err:
        logger.error(f"Ingestion failed for '{file.filename}': {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
    finally:
        if temp_path.exists():
            temp_path.unlink()


@router.post("/ingest/preview", summary="Preview Extracted Layout & Markdown Tables")
async def preview_document(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Inspect extracted sections and markdown tables before committing to the vector database."""
    suffix = Path(file.filename or "upload.tmp").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        shutil.copyfileobj(file.file, temp_file)

    try:
        extractor = VisualDocumentExtractor()
        sections = extractor.extract_document(temp_path)
        preview_sections = [
            {
                "page_number": sec.page_number,
                "has_tables": sec.has_tables,
                "char_count": len(sec.content),
                "snippet": sec.content[:400] + "..." if len(sec.content) > 400 else sec.content,
            }
            for sec in sections
        ]
        return {
            "file_name": file.filename,
            "total_pages": len(sections),
            "tables_detected": sum(1 for s in sections if s.has_tables),
            "sections": preview_sections,
        }
    except Exception as err:
        logger.error(f"Preview failed for '{file.filename}': {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
    finally:
        if temp_path.exists():
            temp_path.unlink()
