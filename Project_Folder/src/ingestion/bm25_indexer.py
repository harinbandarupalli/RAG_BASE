"""BM25 Inverted Keyword Indexer for Sparse Lexical Search.

Complements dense vector embeddings by providing exact keyword, code identifier,
and part-number matching via BM25Okapi scoring.
"""

from typing import Any
import re
from langchain_core.documents import Document

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text for BM25 indexing."""
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    return tokens


class BM25Indexer:
    """In-memory and persisted BM25 sparse index manager."""

    _instance: "BM25Indexer | None" = None
    _documents: list[Document] = []
    _tokenized_corpus: list[list[str]] = []
    _bm25_model = None

    def __new__(cls) -> "BM25Indexer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def index_documents(cls, documents: list[Document]) -> int:
        """Add and index documents in BM25 index."""
        if not documents:
            return 0

        cls._documents.extend(documents)
        new_tokenized = [tokenize_text(doc.page_content) for doc in documents]
        cls._tokenized_corpus.extend(new_tokenized)

        try:
            from rank_bm25 import BM25Okapi
            cls._bm25_model = BM25Okapi(cls._tokenized_corpus)
            logger.info(f"✔ BM25 index rebuilt with {len(cls._documents)} total documents")
            return len(documents)
        except Exception as err:
            logger.warning(f"Could not initialize BM25Okapi: {err}")
            return 0

    @classmethod
    def search(cls, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """Perform sparse BM25 search returning documents and BM25 scores."""
        if cls._bm25_model is None or not cls._documents:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = cls._bm25_model.get_scores(tokens)

        # Pair documents with scores and sort descending
        scored = list(zip(cls._documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter out 0.0 scores
        filtered = [(doc, float(score)) for doc, score in scored if score > 0.0][:top_k]
        logger.debug(f"BM25 search found {len(filtered)} matches for query '{query[:30]}...'")
        return filtered

    @classmethod
    def count(cls) -> int:
        """Return total number of indexed documents in BM25 index."""
        return len(cls._documents)

    @classmethod
    def clear(cls) -> None:
        """Reset the BM25 index."""
        cls._documents.clear()
        cls._tokenized_corpus.clear()
        cls._bm25_model = None
