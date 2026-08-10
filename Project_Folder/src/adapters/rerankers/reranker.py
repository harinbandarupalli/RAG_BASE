"""Cross-Encoder Reranker adapter supporting FlashRank (local CPU) and Cohere API.

Re-scores retrieved candidate passages against the query using deep cross-attention,
drastically improving ranking precision before context is fed to the LLM.
"""

from typing import Any
from langchain_core.documents import Document

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class RerankerAdapter:
    """Adapter for scoring and re-ranking candidate documents."""

    def __init__(self, provider: str | None = None, model_name: str | None = None) -> None:
        self.settings = get_settings()
        self.provider = (provider or self.settings.agent.reranker_provider).lower()
        self.model_name = model_name or self.settings.agent.reranker_model
        self._ranker = None

    def _get_flashrank(self):
        """Lazy load FlashRank ranker."""
        if self._ranker is None:
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name, cache_dir="/tmp/flashrank")
            except Exception as err:
                logger.warning(f"Could not initialize FlashRank ({err}). Reranking will use fallback.")
                return None
        return self._ranker

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Rerank documents against the query, returning top-N documents with relevance scores."""
        if not documents:
            return []

        limit = top_n or self.settings.agent.reranker_top_n

        # --- Provider 1: FlashRank (Fast Local CPU Cross-Encoder) ---
        if self.provider == "flashrank":
            ranker = self._get_flashrank()
            if ranker is not None:
                try:
                    from flashrank import RerankRequest
                    passages = [
                        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
                        for idx, doc in enumerate(documents)
                    ]
                    rerank_req = RerankRequest(query=query, passages=passages)
                    results = ranker.rerank(rerank_req)

                    reranked: list[tuple[Document, float]] = []
                    for item in results[:limit]:
                        doc = Document(page_content=item["text"], metadata=item.get("meta", {}))
                        reranked.append((doc, float(item["score"])))

                    logger.debug(f"FlashRank reordered {len(documents)} documents to top {len(reranked)}")
                    return reranked
                except Exception as err:
                    logger.warning(f"FlashRank rerank failed ({err}). Using original document ordering.")

        # --- Provider 2: Cohere Rerank API ---
        elif self.provider == "cohere":
            api_key = self.settings.agent.cohere_api_key
            if api_key:
                try:
                    import cohere
                    co = cohere.Client(api_key=api_key.get_secret_value())
                    texts = [doc.page_content for doc in documents]
                    response = co.rerank(
                        model="rerank-english-v3.0",
                        query=query,
                        documents=texts,
                        top_n=limit,
                    )
                    reranked = [
                        (documents[item.index], float(item.relevance_score))
                        for item in response.results
                    ]
                    return reranked
                except Exception as err:
                    logger.warning(f"Cohere rerank API failed ({err}). Using original document ordering.")

        # --- Fallback: Pass-through first N documents ---
        logger.debug(f"Passing through top {limit} documents without cross-encoder reranking")
        return [(doc, 1.0 - (idx * 0.05)) for idx, doc in enumerate(documents[:limit])]
