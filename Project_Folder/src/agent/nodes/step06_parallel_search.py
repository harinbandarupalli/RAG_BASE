"""Step 6: Parallel Concurrent Retrieval.

Executes asynchronous parallel searches across all decomposed sub-queries against both:
1. Dense Vector Index (pgvector HNSW)
2. Sparse Keyword Index (BM25)
"""

import asyncio
from typing import Any
from langchain_core.documents import Document

from src.adapters.vectorstores.pgvector_adapter import PGVectorAdapter
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger
from src.ingestion.bm25_indexer import BM25Indexer

logger = get_logger(__name__)


async def step06_parallel_search(state: AgentState) -> dict[str, Any]:
    """Execute Step 6: Parallel retrieval."""
    tracker = state["tracker"]
    tracker.start_span("step06_parallel_search")

    settings = get_settings()
    sub_queries = state.get("sub_queries") or [state.get("user_query", "")]
    vector_adapter = PGVectorAdapter()

    dense_k = settings.agent.dense_top_k
    sparse_k = settings.agent.sparse_top_k

    async def search_single_query(query_text: str) -> tuple[list[tuple[Document, float]], list[tuple[Document, float]]]:
        """Run dense and sparse search for a single query string."""
        # 1. Dense vector search
        dense_task = vector_adapter.similarity_search_with_score(query_text, k=dense_k)
        # 2. Sparse BM25 search (wrapped in asyncio)
        sparse_task = asyncio.to_thread(BM25Indexer.search, query_text, top_k=sparse_k)

        dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task, return_exceptions=True)

        valid_dense = dense_res if isinstance(dense_res, list) else []
        valid_sparse = sparse_res if isinstance(sparse_res, list) else []
        return valid_dense, valid_sparse

    # Run parallel searches across all sub-queries concurrently
    tasks = [search_single_query(sq) for sq in sub_queries]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    combined_docs: list[Document] = []
    dense_hits_count = 0
    sparse_hits_count = 0

    for res in all_results:
        if isinstance(res, tuple):
            dense_list, sparse_list = res
            dense_hits_count += len(dense_list)
            sparse_hits_count += len(sparse_list)

            # Attach retrieval origin tag
            for doc, score in dense_list:
                doc.metadata["retrieval_method"] = "dense"
                doc.metadata["dense_score"] = float(score)
                combined_docs.append(doc)

            for doc, score in sparse_list:
                doc.metadata["retrieval_method"] = "sparse"
                doc.metadata["bm25_score"] = float(score)
                combined_docs.append(doc)

    tracker.end_span(
        "step06_parallel_search",
        {
            "sub_queries_count": len(sub_queries),
            "total_candidates": len(combined_docs),
            "dense_hits": dense_hits_count,
            "sparse_hits": sparse_hits_count,
        },
    )

    logger.info(
        f"Parallel search completed: {len(combined_docs)} total candidates "
        f"({dense_hits_count} dense, {sparse_hits_count} sparse)"
    )
    return {"raw_retrieved_docs": combined_docs}
