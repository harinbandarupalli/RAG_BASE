"""Step 5: Query Decomposition into Atomic Sub-Queries.

Deconstructs multi-part, comparative, or complex questions into distinct, atomic sub-queries
for concurrent parallel retrieval across Dense and Sparse indexes.
"""

from typing import Any
import re
from langchain_core.messages import HumanMessage, SystemMessage

from src.adapters.llm.factory import LLMFactory
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

DECOMPOSE_SYSTEM_PROMPT = """You are an expert query decomposition assistant for enterprise RAG.
Given a complex, comparative, or multi-part question, break it down into 2 to 3 independent, atomic search sub-queries.
Output each sub-query on a new line starting with a dash (-).
If the question is already atomic and simple, return only the original query on a single line starting with (-).
DO NOT answer the questions. Only output the sub-queries.
"""


async def step05_decomposer(state: AgentState) -> dict[str, Any]:
    """Execute Step 5: Query decomposition."""
    tracker = state["tracker"]
    tracker.start_span("step05_decomposer")

    settings = get_settings()
    max_queries = settings.agent.max_sub_queries
    query = state.get("rewritten_query") or state.get("cleaned_query", "")

    # Quick heuristic check: simple short queries don't need LLM decomposition
    words = query.split()
    comparative_markers = ["compare", "difference", "versus", "vs", "both", "and also", "pros and cons"]
    is_complex = len(words) > 8 and any(m in query.lower() for m in comparative_markers)

    if not is_complex:
        tracker.end_span("step05_decomposer", {"sub_queries_count": 1})
        return {"sub_queries": [query]}

    try:
        llm = LLMFactory.create_llm(
            provider=state.get("model_provider"),
            model_name=state.get("model_name"),
            temperature=0.0,
            max_tokens=200,
        )

        messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=f"Question to decompose: {query}"),
        ]

        response = await llm.ainvoke(messages)
        lines = [line.strip().lstrip("-").strip() for line in response.content.splitlines() if line.strip()]
        sub_queries = [line for line in lines if len(line) > 3][:max_queries]

        if query not in sub_queries:
            sub_queries.insert(0, query)

        # Track tokens
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            tracker.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        tracker.end_span("step05_decomposer", {"sub_queries": sub_queries})
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        return {"sub_queries": sub_queries}

    except Exception as err:
        logger.warning(f"Query decomposition failed ({err}). Using single query.")
        tracker.end_span("step05_decomposer", {"error": str(err)})
        return {"sub_queries": [query]}
