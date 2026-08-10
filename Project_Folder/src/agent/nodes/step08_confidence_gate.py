"""Step 8.5: Adaptive Retrieval Confidence Gate.

Evaluates highest reranker relevance score against threshold. If no documents meet
the minimum confidence threshold, skips LLM generation and gracefully refuses
to prevent hallucinations on out-of-domain queries.
"""

from typing import Any
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def step08_confidence_gate(state: AgentState) -> dict[str, Any]:
    """Execute Step 8.5: Confidence threshold evaluation."""
    tracker = state["tracker"]
    tracker.start_span("step08_confidence_gate")

    settings = get_settings()
    min_threshold = settings.agent.reranker.min_relevance_threshold
    reranked = state.get("reranked_docs", [])

    if not reranked:
        tracker.end_span("step08_confidence_gate", {"passed": False, "reason": "no_docs"})
        return {
            "is_refusal": True,
            "llm_response": "I could not find any relevant documents in the knowledge base to answer your question.",
            "citations": [],
            "confidence_score": 0.0,
        }

    top_score = reranked[0][1]
    passed = top_score >= min_threshold

    tracker.end_span(
        "step08_confidence_gate",
        {
            "top_score": round(top_score, 4),
            "min_threshold": min_threshold,
            "passed": passed,
        },
    )

    if not passed:
        logger.info(f"Confidence Gate triggered: top_score={top_score:.4f} < {min_threshold}. Gracefully refusing.")
        return {
            "is_refusal": True,
            "llm_response": (
                "The indexed documents do not contain sufficient information to answer this question accurately. "
                "Please verify your query or upload relevant documentation."
            ),
            "citations": [],
            "confidence_score": round(top_score, 4),
        }

    return {"is_refusal": False, "confidence_score": round(top_score, 4)}
