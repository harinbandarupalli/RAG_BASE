"""Step 2: Intent Categorization & Query Routing.

Classifies incoming query intent to route appropriately:
- 'rag_query': Requires retrieval from knowledge base (default).
- 'chit_chat': Conversational greetings or casual remarks.
- 'direct_reasoning': Self-contained mathematical or logical reasoning.
- 'refusal': Out-of-bounds or invalid.
"""

from typing import Any
import re

from src.agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

GREETING_PATTERNS = [
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b",
    r"^(thanks|thank\s+you|appreciate\s+it)\b",
    r"^(who\s+are\s+you|what\s+can\s+you\s+do)\b",
]


def step02_intent_router(state: AgentState) -> dict[str, Any]:
    """Execute Step 2: Intent classification."""
    tracker = state["tracker"]
    tracker.start_span("step02_intent_router")

    query = state.get("user_query", "").strip().lower()

    # Fast pattern classification for chit-chat
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, query):
            tracker.end_span("step02_intent_router", {"intent": "chit_chat"})
            return {
                "intent": "chit_chat",
                "llm_response": "Hello! I am your Enterprise AI Assistant. How can I help you explore your knowledge base today?",
                "citations": [],
                "confidence_score": 1.0,
            }

    # Default to RAG query
    tracker.end_span("step02_intent_router", {"intent": "rag_query"})
    return {"intent": "rag_query"}
