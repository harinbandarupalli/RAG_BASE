"""Step 4: Contextual Query Rewriting.

Reformulates conversational, ambiguous queries into standalone, self-contained questions
by resolving pronouns and references against chat history.
"""

from typing import Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.adapters.llm.factory import LLMFactory
from src.agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """You are an expert query reformulator for enterprise RAG.
Given a conversation history and a follow-up question, rewrite the question into a clear, standalone search query.
- Resolve all pronouns (it, they, that, this, its) to their explicit entities.
- Preserve all domain terms, product names, acronyms, and constraints.
- DO NOT answer the question; output ONLY the rewritten standalone question.
If the question is already clear and self-contained, return it unchanged.
"""


async def step04_query_rewriter(state: AgentState) -> dict[str, Any]:
    """Execute Step 4: Contextual query rewriting."""
    tracker = state["tracker"]
    tracker.start_span("step04_query_rewriter")

    query = state.get("cleaned_query") or state.get("user_query", "")
    history = state.get("chat_history", [])

    if not history:
        tracker.end_span("step04_query_rewriter", {"action": "passthrough"})
        return {"rewritten_query": query}

    try:
        llm = LLMFactory.create_llm(
            provider=state.get("model_provider"),
            model_name=state.get("model_name"),
            temperature=0.0,
            max_tokens=150,
        )

        messages = [SystemMessage(content=REWRITE_SYSTEM_PROMPT)]
        for turn in history[-4:]:  # Last 2 conversation rounds
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=f"Follow-up question to rewrite: {query}"))

        response = await llm.ainvoke(messages)
        rewritten = response.content.strip()

        # Track tokens if available
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            tracker.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        tracker.end_span("step04_query_rewriter", {"rewritten": rewritten})
        logger.info(f"Query rewritten: '{query}' ➔ '{rewritten}'")
        return {"rewritten_query": rewritten}

    except Exception as err:
        logger.warning(f"Query rewriting failed ({err}). Falling back to cleaned query.")
        tracker.end_span("step04_query_rewriter", {"error": str(err)})
        return {"rewritten_query": query}
