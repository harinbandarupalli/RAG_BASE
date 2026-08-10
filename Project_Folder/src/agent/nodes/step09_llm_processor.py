"""Step 9: LLM Processing with Strict Remote Prompt Hub & Sandboxed Context.

Retrieves prompt strictly from remote hub (FAIL-FAST policy).
Sandboxes retrieved passages inside XML tags to prevent indirect prompt injection,
and handles agentic self-reflection critique if looping back.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from src.adapters.llm.factory import LLMFactory
from src.agent.state import AgentState
from src.core.logger import get_logger
from src.observability.prompt_hub import StrictPromptHub

logger = get_logger(__name__)


def build_sandboxed_context(reranked_docs: list[Any]) -> str:
    """Format retrieved passages into XML-sandboxed context blocks."""
    blocks = []
    for idx, (doc, score) in enumerate(reranked_docs, start=1):
        src = doc.metadata.get("source_name", "unknown")
        page = doc.metadata.get("page_number", 1)
        cid = doc.metadata.get("chunk_id", f"chunk_{idx}")
        blocks.append(
            f'<document id="{cid}" source="{src}" page="{page}" relevance="{score:.3f}">\n'
            f"{doc.page_content.strip()}\n"
            f"</document>"
        )
    return "\n\n".join(blocks)


async def step09_llm_processor(state: AgentState) -> dict[str, Any]:
    """Execute Step 9: LLM generation."""
    tracker = state["tracker"]
    tracker.start_span("step09_llm_processor")

    query = state.get("rewritten_query") or state.get("user_query", "")
    reranked = state.get("reranked_docs", [])
    context_text = build_sandboxed_context(reranked)

    # 1. Fetch prompt strictly from remote hub (FAIL-FAST, NO local fallback)
    # Under our strict governance policy, this will raise PromptNotFoundError/PromptHubError if missing.
    try:
        remote_prompt = await StrictPromptHub.get_prompt()
        system_content = remote_prompt.format(context=context_text, question=query)
    except Exception as err:
        logger.error(f"Strict prompt hub failure: {err}")
        tracker.end_span("step09_llm_processor", {"error": str(err)})
        raise

    # 2. Append self-correction critique if looping back
    critique = state.get("reflection_critique")
    if critique:
        system_content += f"\n\n[SELF-CORRECTION CRITIQUE]: {critique}\nRegenerate strictly grounded in context."

    llm = LLMFactory.create_llm(
        provider=state.get("model_provider"),
        model_name=state.get("model_name"),
        temperature=state.get("temperature", 0.2),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Context:\n<context>\n{context_text}\n</context>\n\nQuestion: {query}"),
    ]

    response = await llm.ainvoke(messages)
    tracker.record_first_token()

    if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
        usage = response.response_metadata["token_usage"]
        tracker.add_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    tracker.end_span("step09_llm_processor", {"output_chars": len(response.content)})
    return {
        "llm_response": response.content,
        "context_text": context_text,
    }
