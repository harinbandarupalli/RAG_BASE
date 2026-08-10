"""Compiled LangGraph State Machine for the 12-Step RAG Agent Pipeline."""

from typing import Any
import uuid

from langgraph.graph import END, StateGraph

from src.agent.nodes.step01_cache import step01_cache
from src.agent.nodes.step01_guardrail_input import step01_guardrail_input
from src.agent.nodes.step02_intent_router import step02_intent_router
from src.agent.nodes.step03_query_cleaner import step03_query_cleaner
from src.agent.nodes.step04_query_rewriter import step04_query_rewriter
from src.agent.nodes.step05_decomposer import step05_decomposer
from src.agent.nodes.step06_parallel_search import step06_parallel_search
from src.agent.nodes.step07_hybrid_search import step07_hybrid_search
from src.agent.nodes.step08_confidence_gate import step08_confidence_gate
from src.agent.nodes.step08_reranker import step08_reranker
from src.agent.nodes.step09_llm_processor import step09_llm_processor
from src.agent.nodes.step10_synthesizer import step10_synthesizer
from src.agent.nodes.step11_guardrail_output import step11_guardrail_output
from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger
from src.observability.tracker import PipelineRunTracker

logger = get_logger(__name__)


def route_after_input_guard(state: AgentState) -> str:
    return END if state.get("is_refusal") else "step01_cache"


def route_after_cache(state: AgentState) -> str:
    return END if state.get("is_cached") else "step02_intent_router"


def route_after_intent(state: AgentState) -> str:
    return "step03_query_cleaner" if state.get("intent") == "rag_query" else END


def route_after_confidence(state: AgentState) -> str:
    return END if state.get("is_refusal") else "step09_llm_processor"


def route_after_output_guard(state: AgentState) -> str:
    settings = get_settings()
    if state.get("is_hallucination") and state.get("reflection_count", 0) <= settings.agent.max_reflection_retries:
        logger.info("Routing back to Step 9 for self-reflection re-generation")
        return "step09_llm_processor"
    return END


def create_agent_graph() -> Any:
    """Build and compile the LangGraph 12-Step RAG pipeline."""
    workflow = StateGraph(AgentState)

    # Add all 12 pipeline nodes
    workflow.add_node("step01_guardrail_input", step01_guardrail_input)
    workflow.add_node("step01_cache", step01_cache)
    workflow.add_node("step02_intent_router", step02_intent_router)
    workflow.add_node("step03_query_cleaner", step03_query_cleaner)
    workflow.add_node("step04_query_rewriter", step04_query_rewriter)
    workflow.add_node("step05_decomposer", step05_decomposer)
    workflow.add_node("step06_parallel_search", step06_parallel_search)
    workflow.add_node("step07_hybrid_search", step07_hybrid_search)
    workflow.add_node("step08_reranker", step08_reranker)
    workflow.add_node("step08_confidence_gate", step08_confidence_gate)
    workflow.add_node("step09_llm_processor", step09_llm_processor)
    workflow.add_node("step10_synthesizer", step10_synthesizer)
    workflow.add_node("step11_guardrail_output", step11_guardrail_output)

    # Set Entry Point
    workflow.set_entry_point("step01_guardrail_input")

    # Add Directed and Conditional Edges
    workflow.add_conditional_edges("step01_guardrail_input", route_after_input_guard)
    workflow.add_conditional_edges("step01_cache", route_after_cache)
    workflow.add_conditional_edges("step02_intent_router", route_after_intent)
    workflow.add_edge("step03_query_cleaner", "step04_query_rewriter")
    workflow.add_edge("step04_query_rewriter", "step05_decomposer")
    workflow.add_edge("step05_decomposer", "step06_parallel_search")
    workflow.add_edge("step06_parallel_search", "step07_hybrid_search")
    workflow.add_edge("step07_hybrid_search", "step08_reranker")
    workflow.add_edge("step08_reranker", "step08_confidence_gate")
    workflow.add_conditional_edges("step08_confidence_gate", route_after_confidence)
    workflow.add_edge("step09_llm_processor", "step10_synthesizer")
    workflow.add_edge("step10_synthesizer", "step11_guardrail_output")
    workflow.add_conditional_edges("step11_guardrail_output", route_after_output_guard)

    return workflow.compile()


# Singleton compiled graph instance
compiled_rag_graph = create_agent_graph()


async def execute_rag_pipeline(
    user_query: str,
    session_id: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> AgentState:
    """Convenience async runner for the compiled LangGraph pipeline."""
    sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
    trace_id = f"trc_{uuid.uuid4().hex[:12]}"

    settings = get_settings()
    active_model = model_name or settings.llm.default_model
    tracker = PipelineRunTracker(trace_id=trace_id, model_name=active_model)

    initial_state: AgentState = {
        "session_id": sess_id,
        "trace_id": trace_id,
        "tracker": tracker,
        "user_query": user_query,
        "chat_history": chat_history or [],
        "cleaned_query": "",
        "rewritten_query": "",
        "sub_queries": [],
        "intent": "rag_query",
        "raw_retrieved_docs": [],
        "reranked_docs": [],
        "context_text": "",
        "model_provider": model_provider,
        "model_name": active_model,
        "temperature": temperature,
        "llm_response": "",
        "citations": [],
        "confidence_score": 0.0,
        "is_cached": False,
        "is_refusal": False,
        "is_hallucination": False,
        "reflection_count": 0,
        "reflection_critique": None,
        "telemetry": {},
    }

    logger.info(f"▶ Initiating LangGraph RAG Agent Pipeline (trace_id={trace_id})")
    final_state = await compiled_rag_graph.ainvoke(initial_state)
    logger.info(f"✔ Completed LangGraph pipeline in {final_state['telemetry'].get('total_latency_ms')}ms")
    return final_state
