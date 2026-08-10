"""Step 1: Input Guardrail & Moderation.

Inspects user input for adversarial prompt injections, system prompt leaks,
jailbreak attempts, and safety violations.
"""

import re
from typing import Any

from src.agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+directives", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(DAN|developer|jailbreak)\s+mode", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|initial\s+instructions)", re.IGNORECASE),
    re.compile(r"print\s+(your\s+)?(base\s+instructions|system\s+message)", re.IGNORECASE),
    re.compile(r"<system>.*?</system>", re.IGNORECASE),
]


def step01_guardrail_input(state: AgentState) -> dict[str, Any]:
    """Execute Step 1: Input safety validation."""
    tracker = state["tracker"]
    tracker.start_span("step01_guardrail_input")

    query = state.get("user_query", "")

    # 1. Check prompt injection heuristics
    for pattern in INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning(f"🚨 Input Guardrail Alert: Adversarial pattern detected in query '{query[:40]}...'")
            tracker.end_span("step01_guardrail_input", {"violation": "prompt_injection"})
            return {
                "is_refusal": True,
                "intent": "refusal",
                "llm_response": "I cannot fulfill this request as it violates safety guidelines.",
                "confidence_score": 0.0,
                "citations": [],
            }

    # 2. Check length guards
    if len(query.strip()) < 2:
        tracker.end_span("step01_guardrail_input", {"violation": "too_short"})
        return {
            "is_refusal": True,
            "intent": "refusal",
            "llm_response": "Please enter a valid question.",
            "confidence_score": 0.0,
            "citations": [],
        }

    tracker.end_span("step01_guardrail_input", {"status": "passed"})
    return {"is_refusal": False}
