"""Step 11: Output Guardrail, Hallucination Self-Reflection & PII Masking.

Evaluates response groundedness against retrieved context. If hallucinated, triggers
a self-correction retry loop back to Step 9. Masks sensitive PII before final delivery.
"""

import re
from typing import Any

from src.agent.state import AgentState
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

# PII regex patterns
PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
]


def mask_pii(text: str) -> str:
    """Mask common personally identifiable information."""
    sanitized = text
    for pattern, repl in PII_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


def step11_guardrail_output(state: AgentState) -> dict[str, Any]:
    """Execute Step 11: Output safety, hallucination check, and PII redaction."""
    tracker = state["tracker"]
    tracker.start_span("step11_guardrail_output")

    settings = get_settings()
    response = state.get("llm_response", "")
    context = state.get("context_text", "")
    reflection_count = state.get("reflection_count", 0)

    is_hallucination = False
    critique = None

    # Check for obvious ungrounded certainty if context is sparse
    if settings.agent.enable_hallucination_check and reflection_count < settings.agent.max_reflection_retries:
        # Fast hallucination heuristic: response asserts specific numbers or names not in context
        if len(context) > 0 and len(response) > 50:
            # Check if key capitalized proper nouns in response are missing from context
            resp_words = set(re.findall(r"\b[A-Z][a-z]{3,}\b", response))
            ctx_words = set(re.findall(r"\b[A-Z][a-z]{3,}\b", context))
            missing = resp_words - ctx_words
            # If large proportion of entities are invented
            if len(missing) > 6:
                is_hallucination = True
                critique = f"Response mentions entities ({', '.join(list(missing)[:3])}) not present in the context."
                logger.warning(f"Self-RAG check triggered hallucination reflection: {critique}")

    # Redact PII from final output
    cleaned_response = mask_pii(response)

    tracker.end_span(
        "step11_guardrail_output",
        {
            "is_hallucination": is_hallucination,
            "reflection_count": reflection_count,
        },
    )

    # Finalize telemetry dictionary
    telemetry = tracker.finalize()

    return {
        "llm_response": cleaned_response,
        "is_hallucination": is_hallucination,
        "reflection_count": reflection_count + (1 if is_hallucination else 0),
        "reflection_critique": critique,
        "telemetry": telemetry,
    }
