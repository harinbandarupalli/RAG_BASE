"""Step 3: Query Cleaning & Normalization.

Sanitizes user input by removing conversational noise, trailing artifacts,
unnecessary preamble, and extra whitespace.
"""

import re
from typing import Any

from src.agent.state import AgentState
from src.core.logger import get_logger

logger = get_logger(__name__)

PREAMBLE_PATTERNS = [
    re.compile(r"^(please\s+tell\s+me|can\s+you\s+(please\s+)?tell\s+me|i\s+want\s+to\s+know|could\s+you\s+explain)\s+", re.IGNORECASE),
    re.compile(r"^(hey\s+assistant|hi\s+bot|search\s+for|find\s+information\s+on)\s+", re.IGNORECASE),
]


def step03_query_cleaner(state: AgentState) -> dict[str, Any]:
    """Execute Step 3: Query cleaning."""
    tracker = state["tracker"]
    tracker.start_span("step03_query_cleaner")

    raw_query = state.get("user_query", "").strip()
    cleaned = raw_query

    # Strip conversational preambles
    for pat in PREAMBLE_PATTERNS:
        cleaned = pat.sub("", cleaned)

    # Normalize excessive spaces and trailing punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"[\?\.!]+$", "", cleaned).strip()

    tracker.end_span("step03_query_cleaner", {"cleaned_query": cleaned})
    logger.debug(f"Query cleaned: '{raw_query}' ➔ '{cleaned}'")

    return {"cleaned_query": cleaned or raw_query}
