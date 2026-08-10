"""Domain-specific exceptions for the Enterprise RAG Starter Kit.

Follows clean architecture principles: all system errors derive from RAGBaseException.
Enforces strict fail-fast policies (e.g. PromptHubError fails without silent fallback).
"""

from typing import Any


class RAGBaseException(Exception):
    """Base exception for all errors raised within the RAG application."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ==============================================================================
# Strict Remote Prompt Hub Exceptions (FAIL-FAST POLICY)
# ==============================================================================

class PromptHubError(RAGBaseException):
    """Raised when communication with the remote prompt registry fails.

    Under our strict governance policy, NO local fallback is allowed.
    The pipeline must halt immediately to prevent untracked prompt drift.
    """
    pass


class PromptNotFoundError(PromptHubError):
    """Raised when a specified prompt template or version is not found in the remote hub."""
    pass


# ==============================================================================
# Guardrails & Safety Exceptions
# ==============================================================================

class GuardrailViolationError(RAGBaseException):
    """Raised when an input or output prompt violates safety/moderation policies."""
    pass


class PromptInjectionDetectedError(GuardrailViolationError):
    """Raised when potential adversarial prompt injection is detected in input."""
    pass


class HallucinationDetectedError(GuardrailViolationError):
    """Raised when generated response contains claims unsupported by retrieved context."""
    pass


# ==============================================================================
# Supervisor, Budget & Resource Exceptions
# ==============================================================================

class TaskSupervisorError(RAGBaseException):
    """Base exception for task execution and resource management errors."""
    pass


class BudgetExceededError(TaskSupervisorError):
    """Raised when a generation exceeds the maximum allowed token or dollar budget."""
    pass


class TaskTimeoutError(TaskSupervisorError):
    """Raised when a pipeline run exceeds the maximum configured execution timeout."""
    pass


class TaskCancelledError(TaskSupervisorError):
    """Raised when a client terminates the pipeline via WebSocket or REST kill-switch."""
    pass


# ==============================================================================
# Adapter & Infrastructure Exceptions
# ==============================================================================

class AdapterError(RAGBaseException):
    """Base exception for external adapter failures (LLMs, Embeddings, Vector Stores)."""
    pass


class LLMProviderError(AdapterError):
    """Raised when an external LLM vendor returns an error or rate limit (429/500)."""
    pass


class VectorStoreError(AdapterError):
    """Raised when vector store operations (indexing, searching) fail."""
    pass


class DatabaseConnectionError(AdapterError):
    """Raised when connection to PostgreSQL or Redis fails."""
    pass


class DocumentParsingError(RAGBaseException):
    """Raised when a document cannot be parsed or OCR fails during ingestion."""
    pass
