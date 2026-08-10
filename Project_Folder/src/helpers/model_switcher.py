"""Dynamic LLM Model Switcher and Provider Registry.

Enables switching LLM providers and models dynamically per request or globally
across OpenAI, Anthropic, Google Gemini, Groq, and Ollama.
"""

from typing import Any
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS_REGISTRY: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "anthropic": ["claude-3-5-sonnet", "claude-3-haiku"],
    "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
    "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "ollama": ["mistral", "llama3", "qwen2.5"],
}


class ModelSwitcher:
    """Manager for dynamic model selection and runtime overrides."""

    _active_provider: str | None = None
    _active_model: str | None = None

    @classmethod
    def list_available_models(cls) -> dict[str, list[str]]:
        """Return the dictionary of supported models grouped by provider."""
        return SUPPORTED_MODELS_REGISTRY

    @classmethod
    def get_active_model(cls) -> dict[str, str]:
        """Return the currently active default provider and model."""
        settings = get_settings()
        return {
            "provider": cls._active_provider or settings.llm.default_provider,
            "model": cls._active_model or settings.llm.default_model,
        }

    @classmethod
    def set_active_model(cls, provider: str, model_name: str) -> dict[str, str]:
        """Update active provider and model globally."""
        provider_norm = provider.lower()
        if provider_norm not in SUPPORTED_MODELS_REGISTRY:
            logger.warning(f"Setting model for unlisted provider '{provider_norm}'")

        cls._active_provider = provider_norm
        cls._active_model = model_name
        logger.info(f"✔ Active default model switched to: {provider_norm}/{model_name}")
        return {"provider": cls._active_provider, "model": cls._active_model}

    @classmethod
    def resolve_model(
        cls,
        requested_provider: str | None = None,
        requested_model: str | None = None,
    ) -> tuple[str, str]:
        """Resolve target provider and model, prioritizing request overrides."""
        active = cls.get_active_model()
        prov = requested_provider or active["provider"]
        model = requested_model or active["model"]
        return prov, model
