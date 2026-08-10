"""LLM Provider Factory supporting dynamic provider switching and fallback chaining.

Supports:
- OpenAI (ChatOpenAI)
- Anthropic (ChatAnthropic)
- Google Gemini (ChatGoogleGenerativeAI)
- Groq (ChatGroq)
- Ollama (ChatOllama)
"""

from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import get_settings
from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for instantiating LangChain chat models with unified interfaces."""

    @staticmethod
    def create_llm(
        provider: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a chat model instance based on provider name and runtime overrides."""
        settings = get_settings()
        target_provider = (provider or settings.llm.default_provider).lower()
        target_model = model_name or settings.llm.default_model
        target_temp = temperature if temperature is not None else settings.llm.temperature
        target_max_tokens = max_tokens or settings.llm.max_tokens

        logger.debug(
            f"Instantiating LLM: provider={target_provider}, model={target_model}, temp={target_temp}"
        )

        try:
            if target_provider == "openai":
                from langchain_openai import ChatOpenAI
                api_key = settings.llm.openai_api_key
                return ChatOpenAI(
                    model=target_model,
                    temperature=target_temp,
                    max_tokens=target_max_tokens,
                    api_key=api_key.get_secret_value() if api_key else None,
                    **kwargs,
                )

            elif target_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                api_key = settings.llm.anthropic_api_key
                return ChatAnthropic(
                    model_name=target_model,
                    temperature=target_temp,
                    max_tokens=target_max_tokens,
                    api_key=api_key.get_secret_value() if api_key else None,
                    **kwargs,
                )

            elif target_provider in ("gemini", "google"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = settings.llm.google_api_key
                return ChatGoogleGenerativeAI(
                    model=target_model,
                    temperature=target_temp,
                    max_output_tokens=target_max_tokens,
                    google_api_key=api_key.get_secret_value() if api_key else None,
                    **kwargs,
                )

            elif target_provider == "groq":
                from langchain_groq import ChatGroq
                api_key = settings.llm.groq_api_key
                return ChatGroq(
                    model_name=target_model,
                    temperature=target_temp,
                    max_tokens=target_max_tokens,
                    api_key=api_key.get_secret_value() if api_key else None,
                    **kwargs,
                )

            elif target_provider == "ollama":
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=target_model,
                    temperature=target_temp,
                    base_url=settings.llm.ollama_base_url,
                    **kwargs,
                )

            else:
                raise LLMProviderError(
                    f"Unsupported LLM provider '{target_provider}'. "
                    f"Supported providers: openai, anthropic, gemini, groq, ollama."
                )

        except ImportError as err:
            raise LLMProviderError(
                f"Missing dependency for provider '{target_provider}': {err}. "
                f"Please ensure provider package is installed."
            ) from err
        except Exception as err:
            raise LLMProviderError(
                f"Failed to instantiate LLM for provider '{target_provider}': {err}"
            ) from err

    @classmethod
    def create_llm_with_fallbacks(
        cls,
        primary_provider: str | None = None,
        primary_model: str | None = None,
        fallback_providers: list[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create primary LLM wrapped with automatic fallback models for high availability."""
        primary = cls.create_llm(provider=primary_provider, model_name=primary_model, **kwargs)

        if not fallback_providers:
            return primary

        fallback_models: list[BaseChatModel] = []
        for fb_prov, fb_model in fallback_providers:
            try:
                fb_llm = cls.create_llm(provider=fb_prov, model_name=fb_model, **kwargs)
                fallback_models.append(fb_llm)
            except Exception as err:
                logger.warning(f"Could not initialize fallback LLM {fb_prov}/{fb_model}: {err}")

        if fallback_models:
            logger.info(f"Configured primary LLM with {len(fallback_models)} fallbacks")
            return primary.with_fallbacks(fallback_models)

        return primary
