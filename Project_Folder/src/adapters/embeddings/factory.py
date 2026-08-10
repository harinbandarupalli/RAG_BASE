"""Embedding Model Factory supporting cloud and local embedding providers.

Supports:
- OpenAI (OpenAIEmbeddings)
- HuggingFace local (HuggingFaceEmbeddings)
- Cohere (CohereEmbeddings)
"""

from typing import Any
from langchain_core.embeddings import Embeddings

from src.core.config import get_settings
from src.core.exceptions import AdapterError
from src.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """Factory for creating LangChain Embeddings instances."""

    @staticmethod
    def create_embeddings(
        provider: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        """Instantiate an embedding model based on provider."""
        settings = get_settings()
        target_provider = (provider or settings.embedding.provider).lower()
        target_model = model_name or settings.embedding.model

        logger.debug(f"Instantiating Embeddings: provider={target_provider}, model={target_model}")

        try:
            if target_provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                api_key = settings.llm.openai_api_key
                return OpenAIEmbeddings(
                    model=target_model,
                    dimensions=settings.embedding.dimensions,
                    api_key=api_key.get_secret_value() if api_key else None,
                    **kwargs,
                )

            elif target_provider in ("huggingface", "local"):
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(
                    model_name=target_model or "all-MiniLM-L6-v2",
                    **kwargs,
                )

            elif target_provider == "cohere":
                from langchain_community.embeddings import CohereEmbeddings
                cohere_key = settings.agent.cohere_api_key
                return CohereEmbeddings(
                    model=target_model or "embed-english-v3.0",
                    cohere_api_key=cohere_key.get_secret_value() if cohere_key else None,
                    **kwargs,
                )

            else:
                raise AdapterError(
                    f"Unsupported embedding provider '{target_provider}'. "
                    f"Supported providers: openai, huggingface, cohere."
                )

        except ImportError as err:
            raise AdapterError(
                f"Missing dependency for embedding provider '{target_provider}': {err}"
            ) from err
        except Exception as err:
            raise AdapterError(
                f"Failed to instantiate embeddings for provider '{target_provider}': {err}"
            ) from err
