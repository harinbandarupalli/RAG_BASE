"""Arize Phoenix open-source tracing adapter."""

from typing import Any
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class PhoenixAdapter:
    """Manager for Arize Phoenix self-hosted tracing."""

    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Register Phoenix OpenInference tracer."""
        if cls._initialized:
            return

        settings = get_settings()
        if settings.observability.backend != "phoenix":
            return

        try:
            from phoenix.otel import register
            from openinference.instrumentation.langchain import LangChainInstrumentor

            tracer_provider = register(
                project_name=settings.app.name,
                endpoint=f"{settings.observability.phoenix_endpoint}/v1/traces",
            )
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            cls._initialized = True
            logger.info(f"Arize Phoenix tracing registered at {settings.observability.phoenix_endpoint}")
        except Exception as err:
            logger.warning(f"Could not initialize Arize Phoenix ({err})")
