"""LangSmith tracing adapter and feedback exporter."""

import os
from typing import Any

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class LangSmithAdapter:
    """Manager for LangSmith tracing, project tagging, and runs."""

    @classmethod
    def setup_environment(cls) -> None:
        """Inject LangSmith environment variables into active process."""
        settings = get_settings()
        if settings.observability.backend == "langsmith" or settings.observability.langsmith_tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.observability.langsmith_project
            if settings.observability.langsmith_api_key:
                os.environ["LANGCHAIN_API_KEY"] = settings.observability.langsmith_api_key.get_secret_value()
            logger.info(f"LangSmith tracing enabled for project '{settings.observability.langsmith_project}'")

    @classmethod
    def get_tracer(cls, project_name: str | None = None) -> Any | None:
        """Get native LangChainTracer callback."""
        settings = get_settings()
        if settings.observability.backend != "langsmith" and not settings.observability.langsmith_tracing:
            return None

        try:
            from langchain_core.tracers import LangChainTracer
            proj = project_name or settings.observability.langsmith_project
            return LangChainTracer(project_name=proj)
        except Exception as err:
            logger.warning(f"Failed to create LangSmith tracer: {err}")
            return None
