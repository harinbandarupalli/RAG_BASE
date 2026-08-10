"""Langfuse tracing adapter and user feedback collector."""

from typing import Any
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class LangfuseAdapter:
    """Manager for Langfuse callback handlers, tracing, and user feedback capture."""

    _client = None

    @classmethod
    def get_client(cls):
        """Lazy load Langfuse client."""
        if cls._client is None:
            settings = get_settings()
            pub_key = settings.observability.langfuse_public_key
            sec_key = settings.observability.langfuse_secret_key
            if pub_key and sec_key:
                try:
                    from langfuse import Langfuse
                    cls._client = Langfuse(
                        public_key=pub_key.get_secret_value(),
                        secret_key=sec_key.get_secret_value(),
                        host=settings.observability.langfuse_host,
                    )
                    logger.info("Langfuse client initialized successfully")
                except Exception as err:
                    logger.warning(f"Could not initialize Langfuse client: {err}")
        return cls._client

    @classmethod
    def get_callback_handler(
        cls,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Any | None:
        """Create a Langfuse CallbackHandler for LangChain / LangGraph execution."""
        settings = get_settings()
        if settings.observability.backend != "langfuse":
            return None

        pub_key = settings.observability.langfuse_public_key
        sec_key = settings.observability.langfuse_secret_key

        if not pub_key or not sec_key:
            return None

        try:
            from langfuse.callback import CallbackHandler
            return CallbackHandler(
                public_key=pub_key.get_secret_value(),
                secret_key=sec_key.get_secret_value(),
                host=settings.observability.langfuse_host,
                session_id=session_id,
                user_id=user_id,
                tags=tags or ["enterprise-rag"],
            )
        except Exception as err:
            logger.warning(f"Failed to create Langfuse callback handler: {err}")
            return None

    @classmethod
    async def record_user_feedback(
        cls,
        trace_id: str,
        score: float,
        comment: str | None = None,
    ) -> bool:
        """Record explicit user feedback (e.g. thumbs up=1.0, thumbs down=0.0) into Langfuse."""
        client = cls.get_client()
        if not client:
            logger.warning("Cannot record feedback: Langfuse client not available")
            return False

        try:
            client.score(
                trace_id=trace_id,
                name="user_feedback",
                value=score,
                comment=comment,
            )
            logger.info(f"✔ User feedback logged to Langfuse: trace_id={trace_id}, score={score}")
            return True
        except Exception as err:
            logger.error(f"Error logging user feedback to Langfuse: {err}")
            return False
