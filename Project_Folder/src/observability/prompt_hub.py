"""Strict Remote Prompt Hub Manager.

CRITICAL GOVERNANCE POLICY:
- The remote prompt hub (Langfuse / LangSmith Hub) is the SINGLE SOURCE OF TRUTH.
- NO LOCAL FALLBACK is permitted.
- If a prompt cannot be retrieved, is missing, or the hub is unreachable,
  the system fails fast with PromptNotFoundError or PromptHubError.
- Includes an in-memory TTL cache to minimize network latency while preserving live prompt synchronization.
"""

from datetime import datetime, timezone
import time
from typing import Any
from langchain_core.prompts import ChatPromptTemplate

from src.core.config import get_settings
from src.core.exceptions import PromptHubError, PromptNotFoundError
from src.core.logger import get_logger

logger = get_logger(__name__)


class CachedPromptEntry:
    """Internal cache entry with timestamp validation."""

    def __init__(self, prompt: ChatPromptTemplate, fetched_at: float) -> None:
        self.prompt = prompt
        self.fetched_at = fetched_at

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.fetched_at) > ttl_seconds


class StrictPromptHub:
    """Manager for fetching prompts strictly from remote registries with zero local fallback."""

    _cache: dict[str, CachedPromptEntry] = {}

    @classmethod
    async def get_prompt(
        cls,
        name: str | None = None,
        version: str | None = None,
        force_refresh: bool = False,
    ) -> ChatPromptTemplate:
        """Fetch prompt from remote hub. Fails immediately if not found.

        Args:
            name: Name of the prompt in remote registry (defaults to config setting).
            version: Specific version or label (e.g. 'production', 'latest', or version int).
            force_refresh: Bypass in-memory cache to guarantee fresh remote fetch.

        Raises:
            PromptNotFoundError: If the prompt is missing from the remote registry.
            PromptHubError: If connection to the remote registry fails.
        """
        settings = get_settings()
        prompt_name = name or settings.observability.remote_prompt_name
        prompt_version = version or settings.observability.remote_prompt_version
        cache_key = f"{prompt_name}:{prompt_version}"

        # 1. Check in-memory cache
        if not force_refresh and cache_key in cls._cache:
            entry = cls._cache[cache_key]
            if not entry.is_expired(settings.observability.prompt_cache_ttl_seconds):
                logger.debug(f"Prompt cache hit for '{cache_key}'")
                return entry.prompt

        logger.info(
            f"Fetching prompt strictly from remote hub: name='{prompt_name}', version='{prompt_version}' "
            f"(Backend: {settings.observability.backend})"
        )

        prompt_template: ChatPromptTemplate | None = None

        # 2. Try Langfuse Prompt Registry
        if settings.observability.backend == "langfuse":
            prompt_template = await cls._fetch_from_langfuse(prompt_name, prompt_version)

        # 3. Try LangChain Hub / LangSmith
        elif settings.observability.backend in ("langsmith", "langchain"):
            prompt_template = await cls._fetch_from_langchain_hub(prompt_name, prompt_version)

        else:
            raise PromptHubError(
                f"Unsupported or unconfigured prompt hub backend: '{settings.observability.backend}'. "
                f"Strict governance policy mandates a valid remote hub (langfuse or langsmith)."
            )

        if prompt_template is None:
            raise PromptNotFoundError(
                f"Prompt '{prompt_name}' (version='{prompt_version}') could not be found in remote hub. "
                f"STRICT POLICY ENFORCED: Local fallback is prohibited in this enterprise configuration."
            )

        # 4. Cache valid remote prompt
        cls._cache[cache_key] = CachedPromptEntry(prompt=prompt_template, fetched_at=time.time())
        logger.info(f"✔ Remote prompt '{cache_key}' successfully synchronized and cached")
        return prompt_template

    @classmethod
    async def _fetch_from_langfuse(
        cls,
        name: str,
        version: str,
    ) -> ChatPromptTemplate | None:
        """Retrieve compiled prompt from Langfuse Prompt Registry."""
        settings = get_settings()
        pub_key = settings.observability.langfuse_public_key
        sec_key = settings.observability.langfuse_secret_key

        if not pub_key or not sec_key:
            raise PromptHubError(
                "Langfuse API credentials (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY) are missing. "
                "Cannot contact remote prompt hub."
            )

        try:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=pub_key.get_secret_value(),
                secret_key=sec_key.get_secret_value(),
                host=settings.observability.langfuse_host,
            )
            # Fetch prompt object from Langfuse
            lf_prompt = client.get_prompt(name=name, label=version if version != "latest" else None)
            if not lf_prompt:
                return None

            # Convert Langfuse prompt to LangChain ChatPromptTemplate
            compiled_messages = lf_prompt.get_langchain_prompt()
            if isinstance(compiled_messages, ChatPromptTemplate):
                return compiled_messages
            elif isinstance(compiled_messages, str):
                return ChatPromptTemplate.from_template(compiled_messages)
            return None

        except Exception as err:
            logger.error(f"Failed to fetch prompt from Langfuse hub: {err}")
            raise PromptHubError(f"Langfuse Remote Prompt Hub error: {err}") from err

    @classmethod
    async def _fetch_from_langchain_hub(
        cls,
        name: str,
        version: str,
    ) -> ChatPromptTemplate | None:
        """Retrieve compiled prompt from LangChain Hub (langchain.hub)."""
        settings = get_settings()
        api_key = settings.observability.langsmith_api_key

        if not api_key:
            raise PromptHubError(
                "LangChain API key (LANGCHAIN_API_KEY) is missing. Cannot contact LangChain Hub."
            )

        try:
            from langchain import hub
            hub_handle = f"{name}:{version}" if version else name
            pulled_prompt = hub.pull(hub_handle)
            if isinstance(pulled_prompt, ChatPromptTemplate):
                return pulled_prompt
            elif hasattr(pulled_prompt, "messages"):
                return pulled_prompt
            return None
        except Exception as err:
            logger.error(f"Failed to fetch prompt from LangChain Hub: {err}")
            raise PromptHubError(f"LangChain Hub error: {err}") from err

    @classmethod
    def invalidate_cache(cls) -> None:
        """Clear cached prompt templates to force remote re-fetch."""
        cls._cache.clear()
        logger.info("Prompt hub local cache invalidated")
