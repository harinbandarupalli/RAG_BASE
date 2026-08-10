"""Unit tests for Strict Remote Prompt Hub fail-fast policy (NO local fallback)."""

import pytest
from src.core.exceptions import PromptHubError
from src.observability.prompt_hub import StrictPromptHub


@pytest.mark.asyncio
async def test_prompt_hub_fails_fast_when_credentials_missing():
    # Attempting to fetch a prompt with unconfigured credentials MUST raise PromptHubError
    # Under strict governance, local fallback is prohibited!
    with pytest.raises(PromptHubError):
        await StrictPromptHub.get_prompt(name="non_existent_prompt", version="v1", force_refresh=True)
